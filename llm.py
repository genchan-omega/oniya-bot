"""DeepSeek API の呼び出し（回答の生成、返事の要否の判定、会話の要約）。

プロンプトの文章は prompts/ と profiles/ のファイルにあり、ここではそれを組み立てて渡す。
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass
from typing import Any

from openai import APIStatusError, AsyncOpenAI

from prompts import PROMPTS_DIR, Profile, load
from store import LoggedMessage, Store
from tools import TASK_TOOL_NAMES, TOOL_DEFS, ToolContext, ToolRunner

log = logging.getLogger("oniya-bot.llm")

API_BASE = "https://api.deepseek.com"
API_BETA_BASE = "https://api.deepseek.com/beta"  # 途中で止まった回答の続きを生成する機能（Chat Prefix Completion）用
REPLY_PREVIEW_CHARS = 200  # 返信先の本文として表示する最大文字数
LINE_MAX_CHARS = 1500  # 会話の1発言として渡す最大文字数


@dataclass
class LLMConfig:
    model: str = "deepseek-flash"
    max_tokens: int = 16384
    max_continuations: int = 3
    max_tool_rounds: int = 6
    thinking: bool = True  # 考えてから答えるモード。返事の要否の判定と要約では常にオフ
    web_search: bool = True
    fact_limit: int = 100  # システムの指示に入れる記憶の最大件数
    summary_limit: int = 10  # システムの指示に入れる過去の会話の要約の最大件数


# DeepSeek が内部で使う関数呼び出しの記法（例: <｜DSML｜function_calls>）。通常は tool_calls として返るが、
# 関数を渡していない呼び出しで関数を使おうとすると、本文に文字として出てくることがある。
TOOL_MARKUP = re.compile(r"<[|｜]+\s*DSML\s*[|｜]+")
FINAL_ANSWER_NOTE = (
    "これ以上、関数は使えません。ここまでに得た情報だけで、質問に答えてください。"
    "分からなかった部分は、分からなかったと一言で書いてください。関数呼び出しの記法は書かないでください。"
)


URL_RE = re.compile(r"https?://[^\s\"'<>\\）)」』]+")


def _norm_url(url: str) -> str:
    return url.rstrip("/.,、。").lower()


def filter_sources(text: str, allowed_text: str) -> tuple[str, int]:
    """「<URL>」の形で書かれた参考リンクのうち、検索結果・読んだページ・会話に出てこない URL の行を取り除く。

    モデルが検索せずに、それらしい URL を作ってしまうことがあるため。取り除いた行数も返す。
    """
    allowed = {_norm_url(u) for u in URL_RE.findall(allowed_text)}
    lines, removed = [], 0
    for line in text.splitlines():
        cited = re.findall(r"<(https?://[^>\s]+)>", line)
        if cited and any(_norm_url(u) not in allowed for u in cited):
            removed += 1
            continue
        lines.append(line)
    out = "\n".join(lines)
    # 参考リンクが1つも残らなかった場合は「参考:」の見出しも消す
    out = re.sub(r"\n*参考[:：]\s*$", "", out.rstrip())
    return out, removed


def strip_tool_markup(text: str) -> str:
    """関数呼び出しの記法が始まる位置より後を取り除く。"""
    m = TOOL_MARKUP.search(text)
    return (text[: m.start()] if m else text).strip()


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


def format_line(m: LoggedMessage, store: Store, self_name: str) -> str:
    """1つの発言を「[日時] 〈発言者〉 本文」の形にする。"""
    who = f"{self_name}（あなた）" if m.is_self else m.author_name + ("（bot）" if m.is_bot else "")
    reply = ""
    if m.reply_author is not None:
        preview = _clip(" ".join((m.reply_content or "").split()), REPLY_PREVIEW_CHARS)
        reply = f"（〈{m.reply_author}〉の発言「{preview}」への返信）"
    ts = (store.fmt(m.created_at) or "")[5:]  # 年を省く
    return f"[{ts}] 〈{who}〉{reply} {_clip(m.content, LINE_MAX_CHARS)}"


def format_transcript(messages: list[LoggedMessage], store: Store, self_name: str) -> str:
    return "\n".join(format_line(m, store, self_name) for m in messages)


def history_window(
    messages: list[LoggedMessage], start_id: int | None, max_count: int, keep: int
) -> tuple[list[LoggedMessage], int | None]:
    """渡す履歴を選ぶ。(履歴, 次に使う履歴の始まりの発言 ID) を返す。

    キャッシュは前回と同じ書き出しの部分に効くので、1件ずつずらさず、始まり（start_id）を固定したまま
    新しい発言を後ろに足していく。max_count 件を超えたら、新しい keep 件まで一度に減らして始まりを進める。
    始まりの発言が消された・古すぎて見つからない場合も、新しい keep 件から始め直す。
    """
    if not messages:
        return [], start_id
    ids = [m.id for m in messages]
    if start_id in ids:
        window = messages[ids.index(start_id):]
        if len(window) <= max_count:
            return window, start_id
    window = messages[-keep:]
    return window, window[0].id


def build_conversation(messages: list[LoggedMessage], store: Store, self_name: str) -> str:
    """会話の履歴、現在の日時、返事をする発言を、この順で1つのユーザーメッセージにまとめる。

    履歴は返事のたびに後ろへ伸びるだけなので、前回の履歴の部分はそのままキャッシュから再利用される。
    毎回変わる現在の日時と返事をする発言は、最後に置く。
    """
    if not messages:
        return "（会話なし）"
    *before, last = messages
    parts = []
    if before:
        parts.append("# このチャンネルの会話（古い順）\n" + format_transcript(before, store, self_name))
    now = store.now()
    parts.append(
        f"# 返事をする発言（現在の日時: {now:%Y-%m-%d %H:%M}（{store.tz}））\n" + format_line(last, store, self_name)
    )
    return "\n\n".join(parts)


class DeepSeek:
    def __init__(self, api_key: str, store: Store, cfg: LLMConfig, profile: Profile) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=API_BASE)
        self.beta = AsyncOpenAI(api_key=api_key, base_url=API_BETA_BASE)
        self.store = store
        self.cfg = cfg
        self.profile = profile
        self.tools = ToolRunner(api_key, store)

    @property
    def name(self) -> str:
        return self.profile.name

    # ---- プロンプトの組み立て ----
    def system_prompt(self, ctx: ToolContext) -> str:
        """変わらない部分のシステムの指示（1. 人格と口調、2. 道具の使い方）を返す。

        DeepSeek のキャッシュは、前回と同じ書き出しの部分に効き、変わった箇所より後ろには効かない。
        そのため変わらないものから順に送る。DeepSeek は関数の一覧（API の tools）を1つ目のシステムの指示の
        直後に置くため、1つ目のシステムの指示には変わらないものだけを入れる。送る順番は次のとおり。
          1. 人格と口調、共通の決まり、道具の使い方（1つ目のシステムの指示）: ファイルを書き換えない限り変わらない
          2. 関数の一覧（API の tools）: 変わらない
          3. サーバーの返信スタイル、共有メモリ、会話に出ている人の人物ノート（2つ目のシステムの指示、server_prompt()）:
             設定・記憶の追加・会話の終わりで変わる。人物ノートは会話に新しい人が加わったときにも変わる
          4. 会話の履歴（ユーザーのメッセージ）: 後ろに伸びる。上限を超えたら一度に減らす（history_window()）
          5. 現在の日時と返事をする発言（ユーザーのメッセージの最後）: 毎回変わる
        """
        values = {"name": self.name, "model": self.cfg.model}

        # 1. 人格と口調
        parts = [load(PROMPTS_DIR / "RULES.md", **values)]
        soul = self.profile.soul(**values)
        if soul:
            parts.append("# 人格（口調・言い回しは「回答の書き方」よりこちらを優先する）\n\n" + soul)
        examples = self.profile.examples(**values)
        if examples:
            parts.append(
                f"# {self.name}の返事の例\n"
                f"「相手」の発言に対する{self.name}の返事の例。言い回しをそのまま繰り返さず、口調・長さ・返し方の参考にする。"
                "例の中の事実（数値や出来事）は、実際の質問への答えに使わない。\n\n" + examples
            )

        # 2. 道具の使い方
        parts.append(load(PROMPTS_DIR / "TOOLS.md", **values))
        return "\n\n".join(parts)

    def server_prompt(self, ctx: ToolContext) -> str | None:
        """3. このサーバーの決まりと共有メモリ（変わりやすいものほど後ろ）。何もなければ None。"""
        parts = []
        style = self.store.get_style(ctx.guild_id)
        if style:
            parts.append(f"# このサーバーでの追加の指定（人格より優先する）\n{style}")
        facts = self.store.list_memories(ctx.guild_id, "fact", self.cfg.fact_limit)
        if facts:
            parts.append("# 覚えていること（[id] 内容）\n" + "\n".join(f"[{m['id']}] {m['content']}" for m in facts))
        # 人物ノートは、会話に出ている人の分だけ渡す。ID の順に並べ、会話中に並びが変わらないようにする
        ids = sorted({*ctx.participants.values(), ctx.user_id})
        notes = [r for r in self.store.get_people(ctx.guild_id, ids) if r["note"]]
        if notes:
            parts.append(
                "# 人物ノート（この会話に出ている人について知っていること。〈 〉は表示名）\n"
                + "\n\n".join(f"## 〈{r['name']}〉\n{r['note']}" for r in notes)
            )
        summaries = self.store.list_memories(ctx.guild_id, "summary", self.cfg.summary_limit)
        if summaries:
            parts.append(
                "# 過去の会話の要約（古い順）\n"
                + "\n".join(f"- {self.store.fmt(m['created_at'])}: {m['content']}" for m in summaries)
            )
        return "\n\n".join(parts) or None

    def _tool_defs(self, names: set[str] | None = None) -> list[dict[str, Any]]:
        """渡す関数の一覧。names を指定すると、その関数だけにする。"""
        defs = TOOL_DEFS
        if not self.cfg.web_search:
            defs = [t for t in defs if t["function"]["name"] not in {"web_search", "read_url"}]
        if names is not None:
            defs = [t for t in defs if t["function"]["name"] in names]
        return defs

    async def _create(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None, beta: bool = False
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
        res = await (self.beta if beta else self.client).chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            max_tokens=self.cfg.max_tokens,
            extra_body={"thinking": {"type": "enabled" if self.cfg.thinking else "disabled"}},
            **kwargs,
        )
        choice = res.choices[0]
        usage = res.usage
        log.info(
            "DeepSeek model=%s finish=%s input=%s (cache_hit=%s) output=%s tool_calls=%d",
            res.model, choice.finish_reason,
            usage.prompt_tokens if usage else "?", getattr(usage, "prompt_cache_hit_tokens", "?"),
            usage.completion_tokens if usage else "?", len(choice.message.tool_calls or []),
        )
        return choice

    # ---- 回答 ----
    async def answer(self, conversation: list[LoggedMessage], ctx: ToolContext) -> str:
        """会話の最後の発言への返事を生成する。関数の呼び出しを処理し、出力上限で止まった場合は続きを生成する。"""
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt(ctx)}]
        server = self.server_prompt(ctx)
        if server:
            messages.append({"role": "system", "content": server})
        messages.append({"role": "user", "content": build_conversation(conversation, self.store, self.name)})
        return await self._complete(messages, ctx, self._tool_defs())

    async def _complete(self, messages: list[dict[str, Any]], ctx: ToolContext, tools: list[dict[str, Any]]) -> str:
        """関数の呼び出しを繰り返して回答を作る。出力上限で止まった場合は続きを生成する。

        messages の最後はユーザーのメッセージ（会話や、実行する予定の内容）にする。
        """
        base = list(messages)
        tool_log: list[str] = []  # 続きの生成で使う、関数の実行結果の記録
        choice = None
        for _ in range(self.cfg.max_tool_rounds):
            choice = await self._create(messages, tools=tools)
            calls = choice.message.tool_calls or []
            if not calls:
                break
            messages.append(choice.message.model_dump(exclude_none=True))
            for call in calls:
                result = await self.tools.run(call.function.name, call.function.arguments, ctx)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                tool_log.append(f"## {call.function.name}({call.function.arguments})\n{result}")
        else:
            # 関数の呼び出しが上限回数まで続いた場合は、関数なしで答えさせる
            log.warning("関数の呼び出しが %d 回続いたため、関数なしで答えさせます", self.cfg.max_tool_rounds)
            choice = await self._create([*messages, {"role": "system", "content": FINAL_ANSWER_NOTE}], tools=None)

        text = choice.message.content or ""
        if TOOL_MARKUP.search(text):
            # モデルが関数呼び出しを文字として書いてしまった場合は、関数なしでもう一度答えさせる
            log.warning("回答に関数呼び出しの記法が含まれていたため、答え直させます")
            retry = await self._create([*messages, {"role": "system", "content": FINAL_ANSWER_NOTE}], tools=None)
            text = strip_tool_markup(retry.message.content or "")
            choice = retry
            if not text:
                text = "調べきれませんでした。質問を具体的にしてもう一度送ってください。"
        finish = choice.finish_reason
        # 続きの生成（prefix）は関数呼び出しを含む履歴と一緒に使えないため、実行結果を文章にして渡す
        cont_base = base + (
            [{"role": "system", "content": "# この回答のために実行した関数の結果\n" + "\n\n".join(tool_log)}] if tool_log else []
        )
        for _ in range(self.cfg.max_continuations):
            if finish != "length":
                break
            try:
                cont = await self._create(
                    [*cont_base, {"role": "assistant", "content": text, "prefix": True}], tools=None, beta=True
                )
            except APIStatusError as e:
                log.warning("続きの生成に失敗しました (%s): %s", e.status_code, e.message)
                break
            text += cont.message.content or ""
            finish = cont.finish_reason
        if finish == "length":
            text += "\n\n（出力できる文字数の上限に達したため、ここで終わっています）"
        text, removed = filter_sources(text, "\n".join(tool_log) + "\n" + messages[len(base) - 1]["content"])
        if removed:
            log.warning("検索結果にない参考リンクを %d 件取り除きました", removed)
        return text.strip() or "(空の応答)"

    async def should_reply(self, conversation: list[LoggedMessage]) -> tuple[bool, bool]:
        """メンションされていない最後の発言に返事をすべきかを判定する。(reply, end) を返す。"""
        res = await self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": load(PROMPTS_DIR / "JUDGE.md", name=self.name)},
                {"role": "user", "content": build_conversation(conversation, self.store, self.name)},
            ],
            max_tokens=50,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        try:
            data = json.loads(res.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            log.warning("判定の出力を読めませんでした: %r", res.choices[0].message.content)
            return False, False
        return bool(data.get("reply")), bool(data.get("end"))

    async def summarize(self, conversation: list[LoggedMessage]) -> str | None:
        """会話を要約する。残す価値がなければ None を返す。"""
        res = await self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": load(PROMPTS_DIR / "SUMMARY.md", name=self.name)},
                {"role": "user", "content": format_transcript(conversation, self.store, self.name)},
            ],
            max_tokens=400,
            extra_body={"thinking": {"type": "disabled"}},
        )
        text = (res.choices[0].message.content or "").strip()
        return None if not text or text.startswith("なし") else text

    async def _plain(self, system: list[str], user: str, *, max_tokens: int, json_mode: bool = False) -> str:
        """関数も thinking も使わない、短い生成。system は先頭から順にシステムの指示として送る。"""
        kwargs: dict[str, Any] = {"response_format": {"type": "json_object"}} if json_mode else {}
        res = await self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[*({"role": "system", "content": t} for t in system), {"role": "user", "content": user}],
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
            **kwargs,
        )
        usage = res.usage
        log.info(
            "DeepSeek (plain) input=%s (cache_hit=%s) output=%s",
            usage.prompt_tokens if usage else "?", getattr(usage, "prompt_cache_hit_tokens", "?"),
            usage.completion_tokens if usage else "?",
        )
        return (res.choices[0].message.content or "").strip()

    # ---- 気にかけること ----
    async def checkin(
        self, ctx: ToolContext, *, name: str, topic: str, context: str | None, registered_at: int,
        since_lines: list[LoggedMessage],
    ) -> str | None:
        """登録しておいた「気にかけること」について、本人にかける一言を作る。声をかけない方がよければ None。"""
        # 1つ目のシステムの指示は返事のときと同じにして、人格の部分にキャッシュを効かせる
        system = [self.system_prompt(ctx)]
        server = self.server_prompt(ctx)
        if server:
            system.append(server)
        system.append(load(PROMPTS_DIR / "CHECKIN.md", name=self.name))
        lines = format_transcript(since_lines, self.store, self.name) if since_lines else "（なし）"
        now = self.store.now()
        user = (
            f"# 声をかける相手\n〈{name}〉\n\n"
            f"# 気にかけていたこと\n{topic}\n"
            + (f"（きっかけ: {context}）\n" if context else "")
            + f"（登録した日時: {self.store.fmt(registered_at)}）\n\n"
            f"# 登録した後の、このチャンネルでの〈{name}〉の発言\n{lines}\n\n"
            f"# 現在の日時\n{now:%Y-%m-%d %H:%M}（{self.store.tz}）"
        )
        text = strip_tool_markup(await self._plain(system, user, max_tokens=300))
        return None if not text or text.startswith("なし") else text

    # ---- 人物ノート ----
    async def update_notes(
        self, session: list[LoggedMessage], people: dict[int, tuple[str, str | None]]
    ) -> dict[int, str]:
        """会話を読み、参加した人の人物ノートを書き直す。people は {ID: (表示名, 今のノート)}。

        書き直したノートだけを {ID: ノート} で返す（変える必要がない人は含めない）。
        """
        if not people:
            return {}
        current = "\n\n".join(
            f"## id={uid} 〈{name}〉\n{note or '（まだなし）'}" for uid, (name, note) in people.items()
        )
        user = (
            "# 今の人物ノート\n" + current + "\n\n# 今回の会話（古い順）\n"
            + format_transcript(session, self.store, self.name)
        )
        raw = await self._plain(
            [load(PROMPTS_DIR / "NOTES.md", name=self.name)], user, max_tokens=1500, json_mode=True
        )
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            log.warning("人物ノートの出力を読めませんでした: %r", raw[:200])
            return {}
        out: dict[int, str] = {}
        for key, note in (data.get("notes") or {}).items():
            try:
                uid = int(key)
            except (TypeError, ValueError):
                continue
            if uid in people and isinstance(note, str) and note.strip() and note.strip() != (people[uid][1] or ""):
                out[uid] = note.strip()[:400]
        return out

    # ---- 予定のタスク ----
    async def run_task(self, ctx: ToolContext, *, instruction: str, owner: str, registered_at: int, repeat: str) -> str | None:
        """登録しておいたタスクを、その時刻に関数を使って実行し、投稿する文章を返す。投稿しない場合は None。"""
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt(ctx)}]
        server = self.server_prompt(ctx)
        if server:
            messages.append({"role": "system", "content": server})
        messages.append({"role": "system", "content": load(PROMPTS_DIR / "TASK.md", name=self.name)})
        now = self.store.now()
        messages.append({
            "role": "user",
            "content": (
                f"# 実行する予定\n{instruction}\n\n"
                f"- 頼んだ人: 〈{owner}〉\n- 登録した日時: {self.store.fmt(registered_at)}\n- 繰り返し: {repeat}\n\n"
                f"# 現在の日時\n{now:%Y-%m-%d %H:%M}（{self.store.tz}）"
            ),
        })
        text = (await self._complete(messages, ctx, self._tool_defs(TASK_TOOL_NAMES))).strip()
        # メンションは bot が付けるので、モデルが書いた「<@名前>」のような偽のメンションは取り除く
        text = re.sub(r"<@!?[^\d>][^>]*>\s*", "", text).strip()
        return None if text in {"", "なし", "(空の応答)"} else text
