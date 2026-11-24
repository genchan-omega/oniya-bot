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
from tools import TOOL_DEFS, ToolContext, ToolRunner

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
    thinking: bool = False
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


def build_conversation(messages: list[LoggedMessage], store: Store, self_name: str) -> str:
    """直近の会話と、返事をする発言を1つのユーザーメッセージにまとめる。"""
    if not messages:
        return "（会話なし）"
    *before, last = messages
    parts = []
    if before:
        parts.append("# このチャンネルの直近の会話（古い順）\n" + format_transcript(before, store, self_name))
    parts.append("# 返事をする発言\n" + format_line(last, store, self_name))
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
        now = self.store.now()
        values = {"name": self.name, "now": f"{now:%Y-%m-%d %H:%M}", "tz": str(self.store.tz), "model": self.cfg.model}
        parts = [load(PROMPTS_DIR / "RULES.md", **values)]

        soul = self.profile.soul(**values)
        if soul:
            parts.append("# 人格（口調・言い回しは「回答の書き方」よりこちらを優先する）\n\n" + soul)
        style = self.store.get_style(ctx.guild_id)
        if style:
            parts.append(f"# このサーバーでの追加の指定（人格より優先する）\n{style}")

        facts = self.store.list_memories(ctx.guild_id, "fact", self.cfg.fact_limit)
        if facts:
            parts.append("# 覚えていること（[id] 内容）\n" + "\n".join(f"[{m['id']}] {m['content']}" for m in facts))
        summaries = self.store.list_memories(ctx.guild_id, "summary", self.cfg.summary_limit)
        if summaries:
            parts.append(
                "# 過去の会話の要約（古い順）\n"
                + "\n".join(f"- {self.store.fmt(m['created_at'])}: {m['content']}" for m in summaries)
            )
        return "\n\n".join(parts)

    def _tool_defs(self) -> list[dict[str, Any]]:
        if self.cfg.web_search:
            return TOOL_DEFS
        return [t for t in TOOL_DEFS if t["function"]["name"] not in {"web_search", "read_url"}]

    async def _create(self, messages: list[dict[str, Any]], *, tools: bool, beta: bool = False) -> Any:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = self._tool_defs()
        res = await (self.beta if beta else self.client).chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            max_tokens=self.cfg.max_tokens,
            extra_body={"thinking": {"type": "enabled" if self.cfg.thinking else "disabled"}},
            **kwargs,
        )
        choice = res.choices[0]
        log.info(
            "DeepSeek model=%s finish=%s tokens=%s tool_calls=%d",
            res.model, choice.finish_reason, res.usage.total_tokens if res.usage else "?",
            len(choice.message.tool_calls or []),
        )
        return choice

    # ---- 回答 ----
    async def answer(self, conversation: list[LoggedMessage], ctx: ToolContext) -> str:
        """会話の最後の発言への返事を生成する。関数の呼び出しを処理し、出力上限で止まった場合は続きを生成する。"""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt(ctx)},
            {"role": "user", "content": build_conversation(conversation, self.store, self.name)},
        ]
        base = list(messages)
        tool_log: list[str] = []  # 続きの生成で使う、関数の実行結果の記録
        choice = None
        for _ in range(self.cfg.max_tool_rounds):
            choice = await self._create(messages, tools=True)
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
            choice = await self._create([*messages, {"role": "system", "content": FINAL_ANSWER_NOTE}], tools=False)

        text = choice.message.content or ""
        if TOOL_MARKUP.search(text):
            # モデルが関数呼び出しを文字として書いてしまった場合は、関数なしでもう一度答えさせる
            log.warning("回答に関数呼び出しの記法が含まれていたため、答え直させます")
            retry = await self._create([*messages, {"role": "system", "content": FINAL_ANSWER_NOTE}], tools=False)
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
                    [*cont_base, {"role": "assistant", "content": text, "prefix": True}], tools=False, beta=True
                )
            except APIStatusError as e:
                log.warning("続きの生成に失敗しました (%s): %s", e.status_code, e.message)
                break
            text += cont.message.content or ""
            finish = cont.finish_reason
        if finish == "length":
            text += "\n\n（出力できる文字数の上限に達したため、ここで終わっています）"
        return text.strip() or "(空の応答)"

    async def should_reply(self, conversation: list[LoggedMessage]) -> tuple[bool, bool]:
        """メンションされていない最後の発言に返事をすべきかを判定する。(reply, end) を返す。"""
        res = await self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": load(PROMPTS_DIR / "JUDGE.md", name=self.name)},
                {"role": "user", "content": build_conversation(conversation[-15:], self.store, self.name)},
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
