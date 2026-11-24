"""DeepSeek API を使って返事をする Discord bot。

- bot へのメンション（または bot のメッセージへの返信）で会話を始める
- 会話中は、メンションがなくても返事が必要かどうかを判断して返事をする
- 一定時間 bot が返事をしないか、会話が終わったと判断したら会話を終える
- 必要なときは DeepSeek のサーバー側 Web 検索・URL 読み取りを使う
- 長い回答は、出力上限で途中で止まった場合は続きを生成し、2000文字ごとに分けて全て送る
- 設定値は .env から読み込む（.env.example を参照）
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import discord
from discord import app_commands
from dotenv import load_dotenv
from openai import APIStatusError, AsyncOpenAI, OpenAIError

load_dotenv()

log = logging.getLogger("oniya-bot")

DISCORD_MESSAGE_LIMIT = 2000
JST = timezone(timedelta(hours=9))


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        sys.exit(f"環境変数 {name} が設定されていません。.env に値を入れてください（書き方は .env.example を参照）。")
    return value


def _bool_env(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
PERSONA = os.getenv("SYSTEM_PROMPT", "").strip()  # 口調・キャラクターの追加指示（任意）
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "16384"))
MAX_CONTINUATIONS = int(os.getenv("MAX_CONTINUATIONS", "3"))
HISTORY_MESSAGES = int(os.getenv("HISTORY_MESSAGES", "30"))
CONVERSATION_TIMEOUT = int(os.getenv("CONVERSATION_TIMEOUT", "600"))  # 秒
WEB_SEARCH = _bool_env("WEB_SEARCH", True)
THINKING = _bool_env("THINKING", False)

API_BASE = "https://api.deepseek.com"
API_BETA_BASE = "https://api.deepseek.com/beta"  # 途中で止まった回答の続きを生成する機能（Chat Prefix Completion）用
WEB_TOOLS = [{"type": "deepseek_web_search"}, {"type": "deepseek_read_url"}]
NO_REPLY = "[NO_REPLY]"


def build_system_prompt(now: datetime | None = None) -> str:
    now = now or datetime.now(JST)
    rules = f"""あなたはDiscordサーバーの会話に参加しているbotです。現在の日時は {now:%Y-%m-%d %H:%M}（日本時間）です。
会話の履歴では、人間の発言は「名前: 本文」の形で渡されます。

# 回答の書き方
- 1文目から質問への答えを書く。前置き、挨拶、質問の言い換え、「〜と仮定すると」「〜を前提に」などの前提の説明、締めの「他に何かあれば」は書かない。
- 質問が複数の意味に取れる場合は、最もありそうな意味を選んで、そのまま答える。前提や解釈の説明はしない。
- 必要な情報だけを、短く具体的に書く。Discordは表（| 区切り）を表示できないので、表は使わず箇条書きにする。

# 自分自身について
- 質問されない限り、自分の名前・正体・仕組み・能力・制限には一切触れない。
- 自分が何のモデルか聞かれた場合だけ「DeepSeek社の {DEEPSEEK_MODEL} で動いているbotです」と答える。Claude、ChatGPT、Geminiなど他のAIだと名乗らない。

# 調べもの
- 最新の情報、ニュース、価格、日付、人物・企業・製品などの固有名詞、数値、事実関係が関わる質問では、記憶で答えずにWeb検索で確かめてから答える。
- メッセージにURLが含まれる場合は、その内容を読んでから答える。
- 調べても分からないことは、分からないと一言で書く。"""
    if PERSONA:
        rules += f"\n\n# 口調・キャラクター\n{PERSONA}"
    return rules


JUDGE_PROMPT = """あなたはDiscordのbot「{bot_name}」が、次のメッセージに返事をすべきかを判定します。
直近の会話と新しいメッセージを読み、次の形式のjsonだけを出力してください。
{{"reply": true または false, "end": true または false}}

reply を true にする場合:
- bot に話しかけている、bot への質問や依頼である
- bot の直前の回答への追加の質問、反論、続きの依頼である
- その場の誰かが答えを求めている質問で、bot が答えると役に立つ

reply を false にする場合:
- 人間同士の会話で、bot が加わる必要がない
- 相づち、感想、スタンプ代わりの短い発言（「w」「了解」「なるほど」など）

end を true にする場合:
- bot との会話が終わった（お礼や締めの言葉で終わった、人間同士で bot と関係ない話題に移った）
- 相づちや短い反応だけでは end を true にしない"""


@dataclass
class ChannelState:
    history: deque[dict[str, str]] = field(default_factory=lambda: deque(maxlen=HISTORY_MESSAGES))
    active_until: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def is_active(self) -> bool:
        return time.monotonic() < self.active_until

    def extend(self) -> None:
        self.active_until = time.monotonic() + CONVERSATION_TIMEOUT

    def end(self) -> None:
        self.active_until = 0.0


@dataclass
class Answer:
    text: str
    finish_reason: str | None
    used_web_tools: bool
    citations: list[tuple[str, str]]  # (title, url)


def split_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Discord の文字数上限に収まるよう、なるべく改行位置で分割する。"""
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text.strip():
        chunks.append(text)
    return chunks or ["(空の応答)"]


def _get(obj: Any, name: str) -> Any:
    """SDK のオブジェクトでも dict でも属性を取り出す（DeepSeek 独自のフィールドは model_extra に入る）。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    value = getattr(obj, name, None)
    if value is None:
        extra = getattr(obj, "model_extra", None) or {}
        value = extra.get(name)
    return value


def extract_citations(message: Any) -> list[tuple[str, str]]:
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for ann in _get(message, "annotations") or []:
        cite = _get(ann, "url_citation")
        url = _get(cite, "url")
        if not url or url in seen:
            continue
        seen.add(url)
        result.append(((_get(cite, "title") or url).strip(), url))
    return result


def format_citations(citations: list[tuple[str, str]], limit: int = 5) -> str:
    if not citations:
        return ""
    # <URL> の形にすると Discord がリンクのプレビューを表示しない
    lines = [f"- {title} <{url}>" for title, url in citations[:limit]]
    return "\n\n参考:\n" + "\n".join(lines)


class DeepSeek:
    def __init__(self, api_key: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=API_BASE)
        self.beta = AsyncOpenAI(api_key=api_key, base_url=API_BETA_BASE)

    def _extra(self, tools: bool) -> dict[str, Any]:
        extra: dict[str, Any] = {"thinking": {"type": "enabled" if THINKING else "disabled"}}
        if tools:
            extra["tools"] = WEB_TOOLS
        return extra

    async def _call(self, messages: list[dict[str, Any]], *, tools: bool, beta: bool = False) -> Answer:
        client = self.beta if beta else self.client
        res = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            extra_body=self._extra(tools),
        )
        choice = res.choices[0]
        message = choice.message
        server_calls = _get(message, "server_tool_calls") or []
        usage = _get(res.usage, "server_tool_use")
        log.info(
            "DeepSeek model=%s finish=%s tokens=%s server_tools=%s",
            res.model, choice.finish_reason, res.usage.total_tokens if res.usage else "?", usage,
        )
        return Answer(
            text=message.content or "",
            finish_reason=choice.finish_reason,
            used_web_tools=bool(server_calls),
            citations=extract_citations(message),
        )

    async def answer(self, history: list[dict[str, str]]) -> str:
        """回答を生成する。出力上限で止まった場合は続きを生成してつなげる。"""
        messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt()}, *history]
        first = await self._call(messages, tools=WEB_SEARCH)
        text, citations, finish = first.text, first.citations, first.finish_reason

        for _ in range(MAX_CONTINUATIONS):
            if finish != "length":
                break
            if first.used_web_tools:
                # Web ツールを使った回答は API の仕様で続きを生成できない
                break
            try:
                cont = await self._call(
                    [*messages, {"role": "assistant", "content": text, "prefix": True}], tools=False, beta=True
                )
            except APIStatusError as e:
                log.warning("続きの生成に失敗しました (%s): %s", e.status_code, e.message)
                break
            text += cont.text
            finish = cont.finish_reason

        if finish == "length":
            text += "\n\n（出力できる文字数の上限に達したため、ここで終わっています）"
        return text.strip() + format_citations(citations)

    async def should_reply(self, bot_name: str, history: list[dict[str, str]]) -> tuple[bool, bool]:
        """メンションされていないメッセージに返事をすべきかを判定する。(reply, end) を返す。"""
        recent = history[-12:]
        transcript = "\n".join(
            (f"{bot_name}（bot）: {m['content']}" if m["role"] == "assistant" else m["content"]) for m in recent[:-1]
        )
        prompt = f"# 直近の会話\n{transcript or '（なし）'}\n\n# 新しいメッセージ\n{recent[-1]['content']}"
        res = await self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT.format(bot_name=bot_name)},
                {"role": "user", "content": prompt},
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


def describe_api_error(e: Exception) -> str:
    if isinstance(e, APIStatusError):
        reasons = {
            401: "API Key が正しくありません。",
            402: "DeepSeek の残高が足りません。",
            429: "リクエストが多すぎます。少し待ってから送ってください。",
        }
        return f"エラーが起きました（{e.status_code}）。{reasons.get(e.status_code, 'しばらくしてからもう一度試してください。')}"
    return "エラーが起きました。DeepSeek API に接続できませんでした。"


intents = discord.Intents.default()
intents.message_content = True  # Developer Portal で Message Content Intent をオンにする必要がある


class OniyaBot(discord.Client):
    def __init__(self, deepseek: DeepSeek) -> None:
        super().__init__(intents=intents, allowed_mentions=discord.AllowedMentions(everyone=False, roles=False))
        self.tree = app_commands.CommandTree(self)
        self.deepseek = deepseek
        self.channels: dict[int, ChannelState] = {}

    def state(self, channel_id: int) -> ChannelState:
        return self.channels.setdefault(channel_id, ChannelState())

    async def setup_hook(self) -> None:
        # スラッシュコマンドを Discord に登録する（全サーバーへの反映には最大1時間ほどかかることがある）
        await self.tree.sync()

    def is_addressed(self, message: discord.Message) -> bool:
        """メンションされたか、bot のメッセージへの返信か。"""
        assert self.user is not None
        if self.user in message.mentions:
            return True
        ref = message.reference.resolved if message.reference else None
        return isinstance(ref, discord.Message) and ref.author.id == self.user.id

    async def respond(self, message: discord.Message, state: ChannelState) -> None:
        async with message.channel.typing():
            try:
                text = await self.deepseek.answer(list(state.history))
            except OpenAIError as e:
                log.exception("DeepSeek API の呼び出しに失敗しました")
                await message.reply(describe_api_error(e), mention_author=False)
                return
        state.history.append({"role": "assistant", "content": text})
        state.extend()
        chunks = split_message(text)
        await message.reply(chunks[0], mention_author=False)
        for chunk in chunks[1:]:
            await message.channel.send(chunk)


def create_bot(deepseek: DeepSeek) -> OniyaBot:
    bot = OniyaBot(deepseek)

    @bot.event
    async def on_ready() -> None:
        log.info("ログインしました: %s (id=%s) model=%s", bot.user, bot.user.id if bot.user else "?", DEEPSEEK_MODEL)

    @bot.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot or bot.user is None:
            return
        content = message.clean_content.strip()  # メンションを @名前 の形に変えた本文
        if not content:
            return
        addressed = bot.is_addressed(message)
        state = bot.state(message.channel.id)
        if not addressed and not state.is_active():
            return  # 会話中でなければ、メンションされていないメッセージは読まない

        async with state.lock:
            state.history.append({"role": "user", "content": f"{message.author.display_name}: {content}"})
            if addressed:
                state.extend()
                await bot.respond(message, state)
                return
            try:
                reply, end = await bot.deepseek.should_reply(bot.user.display_name, list(state.history))
            except OpenAIError:
                log.exception("返事の要否の判定に失敗しました")
                return
            log.info("判定 channel=%s reply=%s end=%s", message.channel.id, reply, end)
            if reply:
                await bot.respond(message, state)
            elif end:
                state.end()
                log.info("会話を終了しました channel=%s", message.channel.id)

    @bot.tree.command(name="ask", description="DeepSeek に質問する")
    @app_commands.describe(question="質問の内容")
    async def ask_command(interaction: discord.Interaction, question: str) -> None:
        await interaction.response.defer(thinking=True)  # 3秒以内に応答しないと失敗扱いになるため先に保留する
        state = bot.state(interaction.channel_id or 0)
        async with state.lock:
            state.history.append({"role": "user", "content": f"{interaction.user.display_name}: {question}"})
            try:
                text = await bot.deepseek.answer(list(state.history))
            except OpenAIError as e:
                log.exception("DeepSeek API の呼び出しに失敗しました")
                await interaction.followup.send(describe_api_error(e))
                return
            state.history.append({"role": "assistant", "content": text})
            state.extend()
        for chunk in split_message(text):
            await interaction.followup.send(chunk)

    @bot.tree.command(name="reset", description="このチャンネルの会話履歴を消して、会話を終える")
    async def reset_command(interaction: discord.Interaction) -> None:
        bot.channels.pop(interaction.channel_id or 0, None)
        await interaction.response.send_message("このチャンネルの会話履歴を消しました。")

    return bot


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    token = _require_env("DISCORD_TOKEN")
    deepseek = DeepSeek(_require_env("DEEPSEEK_API_KEY"))
    create_bot(deepseek).run(token, log_handler=None)


if __name__ == "__main__":
    main()
