"""DeepSeek API を使って返事をする Discord bot。

- bot へのメンション、または /ask コマンドで質問を受け付ける
- チャンネルごとに直近の会話を記憶し、続けて質問できる（/reset で消去）
- 設定値は .env から読み込む（.env.example を参照）
"""

from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict, deque

import discord
from discord import app_commands
from dotenv import load_dotenv
from openai import APIStatusError, AsyncOpenAI, OpenAIError

load_dotenv()

log = logging.getLogger("oniya-bot")

DISCORD_MESSAGE_LIMIT = 2000


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        sys.exit(f"環境変数 {name} が設定されていません。.env に値を入れてください（書き方は .env.example を参照）。")
    return value


DISCORD_TOKEN = _require_env("DISCORD_TOKEN")
DEEPSEEK_API_KEY = _require_env("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "あなたはDiscordサーバーで質問に答える親切なアシスタントです。日本語で簡潔に答えてください。")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "10"))  # 記憶する往復数

ai = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# チャンネルIDごとの会話履歴。1往復で user と assistant の2件を入れる。
histories: dict[int, deque[dict[str, str]]] = defaultdict(lambda: deque(maxlen=HISTORY_TURNS * 2))


def split_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Discord の文字数上限に収まるよう、なるべく改行位置で分割する。"""
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks or ["(空の応答)"]


async def ask_deepseek(channel_id: int, prompt: str) -> str:
    history = histories[channel_id]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": prompt}]
    try:
        res = await ai.chat.completions.create(model=DEEPSEEK_MODEL, messages=messages, max_tokens=MAX_TOKENS)
    except APIStatusError as e:
        log.warning("DeepSeek API error %s: %s", e.status_code, e.message)
        reasons = {
            401: "API Key が正しくありません。",
            402: "DeepSeek の残高が足りません。",
            429: "リクエストが多すぎます。少し待ってから送ってください。",
        }
        return f"エラーが起きました（{e.status_code}）。{reasons.get(e.status_code, 'しばらくしてからもう一度試してください。')}"
    except OpenAIError:
        log.exception("DeepSeek API への接続に失敗しました")
        return "エラーが起きました。DeepSeek API に接続できませんでした。"

    answer = (res.choices[0].message.content or "").strip() or "(空の応答)"
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": answer})
    return answer


intents = discord.Intents.default()
intents.message_content = True  # Developer Portal で Message Content Intent をオンにする必要がある


class OniyaBot(discord.Client):
    def __init__(self) -> None:
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        # スラッシュコマンドを Discord に登録する（全サーバーへの反映には最大1時間ほどかかることがある）
        await self.tree.sync()


bot = OniyaBot()


@bot.event
async def on_ready() -> None:
    log.info("ログインしました: %s (id=%s)", bot.user, bot.user.id if bot.user else "?")


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot or bot.user is None or bot.user not in message.mentions:
        return
    prompt = message.content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
    if not prompt:
        await message.reply("質問をメンションのあとに書いてください。")
        return
    async with message.channel.typing():
        answer = await ask_deepseek(message.channel.id, prompt)
    chunks = split_message(answer)
    await message.reply(chunks[0])
    for chunk in chunks[1:]:
        await message.channel.send(chunk)


@bot.tree.command(name="ask", description="DeepSeek に質問する")
@app_commands.describe(question="質問の内容")
async def ask_command(interaction: discord.Interaction, question: str) -> None:
    await interaction.response.defer(thinking=True)  # 3秒以内に応答しないと失敗扱いになるため先に保留する
    answer = await ask_deepseek(interaction.channel_id or 0, question)
    for chunk in split_message(answer):
        await interaction.followup.send(chunk)


@bot.tree.command(name="reset", description="このチャンネルの会話履歴を消す")
async def reset_command(interaction: discord.Interaction) -> None:
    histories.pop(interaction.channel_id or 0, None)
    await interaction.response.send_message("このチャンネルの会話履歴を消しました。")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    bot.run(DISCORD_TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
