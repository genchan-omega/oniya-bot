"""DeepSeek API を使って返事をする Discord bot。

- bot が読めるチャンネルの発言を SQLite に記録し、返事を作るときはチャンネルの直近の会話を丸ごと渡す
- bot へのメンション（または bot のメッセージへの返信）で会話を始める
- 会話中は、メンションがなくても返事が必要かどうかを判断して返事をする
- 一定時間 bot が返事をしないか、会話が終わったと判断したら会話を終え、内容を要約して記憶する
- Web 検索、過去の発言の検索、リマインダー、定期投稿、予定の管理、長期の記憶、返信スタイルの保存を関数呼び出しで行う
- 会話に出た予定や宣言を「気にかけること」として登録し、その時刻に bot から本人に「どうだった？」と声をかける
- 1人1枚の人物ノートを会話が終わるたびに書き直し、会話に出ている人の分だけ DeepSeek に渡す
- 人格は profiles/<PROFILE>/、共通の決まりは prompts/ のファイルに書く
- 設定値は .env から読み込む（.env.example を参照）
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import discord
from discord import app_commands
from discord.ext import tasks
from dotenv import load_dotenv
from openai import APIStatusError, OpenAIError

from llm import DeepSeek, LLMConfig, history_window
from prompts import load_profile
from store import LoggedMessage, Store
from tools import ToolContext

load_dotenv()

log = logging.getLogger("oniya-bot")

DISCORD_MESSAGE_LIMIT = 2000
LATE_NOTICE_SEC = 300  # これより遅れて送るリマインダーには「遅れて送った」と書く
REPLY_STORE_CHARS = 300  # 返信先の本文として記録する最大文字数


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        sys.exit(f"環境変数 {name} が設定されていません。.env に値を入れてください（書き方は .env.example を参照）。")
    return value


def _bool_env(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# 返事を作るときに渡す会話の履歴。キャッシュが効くよう、履歴の始まりを固定したまま新しい発言を後ろに足していき、
# CONTEXT_MAX 件を超えたら新しい CONTEXT_KEEP 件まで一度に減らす。
CONTEXT_MAX = int(os.getenv("CONTEXT_MAX", "60"))
CONTEXT_KEEP = int(os.getenv("CONTEXT_KEEP", "30"))
CONVERSATION_TIMEOUT = int(os.getenv("CONVERSATION_TIMEOUT", "600"))  # 秒
LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "180"))  # 発言の記録を残す日数
TZ = ZoneInfo(os.getenv("TZ_NAME", "Asia/Tokyo"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PROFILE = os.getenv("PROFILE", "oniya")


def _hours_env(name: str, default: str) -> tuple[int, int]:
    start, end = (int(x) for x in os.getenv(name, default).split("-"))
    return start % 24, end % 24


# 「気にかけること」の声かけ。夜中（QUIET_HOURS の間）は送らず、終わりの時刻まで遅らせる
QUIET_HOURS = _hours_env("QUIET_HOURS", "0-8")
# 頻度ごとの1日の上限（サーバー全体, 1人あたり）。頻度はサーバーごとに /checkin で変えられる
FOLLOWUP_LIMITS = {"off": (0, 0), "low": (2, 1), "normal": (5, 1), "high": (10, 2)}
FOLLOWUP_LEVEL_NAMES = {"off": "オフ", "low": "控えめ", "normal": "普通", "high": "多め"}
FOLLOWUP_EXPIRE_SEC = 2 * 86400  # bot が止まっていたなどで、これより遅れた声かけは送らない


def load_llm_config() -> LLMConfig:
    return LLMConfig(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        max_tokens=int(os.getenv("MAX_TOKENS", "16384")),
        max_continuations=int(os.getenv("MAX_CONTINUATIONS", "3")),
        thinking=_bool_env("THINKING", True),
        web_search=_bool_env("WEB_SEARCH", True),
    )


@dataclass
class ChannelState:
    guild_id: int
    active_until: float = 0.0
    session_start: int | None = None  # 今の会話が始まった時刻（UNIX 時間）。会話中でなければ None
    history_from: int | None = None  # DeepSeek に渡す履歴の最初の発言の ID
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def is_active(self) -> bool:
        return time.monotonic() < self.active_until

    def extend(self, started_at: int) -> None:
        if self.session_start is None:
            self.session_start = started_at
        self.active_until = time.monotonic() + CONVERSATION_TIMEOUT


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


def describe_api_error(e: Exception) -> str:
    if isinstance(e, APIStatusError):
        reasons = {
            401: "API Key が正しくありません。",
            402: "DeepSeek の残高が足りません。",
            429: "リクエストが多すぎます。少し待ってから送ってください。",
        }
        return f"エラーが起きました（{e.status_code}）。{reasons.get(e.status_code, 'しばらくしてからもう一度試してください。')}"
    return "エラーが起きました。DeepSeek API に接続できませんでした。"


def message_text(message: discord.Message) -> str:
    """記録する本文。メンションは @名前 にし、添付ファイルやスタンプは名前を書き添える。"""
    text = message.clean_content.strip()
    extras = [f"[添付: {a.filename}]" for a in message.attachments]
    extras += [f"[スタンプ: {s.name}]" for s in message.stickers]
    return "\n".join([text, *extras]).strip()


intents = discord.Intents.default()
intents.message_content = True  # Developer Portal で Message Content Intent をオンにする必要がある


class OniyaBot(discord.Client):
    def __init__(self, deepseek: DeepSeek, store: Store) -> None:
        super().__init__(intents=intents, allowed_mentions=discord.AllowedMentions(everyone=True, roles=True))
        self.tree = app_commands.CommandTree(self)
        self.deepseek = deepseek
        self.store = store
        self.channels: dict[int, ChannelState] = {}
        self.backfilled: set[int] = set()  # 起動後に過去の発言を読み込んだチャンネル

    def state(self, channel_id: int, guild_id: int) -> ChannelState:
        return self.channels.setdefault(channel_id, ChannelState(guild_id=guild_id))

    async def setup_hook(self) -> None:
        # スラッシュコマンドを Discord に登録する（全サーバーへの反映には最大1時間ほどかかることがある）
        synced = await self.tree.sync()
        # Discord のアプリに反映されないときは、ここに出たコマンド名と見比べる（アプリは Ctrl+R / Cmd+R で再読み込み）
        log.info("スラッシュコマンドを登録しました: %s", ", ".join("/" + c.name for c in synced))
        self.scheduler.start()

    async def close(self) -> None:
        await self.deepseek.tools.close()
        await super().close()

    # ---- 発言の記録 ----
    def to_logged(self, message: discord.Message) -> LoggedMessage:
        reply_author = reply_content = None
        ref = message.reference.resolved if message.reference else None
        if isinstance(ref, discord.Message):
            reply_author = ref.author.display_name
            reply_content = message_text(ref)[:REPLY_STORE_CHARS]
        elif message.reference is not None and message.type == discord.MessageType.reply:
            reply_author, reply_content = "不明", "（削除されたか、読み込めない発言）"
        return LoggedMessage(
            id=message.id,
            guild_id=message.guild.id if message.guild else 0,
            channel_id=message.channel.id,
            author_id=message.author.id,
            author_name=message.author.display_name,
            is_bot=message.author.bot,
            is_self=self.user is not None and message.author.id == self.user.id,
            content=message_text(message),
            reply_author=reply_author,
            reply_content=reply_content,
            created_at=int(message.created_at.timestamp()),
        )

    async def backfill(self, channel: discord.abc.Messageable, channel_id: int) -> None:
        """起動中に流れた発言を取りこぼさないよう、チャンネルごとに最初の1回だけ直近の発言を読み込む。"""
        if channel_id in self.backfilled:
            return
        self.backfilled.add(channel_id)
        try:
            items = [self.to_logged(m) async for m in channel.history(limit=CONTEXT_MAX)]
        except discord.DiscordException:
            log.exception("過去の発言を読み込めませんでした channel=%s", channel_id)
            return
        added = self.store.log_messages(items)
        log.info("過去の発言を読み込みました channel=%s added=%d", channel_id, added)

    def conversation(self, state: ChannelState, channel_id: int, upto_id: int | None = None) -> list[LoggedMessage]:
        """DeepSeek に渡す履歴（最後が返事をする発言）を返す。履歴の始まりは state に覚えておく。"""
        msgs = self.store.recent_messages(channel_id, CONTEXT_MAX + 10)
        if upto_id is not None:  # 返事をする発言より後の発言は含めない
            msgs = [m for m in msgs if m.id <= upto_id]
        window, state.history_from = history_window(msgs, state.history_from, CONTEXT_MAX, CONTEXT_KEEP)
        return window

    # ---- 会話 ----
    def is_addressed(self, message: discord.Message) -> bool:
        """メンションされたか、bot のメッセージへの返信か。"""
        assert self.user is not None
        if self.user in message.mentions:
            return True
        ref = message.reference.resolved if message.reference else None
        return isinstance(ref, discord.Message) and ref.author.id == self.user.id

    async def generate(self, state: ChannelState, conversation: list[LoggedMessage], ctx: ToolContext) -> str:
        try:
            text = await self.deepseek.answer(conversation, ctx)
        except OpenAIError as e:
            log.exception("DeepSeek API の呼び出しに失敗しました")
            return describe_api_error(e)
        state.extend(conversation[-1].created_at if conversation else int(time.time()))
        return text

    async def respond(self, message: discord.Message, state: ChannelState) -> None:
        conversation = self.conversation(state, message.channel.id, message.id)
        ctx = ToolContext(
            guild_id=state.guild_id, channel_id=message.channel.id,
            user_id=message.author.id, user_name=message.author.display_name,
            participants={m.author_name: m.author_id for m in conversation if not m.is_bot},
        )
        async with message.channel.typing():
            text = await self.generate(state, conversation, ctx)
        chunks = split_message(text)
        await message.reply(chunks[0], mention_author=False)  # 送った発言は on_message で記録される
        for chunk in chunks[1:]:
            await message.channel.send(chunk)

    async def end_conversation(self, state: ChannelState, channel_id: int) -> None:
        """会話を終え、内容を要約して記憶に残す。"""
        start, state.session_start = state.session_start, None
        state.active_until = 0.0
        if start is None:
            return
        session = self.store.recent_messages(channel_id, 80, since=start)
        if not any(m.is_self for m in session):
            return
        try:
            summary = await self.deepseek.summarize(session)
        except OpenAIError:
            log.exception("会話の要約に失敗しました")
            return
        if summary:
            self.store.add_memory(state.guild_id, "summary", summary, None)
        log.info("会話を終了しました channel=%s summary=%s", channel_id, summary)
        await self.update_notes(state.guild_id, session)

    async def update_notes(self, guild_id: int, session: list[LoggedMessage]) -> None:
        """会話に参加した人（bot を除く）の人物ノートを書き直す。"""
        names = {m.author_id: m.author_name for m in session if not m.is_bot}
        rows = {r["user_id"]: r["note"] for r in self.store.get_people(guild_id, list(names))}
        people = {uid: (name, rows.get(uid)) for uid, name in names.items()}
        try:
            updated = await self.deepseek.update_notes(session, people)
        except OpenAIError:
            log.exception("人物ノートの更新に失敗しました")
            return
        for uid, note in updated.items():
            self.store.set_note(guild_id, uid, names[uid], note)
        if updated:
            log.info("人物ノートを更新しました: %s", ", ".join(names[u] for u in updated))

    async def send_to(self, channel_id: int, text: str) -> None:
        channel = self.get_channel(channel_id) or await self.fetch_channel(channel_id)
        for chunk in split_message(text):
            await channel.send(chunk)  # type: ignore[union-attr]

    # ---- 定期処理 ----
    @tasks.loop(seconds=30)
    async def scheduler(self) -> None:
        now_ts = int(time.time())
        # リマインダー・定期投稿
        for row in self.store.due_reminders(now_ts):
            prefix = f"<@{row['user_id']}> " if row["mention_user"] else ""
            late = "（bot が止まっていたため遅れて送っています）\n" if now_ts - row["due_at"] > LATE_NOTICE_SEC else ""
            label = "定期投稿" if row["rule"] else "リマインダー"
            try:
                await self.send_to(row["channel_id"], f"{prefix}⏰ {label}: {row['text']}\n{late}".strip())
            except discord.DiscordException:
                log.exception("リマインダー %s を送れませんでした", row["id"])
            self.store.finish_reminder(row)
        # 予定の事前通知
        for ev in self.store.events_to_notify(now_ts):
            mins = max(0, (ev["start_at"] - now_ts) // 60)
            when = "まもなく" if mins == 0 else f"{mins}分後"
            place = f"（場所: {ev['location']}）" if ev["location"] else ""
            try:
                await self.send_to(
                    ev["channel_id"], f"📅 予定「{ev['title']}」が{when}（{self.store.fmt(ev['start_at'])}）に始まります{place}"
                )
            except discord.DiscordException:
                log.exception("予定 %s の通知を送れませんでした", ev["id"])
            self.store.mark_event_notified(ev["id"])
        # 気にかけていたことの声かけ
        for row in self.store.due_followups(now_ts):
            await self.send_followup(row, now_ts)
        # 時間切れになった会話を終える
        for channel_id, state in list(self.channels.items()):
            if state.session_start is not None and not state.is_active() and not state.lock.locked():
                async with state.lock:
                    await self.end_conversation(state, channel_id)

    def _quiet_until(self, now: datetime) -> datetime | None:
        """今が声をかけない時間帯なら、その終わりの日時を返す。"""
        start, end = QUIET_HOURS
        h = now.hour
        quiet = start <= h < end if start <= end else (h >= start or h < end)
        if not quiet:
            return None
        until = now.replace(hour=end, minute=0, second=0, microsecond=0)
        return until if until > now else until + timedelta(days=1)

    async def send_followup(self, row, now_ts: int) -> None:  # row は followups の1行
        fid, gid, uid = row["id"], row["guild_id"], row["user_id"]
        level = self.store.get_followup_level(gid)
        if level == "off" or not self.store.wants_followups(gid, uid):
            self.store.finish_followup(fid, "skipped")
            return
        if now_ts - row["check_at"] > FOLLOWUP_EXPIRE_SEC:
            self.store.finish_followup(fid, "expired")
            return
        now = self.store.now()
        quiet_end = self._quiet_until(now)
        if quiet_end is not None:
            self.store.postpone_followup(fid, int(quiet_end.timestamp()))
            return
        # 1日の上限に達していたら、翌日の朝に回す
        day_start = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        guild_cap, person_cap = FOLLOWUP_LIMITS.get(level, FOLLOWUP_LIMITS["normal"])
        if (self.store.sent_followups_since(gid, day_start) >= guild_cap
                or self.store.sent_followups_since(gid, day_start, uid) >= person_cap):
            tomorrow = now.replace(hour=QUIET_HOURS[1], minute=0, second=0, microsecond=0) + timedelta(days=1)
            self.store.postpone_followup(fid, int(tomorrow.timestamp()))
            return
        state = self.state(row["channel_id"], gid)
        if state.lock.locked():
            return  # そのチャンネルで返事を作っている最中なら、次の回に回す
        ctx = ToolContext(
            guild_id=gid, channel_id=row["channel_id"], user_id=uid, user_name=row["user_name"],
            participants={row["user_name"]: uid},
        )
        try:
            text = await self.deepseek.checkin(
                ctx, name=row["user_name"], topic=row["topic"], context=row["context"],
                registered_at=row["created_at"],
                since_lines=self.store.messages_by(row["channel_id"], uid, row["created_at"]),
            )
        except OpenAIError:
            log.exception("声かけ %s の文章を作れませんでした", fid)
            self.store.postpone_followup(fid, now_ts + 1800)
            return
        if text is None:
            self.store.finish_followup(fid, "skipped")
            log.info("声かけ %s（%s）は不要と判断しました", fid, row["topic"])
            return
        try:
            await self.send_to(row["channel_id"], f"<@{uid}> {text}")
        except discord.DiscordException:
            log.exception("声かけ %s を送れませんでした", fid)
            self.store.finish_followup(fid, "skipped")
            return
        self.store.finish_followup(fid, "sent")
        state.extend(now_ts)  # 返信しなくても答えられるよう、そのチャンネルを会話中にする
        log.info("声かけ %s を送りました: %s", fid, row["topic"])

    @tasks.loop(hours=24)
    async def pruner(self) -> None:
        removed = self.store.prune_messages(int(time.time()) - LOG_RETENTION_DAYS * 86400)
        if removed:
            log.info("古い発言の記録を %d 件削除しました", removed)

    @scheduler.before_loop
    async def before_scheduler(self) -> None:
        await self.wait_until_ready()
        if not self.pruner.is_running():
            self.pruner.start()

    @scheduler.error
    async def scheduler_error(self, error: BaseException) -> None:
        log.exception("定期処理でエラーが起きました", exc_info=error)
        self.scheduler.restart()


def create_bot(deepseek: DeepSeek, store: Store) -> OniyaBot:
    bot = OniyaBot(deepseek, store)

    @bot.event
    async def on_ready() -> None:
        log.info(
            "ログインしました: %s (id=%s) model=%s profile=%s",
            bot.user, bot.user.id if bot.user else "?", deepseek.cfg.model, deepseek.name,
        )

    @bot.event
    async def on_message(message: discord.Message) -> None:
        if bot.user is None:
            return
        # 自分や他の bot の発言も含めて、全て記録する
        if message.type in (discord.MessageType.default, discord.MessageType.reply, discord.MessageType.chat_input_command):
            store.log_message(bot.to_logged(message))
            if not message.author.bot and message.guild:
                store.touch_person(message.guild.id, message.author.id, message.author.display_name)
        if message.author.bot or not message_text(message):
            return

        addressed = bot.is_addressed(message)
        state = bot.state(message.channel.id, message.guild.id if message.guild else 0)
        if not addressed and not state.is_active():
            return  # 会話中でなければ、メンションされていない発言には返事をしない

        async with state.lock:
            await bot.backfill(message.channel, message.channel.id)
            if addressed:
                await bot.respond(message, state)
                return
            try:
                reply, end = await bot.deepseek.should_reply(bot.conversation(state, message.channel.id, message.id))
            except OpenAIError:
                log.exception("返事の要否の判定に失敗しました")
                return
            log.info("判定 channel=%s reply=%s end=%s", message.channel.id, reply, end)
            if reply:
                await bot.respond(message, state)
            elif end:
                await bot.end_conversation(state, message.channel.id)

    @bot.event
    async def on_raw_message_edit(payload: discord.RawMessageUpdateEvent) -> None:
        # payload.message は編集後の発言
        if "content" in payload.data:
            store.edit_message(payload.message_id, message_text(payload.message))

    @bot.event
    async def on_raw_message_delete(payload: discord.RawMessageDeleteEvent) -> None:
        store.delete_message(payload.message_id)

    @bot.tree.command(name="reset", description="このチャンネルでの会話を終える")
    async def reset_command(interaction: discord.Interaction) -> None:
        await interaction.response.send_message("このチャンネルでの会話を終えました。")
        state = bot.channels.get(interaction.channel_id or 0)
        if state:
            async with state.lock:
                await bot.end_conversation(state, interaction.channel_id or 0)

    @bot.tree.command(name="memory", description="このサーバーで bot が覚えていることを表示する")
    async def memory_command(interaction: discord.Interaction) -> None:
        gid = interaction.guild_id or 0
        facts = store.list_memories(gid, "fact", 100)
        summaries = store.list_memories(gid, "summary", 10)
        style = store.get_style(gid)
        lines = ["**覚えていること**"] + ([f"[{m['id']}] {m['content']}" for m in facts] or ["（なし）"])
        lines += ["", "**最近の会話の要約**"] + (
            [f"[{m['id']}] {store.fmt(m['created_at'])}: {m['content']}" for m in summaries] or ["（なし）"]
        )
        lines += ["", "**このサーバーでの追加の指定**", style or "（なし）"]
        mine = store.get_people(gid, [interaction.user.id])
        lines += ["", "**あなたの人物ノート**", (mine[0]["note"] if mine and mine[0]["note"] else None) or "（なし）"]
        lines += ["", "消したい項目は「[id] を忘れて」「俺のノート消して」と bot に頼んでください。"]
        chunks = split_message("\n".join(lines))
        await interaction.response.send_message(chunks[0], ephemeral=True)
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk, ephemeral=True)

    @bot.tree.command(name="checkin", description="bot から声をかける「気にかけること」の一覧と頻度")
    @app_commands.describe(level="声をかける頻度（変更にはサーバー管理の権限が要る）")
    @app_commands.choices(level=[app_commands.Choice(name=v, value=k) for k, v in FOLLOWUP_LEVEL_NAMES.items()])
    async def checkin_command(interaction: discord.Interaction, level: app_commands.Choice[str] | None = None) -> None:
        gid = interaction.guild_id or 0
        perms = getattr(interaction.user, "guild_permissions", None)
        notice = ""
        if level is not None:
            if perms is None or not perms.manage_guild:
                notice = "頻度を変えるにはサーバー管理の権限が必要です。\n\n"
            else:
                store.set_followup_level(gid, level.value)
                notice = f"声かけの頻度を「{level.name}」にしました。\n\n"
        cur = store.get_followup_level(gid)
        guild_cap, person_cap = FOLLOWUP_LIMITS[cur]
        items = store.list_followups(gid)
        lines = [
            notice + f"**声かけの頻度**: {FOLLOWUP_LEVEL_NAMES[cur]}（1日にサーバー全体で{guild_cap}回、1人{person_cap}回まで。"
            f"{QUIET_HOURS[0]}〜{QUIET_HOURS[1]}時は送らない）",
            "", "**気にかけていること**",
        ] + ([f"[{f['id']}] {f['when']} {f['who']}: {f['topic']}" for f in items] or ["（なし）"])
        lines += ["", "声をかけてほしくない場合は「そういうのいらない」と bot に言ってください。"]
        chunks = split_message("\n".join(lines))
        await interaction.response.send_message(chunks[0], ephemeral=True)
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk, ephemeral=True)

    return bot


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    token = _require_env("DISCORD_TOKEN")
    api_key = _require_env("DEEPSEEK_API_KEY")
    store = Store(DATA_DIR / "oniya.db", TZ)
    log.info("データの保存先: %s", (DATA_DIR / "oniya.db").resolve())
    # 再デプロイ後もこの件数が0に戻らなければ、ボリュームにデータが残っている
    log.info("保存されている件数: %s", ", ".join(f"{k} {v}" for k, v in store.counts().items()))
    deepseek = DeepSeek(api_key, store, load_llm_config(), load_profile(PROFILE))
    create_bot(deepseek, store).run(token, log_handler=None)


if __name__ == "__main__":
    main()
