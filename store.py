"""SQLite に保存するデータ（リマインダー・定期投稿・予定・記憶・返信スタイル）を扱う。

時刻はすべて UNIX 時間（秒, UTC）で保存し、表示や入力のときだけ日本時間などに変換する。
"""

from __future__ import annotations

import calendar
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,          -- 登録した人
    mention_user INTEGER NOT NULL,     -- 1: 登録した人にメンションする / 0: しない
    text TEXT NOT NULL,
    due_at INTEGER NOT NULL,           -- 次に送る時刻
    rule TEXT,                         -- 繰り返しの規則（JSON）。NULL なら1回だけ
    active INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders (active, due_at);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    start_at INTEGER NOT NULL,
    end_at INTEGER,
    location TEXT,
    notes TEXT,
    notify_before_min INTEGER,         -- 開始の何分前に知らせるか。NULL なら知らせない
    notified INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_start ON events (guild_id, start_at);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    kind TEXT NOT NULL,                -- 'fact': 覚えておく事柄 / 'summary': 過去の会話の要約
    content TEXT NOT NULL,
    user_id INTEGER,                   -- 記録のきっかけになった人
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_guild ON memories (guild_id, kind, created_at);

CREATE TABLE IF NOT EXISTS guild_settings (
    guild_id INTEGER PRIMARY KEY,
    reply_style TEXT
    -- followup_level は後から追加した列（Store.__init__ の _migrate を参照）
);

-- 人物ノート。サーバーのメンバー1人につき1枚。会話が終わるたびに書き直す（追記ではない）
CREATE TABLE IF NOT EXISTS people (
    guild_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,          -- ノートの対象の人
    name TEXT NOT NULL,                -- 最後に見た表示名
    note TEXT,
    no_followups INTEGER NOT NULL DEFAULT 0,  -- 1: この人には「気にかけること」の声かけをしない
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (guild_id, user_id)
);

-- 気にかけること。決まった時刻に、bot から本人に「どうだった？」と声をかける
CREATE TABLE IF NOT EXISTS followups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,       -- 声をかけるチャンネル（その話が出たチャンネル）
    user_id INTEGER NOT NULL,          -- 声をかける相手
    user_name TEXT NOT NULL,
    topic TEXT NOT NULL,               -- 何について聞くか（例: 金曜の面接）
    context TEXT,                      -- きっかけになった発言など
    check_at INTEGER NOT NULL,         -- 声をかける時刻
    source TEXT,                       -- 'chat' / 'reminder:<id>' / 'event:<id>'
    status TEXT NOT NULL DEFAULT 'pending',  -- pending / sent / skipped / cancelled / expired
    created_by INTEGER,
    created_at INTEGER NOT NULL,
    sent_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_followups_due ON followups (status, check_at);

-- bot が読めるチャンネルの発言の記録。返事を作るときに直近の会話として渡し、古い話は検索に使う
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,            -- Discord のメッセージ ID
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    author_id INTEGER NOT NULL,
    author_name TEXT NOT NULL,
    is_bot INTEGER NOT NULL,           -- 1: bot の発言（このbot自身を含む）
    is_self INTEGER NOT NULL,          -- 1: このbot自身の発言
    content TEXT NOT NULL,
    reply_author TEXT,                 -- 返信先の発言者
    reply_content TEXT,                -- 返信先の本文（先頭だけ）
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages (channel_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_guild ON messages (guild_id, created_at);
"""


@dataclass
class LoggedMessage:
    id: int
    guild_id: int
    channel_id: int
    author_id: int
    author_name: str
    is_bot: bool
    is_self: bool
    content: str
    reply_author: str | None
    reply_content: str | None
    created_at: int

WEEKDAYS_JA = "月火水木金土日"


@dataclass
class Rule:
    """繰り返しの規則。freq は daily / weekly / monthly。"""

    freq: str
    hour: int
    minute: int
    weekdays: list[int] | None = None  # 0=月曜 ... 6=日曜（weekly のとき）
    day_of_month: int | None = None  # 1〜31（monthly のとき。月末を超える日はその月の末日）

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Rule:
        return cls(**json.loads(text))

    def validate(self) -> None:
        if self.freq not in {"daily", "weekly", "monthly"}:
            raise ValueError("freq は daily / weekly / monthly のどれかにしてください")
        if not (0 <= self.hour <= 23 and 0 <= self.minute <= 59):
            raise ValueError("時刻が正しくありません")
        if self.freq == "weekly":
            if not self.weekdays or any(not 0 <= d <= 6 for d in self.weekdays):
                raise ValueError("weekly では weekdays（0=月曜〜6=日曜）を1つ以上指定してください")
        if self.freq == "monthly" and not (self.day_of_month and 1 <= self.day_of_month <= 31):
            raise ValueError("monthly では day_of_month（1〜31）を指定してください")

    def describe(self) -> str:
        t = f"{self.hour:02d}:{self.minute:02d}"
        if self.freq == "daily":
            return f"毎日 {t}"
        if self.freq == "weekly":
            return "毎週" + "・".join(WEEKDAYS_JA[d] for d in sorted(self.weekdays or [])) + f"曜 {t}"
        return f"毎月{self.day_of_month}日 {t}"

    def next_after(self, after: datetime) -> datetime:
        """after（タイムゾーン付き）より後で、規則に合う最初の日時を返す。"""
        tz = after.tzinfo

        def at(d: date) -> datetime:
            return datetime(d.year, d.month, d.day, self.hour, self.minute, tzinfo=tz)

        start = after.date()
        if self.freq == "daily":
            cand = at(start)
            return cand if cand > after else at(start + timedelta(days=1))
        if self.freq == "weekly":
            days = set(self.weekdays or [])
            for i in range(8):
                d = start + timedelta(days=i)
                if d.weekday() in days and at(d) > after:
                    return at(d)
            raise ValueError("次の日時を計算できませんでした")
        y, m = start.year, start.month
        for _ in range(14):
            last = calendar.monthrange(y, m)[1]
            cand = at(date(y, m, min(self.day_of_month or 1, last)))
            if cand > after:
                return cand
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        raise ValueError("次の日時を計算できませんでした")


class Store:
    def __init__(self, path: Path, tz: ZoneInfo) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._migrate()
        self.db.commit()
        self.tz = tz

    def _migrate(self) -> None:
        """前の版で作ったデータベースに、後から追加した列を足す。"""
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(guild_settings)")}
        if "followup_level" not in cols:
            self.db.execute("ALTER TABLE guild_settings ADD COLUMN followup_level TEXT")

    # ---- 時刻の変換 ----
    def parse_local(self, text: str) -> int:
        """'2026-11-25T09:00' のような日時（タイムゾーンなしなら設定のタイムゾーン）を UNIX 時間にする。"""
        dt = datetime.fromisoformat(text.strip().replace(" ", "T"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        return int(dt.timestamp())

    def fmt(self, ts: int | None) -> str | None:
        if ts is None:
            return None
        dt = datetime.fromtimestamp(ts, self.tz)
        return f"{dt:%Y-%m-%d}({WEEKDAYS_JA[dt.weekday()]}) {dt:%H:%M}"

    def now(self) -> datetime:
        return datetime.now(self.tz)

    # ---- リマインダー・定期投稿 ----
    def add_reminder(
        self, *, guild_id: int, channel_id: int, user_id: int, text: str, due_at: int,
        rule: Rule | None = None, mention_user: bool = True,
    ) -> int:
        cur = self.db.execute(
            "INSERT INTO reminders (guild_id, channel_id, user_id, mention_user, text, due_at, rule, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (guild_id, channel_id, user_id, int(mention_user), text, due_at, rule.to_json() if rule else None, int(time.time())),
        )
        self.db.commit()
        return int(cur.lastrowid or 0)

    def list_reminders(self, guild_id: int) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM reminders WHERE guild_id = ? AND active = 1 ORDER BY due_at", (guild_id,)
        ).fetchall()
        return [self._reminder_view(r) for r in rows]

    def _reminder_view(self, r: sqlite3.Row) -> dict[str, Any]:
        rule = Rule.from_json(r["rule"]) if r["rule"] else None
        return {
            "id": r["id"],
            "text": r["text"],
            "next": self.fmt(r["due_at"]),
            "repeat": rule.describe() if rule else "なし（1回だけ）",
            "channel_id": r["channel_id"],
            "mention_user_id": r["user_id"] if r["mention_user"] else None,
        }

    def cancel_reminder(self, guild_id: int, reminder_id: int) -> bool:
        cur = self.db.execute(
            "UPDATE reminders SET active = 0 WHERE id = ? AND guild_id = ? AND active = 1", (reminder_id, guild_id)
        )
        self.db.commit()
        return cur.rowcount > 0

    def due_reminders(self, now_ts: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM reminders WHERE active = 1 AND due_at <= ? ORDER BY due_at", (now_ts,)
        ).fetchall()

    def finish_reminder(self, row: sqlite3.Row) -> None:
        """送信後の処理。繰り返しなら次の時刻へ進め、1回だけなら無効にする。"""
        if row["rule"]:
            rule = Rule.from_json(row["rule"])
            nxt = rule.next_after(max(self.now(), datetime.fromtimestamp(row["due_at"], self.tz)))
            self.db.execute("UPDATE reminders SET due_at = ? WHERE id = ?", (int(nxt.timestamp()), row["id"]))
        else:
            self.db.execute("UPDATE reminders SET active = 0 WHERE id = ?", (row["id"],))
        self.db.commit()

    # ---- 予定 ----
    def add_event(
        self, *, guild_id: int, channel_id: int, user_id: int, title: str, start_at: int,
        end_at: int | None, location: str | None, notes: str | None, notify_before_min: int | None,
    ) -> int:
        cur = self.db.execute(
            "INSERT INTO events (guild_id, channel_id, user_id, title, start_at, end_at, location, notes,"
            " notify_before_min, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (guild_id, channel_id, user_id, title, start_at, end_at, location, notes, notify_before_min, int(time.time())),
        )
        self.db.commit()
        return int(cur.lastrowid or 0)

    def list_events(self, guild_id: int, start: int, end: int) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM events WHERE guild_id = ? AND start_at >= ? AND start_at < ? ORDER BY start_at",
            (guild_id, start, end),
        ).fetchall()
        return [
            {
                "id": r["id"], "title": r["title"], "start": self.fmt(r["start_at"]), "end": self.fmt(r["end_at"]),
                "location": r["location"], "notes": r["notes"],
                "notify_before_min": r["notify_before_min"],
            }
            for r in rows
        ]

    def delete_event(self, guild_id: int, event_id: int) -> bool:
        cur = self.db.execute("DELETE FROM events WHERE id = ? AND guild_id = ?", (event_id, guild_id))
        self.db.commit()
        return cur.rowcount > 0

    def events_to_notify(self, now_ts: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM events WHERE notified = 0 AND notify_before_min IS NOT NULL"
            " AND start_at - notify_before_min * 60 <= ? AND start_at > ? - 3600",
            (now_ts, now_ts),
        ).fetchall()

    def mark_event_notified(self, event_id: int) -> None:
        self.db.execute("UPDATE events SET notified = 1 WHERE id = ?", (event_id,))
        self.db.commit()

    # ---- 記憶 ----
    def add_memory(self, guild_id: int, kind: str, content: str, user_id: int | None) -> int:
        cur = self.db.execute(
            "INSERT INTO memories (guild_id, kind, content, user_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (guild_id, kind, content, user_id, int(time.time())),
        )
        self.db.commit()
        return int(cur.lastrowid or 0)

    def list_memories(self, guild_id: int, kind: str, limit: int) -> list[sqlite3.Row]:
        rows = self.db.execute(
            "SELECT * FROM memories WHERE guild_id = ? AND kind = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (guild_id, kind, limit),
        ).fetchall()
        return list(reversed(rows))  # 古い順に並べる

    def delete_memory(self, guild_id: int, memory_id: int) -> bool:
        cur = self.db.execute("DELETE FROM memories WHERE id = ? AND guild_id = ?", (memory_id, guild_id))
        self.db.commit()
        return cur.rowcount > 0

    # ---- 発言の記録 ----
    def log_message(self, m: LoggedMessage) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO messages (id, guild_id, channel_id, author_id, author_name, is_bot, is_self,"
            " content, reply_author, reply_content, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (m.id, m.guild_id, m.channel_id, m.author_id, m.author_name, int(m.is_bot), int(m.is_self),
             m.content, m.reply_author, m.reply_content, m.created_at),
        )
        self.db.commit()

    def log_messages(self, items: list[LoggedMessage]) -> int:
        """まとめて記録する。既にある発言は上書きしない。追加した件数を返す。"""
        before = self.db.total_changes
        self.db.executemany(
            "INSERT OR IGNORE INTO messages (id, guild_id, channel_id, author_id, author_name, is_bot, is_self,"
            " content, reply_author, reply_content, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(m.id, m.guild_id, m.channel_id, m.author_id, m.author_name, int(m.is_bot), int(m.is_self),
              m.content, m.reply_author, m.reply_content, m.created_at) for m in items],
        )
        self.db.commit()
        return self.db.total_changes - before

    def edit_message(self, message_id: int, content: str) -> None:
        self.db.execute("UPDATE messages SET content = ? WHERE id = ?", (content, message_id))
        self.db.commit()

    def delete_message(self, message_id: int) -> None:
        self.db.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        self.db.commit()

    @staticmethod
    def _row_to_message(r: sqlite3.Row) -> LoggedMessage:
        return LoggedMessage(
            id=r["id"], guild_id=r["guild_id"], channel_id=r["channel_id"], author_id=r["author_id"],
            author_name=r["author_name"], is_bot=bool(r["is_bot"]), is_self=bool(r["is_self"]),
            content=r["content"], reply_author=r["reply_author"], reply_content=r["reply_content"],
            created_at=r["created_at"],
        )

    def counts(self) -> dict[str, int]:
        """保存されている件数を返す（起動時のログ用。ボリュームにデータが残っているかの確認に使う）。"""
        q = lambda sql: self.db.execute(sql).fetchone()[0]
        return {
            "発言": q("SELECT COUNT(*) FROM messages"),
            "記憶": q("SELECT COUNT(*) FROM memories WHERE kind = 'fact'"),
            "要約": q("SELECT COUNT(*) FROM memories WHERE kind = 'summary'"),
            "リマインダー": q("SELECT COUNT(*) FROM reminders"),
            "予定": q("SELECT COUNT(*) FROM events"),
            "人物ノート": q("SELECT COUNT(*) FROM people WHERE note IS NOT NULL"),
            "気にかけること": q("SELECT COUNT(*) FROM followups WHERE status = 'pending'"),
        }

    def recent_messages(self, channel_id: int, limit: int, since: int | None = None) -> list[LoggedMessage]:
        """チャンネルの直近の発言を古い順に返す。since を指定するとその時刻以降だけにする。"""
        rows = self.db.execute(
            "SELECT * FROM messages WHERE channel_id = ? AND created_at >= ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (channel_id, since or 0, limit),
        ).fetchall()
        return [self._row_to_message(r) for r in reversed(rows)]

    def search_messages(
        self, guild_id: int, keywords: list[str], *, since: int | None = None, limit: int = 20
    ) -> list[LoggedMessage]:
        """サーバー内の発言を、全てのキーワードを含むものに絞って新しい順に返す。"""
        words = [w for w in keywords if w][:5]
        if not words:
            return []
        cond = " AND ".join("content LIKE ? ESCAPE '\\'" for _ in words)
        params: list[Any] = [guild_id, since or 0]
        params += ["%" + w.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%" for w in words]
        rows = self.db.execute(
            f"SELECT * FROM messages WHERE guild_id = ? AND created_at >= ? AND {cond}"
            " ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def prune_messages(self, older_than: int) -> int:
        cur = self.db.execute("DELETE FROM messages WHERE created_at < ?", (older_than,))
        self.db.commit()
        return cur.rowcount

    # ---- 返信スタイル ----
    def get_style(self, guild_id: int) -> str | None:
        row = self.db.execute("SELECT reply_style FROM guild_settings WHERE guild_id = ?", (guild_id,)).fetchone()
        return row["reply_style"] if row else None

    def set_style(self, guild_id: int, style: str | None) -> None:
        self.db.execute(
            "INSERT INTO guild_settings (guild_id, reply_style) VALUES (?, ?)"
            " ON CONFLICT(guild_id) DO UPDATE SET reply_style = excluded.reply_style",
            (guild_id, style),
        )
        self.db.commit()

    # ---- 気にかけることの頻度 ----
    def get_followup_level(self, guild_id: int) -> str:
        row = self.db.execute("SELECT followup_level FROM guild_settings WHERE guild_id = ?", (guild_id,)).fetchone()
        return (row["followup_level"] if row else None) or "normal"

    def set_followup_level(self, guild_id: int, level: str) -> None:
        self.db.execute(
            "INSERT INTO guild_settings (guild_id, followup_level) VALUES (?, ?)"
            " ON CONFLICT(guild_id) DO UPDATE SET followup_level = excluded.followup_level",
            (guild_id, level),
        )
        self.db.commit()

    # ---- 人物ノート ----
    def touch_person(self, guild_id: int, user_id: int, name: str) -> None:
        """表示名を記録する（ノートがまだなければ空のノートを作る）。"""
        self.db.execute(
            "INSERT INTO people (guild_id, user_id, name, updated_at) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(guild_id, user_id) DO UPDATE SET name = excluded.name",
            (guild_id, user_id, name, int(time.time())),
        )
        self.db.commit()

    def get_people(self, guild_id: int, user_ids: list[int]) -> list[sqlite3.Row]:
        if not user_ids:
            return []
        marks = ",".join("?" * len(user_ids))
        return self.db.execute(
            f"SELECT * FROM people WHERE guild_id = ? AND user_id IN ({marks}) ORDER BY user_id",
            (guild_id, *user_ids),
        ).fetchall()

    def list_people(self, guild_id: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM people WHERE guild_id = ? AND note IS NOT NULL ORDER BY name", (guild_id,)
        ).fetchall()

    def find_person(self, guild_id: int, name: str) -> sqlite3.Row | None:
        """表示名から人を探す。完全一致を優先し、なければ部分一致が1人だけのときに返す。"""
        name = name.strip().lstrip("@")
        rows = self.db.execute("SELECT * FROM people WHERE guild_id = ?", (guild_id,)).fetchall()
        exact = [r for r in rows if r["name"].lower() == name.lower()]
        if exact:
            return exact[0]
        partial = [r for r in rows if name and (name.lower() in r["name"].lower() or r["name"].lower() in name.lower())]
        return partial[0] if len(partial) == 1 else None

    def set_note(self, guild_id: int, user_id: int, name: str, note: str | None) -> None:
        self.db.execute(
            "INSERT INTO people (guild_id, user_id, name, note, updated_at) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(guild_id, user_id) DO UPDATE SET name = excluded.name, note = excluded.note,"
            " updated_at = excluded.updated_at",
            (guild_id, user_id, name, note, int(time.time())),
        )
        self.db.commit()

    def set_no_followups(self, guild_id: int, user_id: int, name: str, value: bool) -> int:
        """声かけを受けない設定にする。受けない設定にした場合は、その人の残っている声かけを取り消し、件数を返す。"""
        self.touch_person(guild_id, user_id, name)
        self.db.execute(
            "UPDATE people SET no_followups = ? WHERE guild_id = ? AND user_id = ?", (int(value), guild_id, user_id)
        )
        cancelled = 0
        if value:
            cancelled = self.db.execute(
                "UPDATE followups SET status = 'cancelled' WHERE guild_id = ? AND user_id = ? AND status = 'pending'",
                (guild_id, user_id),
            ).rowcount
        self.db.commit()
        return cancelled

    def wants_followups(self, guild_id: int, user_id: int) -> bool:
        row = self.db.execute(
            "SELECT no_followups FROM people WHERE guild_id = ? AND user_id = ?", (guild_id, user_id)
        ).fetchone()
        return not (row and row["no_followups"])

    # ---- 気にかけること ----
    def add_followup(
        self, *, guild_id: int, channel_id: int, user_id: int, user_name: str, topic: str, check_at: int,
        context: str | None = None, source: str = "chat", created_by: int | None = None,
    ) -> int:
        cur = self.db.execute(
            "INSERT INTO followups (guild_id, channel_id, user_id, user_name, topic, context, check_at, source,"
            " created_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (guild_id, channel_id, user_id, user_name, topic, context, check_at, source, created_by, int(time.time())),
        )
        self.db.commit()
        return int(cur.lastrowid or 0)

    def list_followups(self, guild_id: int) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM followups WHERE guild_id = ? AND status = 'pending' ORDER BY check_at", (guild_id,)
        ).fetchall()
        return [
            {"id": r["id"], "who": r["user_name"], "topic": r["topic"], "when": self.fmt(r["check_at"])}
            for r in rows
        ]

    def cancel_followup(self, guild_id: int, followup_id: int) -> bool:
        cur = self.db.execute(
            "UPDATE followups SET status = 'cancelled' WHERE id = ? AND guild_id = ? AND status = 'pending'",
            (followup_id, guild_id),
        )
        self.db.commit()
        return cur.rowcount > 0

    def cancel_followups_from(self, source: str) -> None:
        """リマインダーや予定を取り消したときに、そこから作った声かけも取り消す。"""
        self.db.execute("UPDATE followups SET status = 'cancelled' WHERE source = ? AND status = 'pending'", (source,))
        self.db.commit()

    def due_followups(self, now_ts: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM followups WHERE status = 'pending' AND check_at <= ? ORDER BY check_at", (now_ts,)
        ).fetchall()

    def sent_followups_since(self, guild_id: int, since: int, user_id: int | None = None) -> int:
        sql = "SELECT COUNT(*) FROM followups WHERE guild_id = ? AND status = 'sent' AND sent_at >= ?"
        params: list[Any] = [guild_id, since]
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        return int(self.db.execute(sql, params).fetchone()[0])

    def postpone_followup(self, followup_id: int, check_at: int) -> None:
        self.db.execute("UPDATE followups SET check_at = ? WHERE id = ?", (check_at, followup_id))
        self.db.commit()

    def finish_followup(self, followup_id: int, status: str) -> None:
        self.db.execute(
            "UPDATE followups SET status = ?, sent_at = ? WHERE id = ?", (status, int(time.time()), followup_id)
        )
        self.db.commit()

    def messages_by(self, channel_id: int, user_id: int, since: int, limit: int = 20) -> list[LoggedMessage]:
        """チャンネルでのある人の発言を、since 以降に限って古い順に返す。"""
        rows = self.db.execute(
            "SELECT * FROM messages WHERE channel_id = ? AND author_id = ? AND created_at >= ?"
            " ORDER BY created_at DESC LIMIT ?",
            (channel_id, user_id, since, limit),
        ).fetchall()
        return [self._row_to_message(r) for r in reversed(rows)]
