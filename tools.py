"""DeepSeek に渡す関数（ツール）の定義と、その実行処理。

DeepSeek は「どの関数をどの引数で呼ぶか」を返すだけで、実際の処理はこのファイルの bot 側で行う。
Web 検索・URL 読み取りは、DeepSeek の Search API / Fetch API を bot から呼ぶ。
（DeepSeek のサーバー側 Web 検索は、自作の関数と同時に使うと thinking モードが必須になるため使わない）
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import aiohttp

from store import Rule, Store

log = logging.getLogger("oniya-bot.tools")

SEARCH_URL = "https://api.deepseek.com/v1/tools/search"
FETCH_URL = "https://api.deepseek.com/v1/tools/fetch"
FETCH_MAX_CHARS = 12000  # 読み取ったページのうち、モデルに渡す最大文字数
FOLLOWUP_AFTER_REMINDER_MIN = 180  # リマインダーを送ってから「ちゃんとやったか」を聞くまでの分数
FOLLOWUP_AFTER_EVENT_MIN = 120  # 終了時刻のない予定で、開始から「どうだった」を聞くまでの分数
NOTE_MAX_CHARS = 400  # 人物ノートの最大文字数


@dataclass
class ToolContext:
    guild_id: int
    channel_id: int
    user_id: int
    user_name: str
    # 渡した会話に出てくる人（bot を除く）の表示名と ID。「誰について」の名前を ID に直すのに使う
    participants: dict[str, int] = field(default_factory=dict)


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


DATETIME_DESC = "日時。'YYYY-MM-DDTHH:MM' の形式（bot の設定のタイムゾーンとして扱う）"

TOOL_DEFS: list[dict[str, Any]] = [
    _fn(
        "web_search",
        "Webを検索する。最新の情報、ニュース、価格、日付、固有名詞、数値、事実関係を確かめるときに使う。",
        {
            "query": {"type": "string", "description": "検索語（128文字以内）。site: や \"\" も使える"},
            "count": {"type": "integer", "description": "取得件数（1〜10、既定5）"},
        },
        ["query"],
    ),
    _fn(
        "read_url",
        "指定したURLのページを読み、本文をMarkdownで受け取る。",
        {"url": {"type": "string"}},
        ["url"],
    ),
    _fn(
        "create_reminder",
        "指定した日時に1回だけ、このチャンネルでリマインドを送る。",
        {
            "when": {"type": "string", "description": DATETIME_DESC},
            "text": {"type": "string", "description": "リマインドの内容"},
            "mention_user": {"type": "boolean", "description": "依頼した人にメンションするか（既定 true）"},
            "follow_up": {
                "type": "boolean",
                "description": "リマインドの数時間後に、ちゃんとやったかを依頼した人に聞くか。"
                "やるべきこと（提出、支払い、連絡など）なら true、単なるお知らせなら false（既定 true）",
            },
        },
        ["when", "text"],
    ),
    _fn(
        "create_recurring_post",
        "決まった周期で、このチャンネルにメッセージを繰り返し投稿する（定期投稿・繰り返しのリマインダー）。",
        {
            "text": {"type": "string", "description": "投稿する内容"},
            "freq": {"type": "string", "enum": ["daily", "weekly", "monthly"]},
            "time": {"type": "string", "description": "投稿する時刻 'HH:MM'"},
            "weekdays": {
                "type": "array", "items": {"type": "integer"},
                "description": "freq が weekly のときの曜日。0=月曜, 1=火曜, ..., 6=日曜",
            },
            "day_of_month": {"type": "integer", "description": "freq が monthly のときの日（1〜31）"},
            "mention_user": {"type": "boolean", "description": "依頼した人にメンションするか（既定 false）"},
        },
        ["text", "freq", "time"],
    ),
    _fn(
        "search_messages",
        "このサーバーの過去の発言を検索する。渡された会話より前の話を持ち出されたときに使う。"
        "キーワードを全て含む発言を新しい順に返す。",
        {
            "keywords": {"type": "array", "items": {"type": "string"}, "description": "検索する語（1〜5個）"},
            "days": {"type": "integer", "description": "何日前までさかのぼるか（既定 90）"},
        },
        ["keywords"],
    ),
    _fn("list_reminders", "このサーバーで有効なリマインダーと定期投稿の一覧を取得する。", {}, []),
    _fn(
        "cancel_reminder",
        "リマインダーまたは定期投稿を取り消す。id は list_reminders で確認する。",
        {"id": {"type": "integer"}},
        ["id"],
    ),
    _fn(
        "add_event",
        "サーバーの予定を登録する。",
        {
            "title": {"type": "string"},
            "start": {"type": "string", "description": "開始日時。" + DATETIME_DESC},
            "end": {"type": "string", "description": "終了日時（任意）。" + DATETIME_DESC},
            "location": {"type": "string"},
            "notes": {"type": "string"},
            "notify_before_min": {
                "type": "integer",
                "description": "開始の何分前にこのチャンネルで知らせるか。知らせない場合は指定しない",
            },
            "follow_up": {
                "type": "boolean",
                "description": "予定が終わった後に、登録した人にどうだったかを聞くか（既定 true）",
            },
        },
        ["title", "start"],
    ),
    _fn(
        "list_events",
        "指定した期間に始まるサーバーの予定を取得する。",
        {
            "from": {"type": "string", "description": "期間の開始。" + DATETIME_DESC},
            "to": {"type": "string", "description": "期間の終了。" + DATETIME_DESC},
        },
        ["from", "to"],
    ),
    _fn("delete_event", "予定を削除する。id は list_events で確認する。", {"id": {"type": "integer"}}, ["id"]),
    _fn(
        "add_followup",
        "あとで本人に様子を聞くこと（気にかけること）を登録する。会話の中で、日時の見当がつく予定・締め切り・"
        "宣言（「金曜に面接」「明日レポート出す」「来週から筋トレする」）が出たときに使う。"
        "登録した時刻に、あなたが本人に「どうだった？」「ちゃんとやったか？」と声をかける。"
        "体調・病気・家族・恋愛・お金の悩みなど立ち入った話題や、冗談・仮定の話では登録しない。"
        "登録したことは返信に書かない。",
        {
            "topic": {"type": "string", "description": "何について聞くか。日付を入れる（例: 11/27(金)の面接、11/30締め切りのレポート提出）"},
            "when": {
                "type": "string",
                "description": "声をかける日時。その予定が終わった頃にする（例: 面接が午後なら同じ日の20:00、"
                "締め切りならその少し後）。" + DATETIME_DESC,
            },
            "about": {"type": "string", "description": "誰に聞くか（会話に出てくる表示名）。省略すると発言した本人"},
            "context": {"type": "string", "description": "きっかけになった発言の要点（1文）"},
        },
        ["topic", "when"],
    ),
    _fn("list_followups", "このサーバーで登録されている「気にかけること」の一覧を取得する。", {}, []),
    _fn(
        "cancel_followup",
        "「気にかけること」を取り消す。予定がなくなった、もう終わって報告を受けた、と分かったときに使う。"
        "id は list_followups で確認する。",
        {"id": {"type": "integer"}},
        ["id"],
    ),
    _fn(
        "set_followups_for_me",
        "発言した本人が、あなたからの声かけ（「どうだった？」など）を受けるかを設定する。"
        "「そういうのいらない」「聞かないで」と言われたら false、「また聞いて」と言われたら true にする。",
        {"enabled": {"type": "boolean"}},
        ["enabled"],
    ),
    _fn(
        "update_person_note",
        "人物ノート（1人につき1枚のメモ）を書き直す。人についての情報（呼び方、所属、好み、取り組み中のこと、"
        "最近の出来事）を覚えてと頼まれたとき、または今後も役に立つ情報が出たときに使う。"
        "今のノートの内容に新しい情報を合わせ、古くなった情報は消して、全文を渡す。"
        "本人が自分のノートを消してと言った場合は空文字を渡す。",
        {
            "about": {"type": "string", "description": "誰のノートか（会話に出てくる表示名）。省略すると発言した本人"},
            "note": {"type": "string", "description": f"ノートの全文（{NOTE_MAX_CHARS}字以内、箇条書き）"},
        },
        ["note"],
    ),
    _fn(
        "remember",
        "後の会話でも使う、サーバー全体に関わる情報を長期の記憶に保存する（決まったこと、内輪のルール、恒例の行事など）。"
        "特定の人についての情報は update_person_note を使う。",
        {"content": {"type": "string", "description": "保存する内容（1〜2文）"}},
        ["content"],
    ),
    _fn(
        "forget",
        "長期の記憶から1件削除する。id はシステムの指示にある記憶の一覧で確認する。",
        {"id": {"type": "integer"}},
        ["id"],
    ),
    _fn(
        "set_reply_style",
        "このサーバーでの返信の口調・形式の指定を保存する（以後の全ての返信に適用される）。"
        "返信の仕方を変えてと頼まれたときに使う。既存の指定に追記する場合は、既存の内容も含めた全文を渡す。"
        "空文字を渡すと指定を消す。",
        {"style": {"type": "string"}},
        ["style"],
    ),
]


class ToolRunner:
    def __init__(self, api_key: str, store: Store) -> None:
        self.api_key = api_key
        self.store = store
        self._session: aiohttp.ClientSession | None = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=60),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def run(self, name: str, arguments: str, ctx: ToolContext) -> str:
        """関数を実行し、結果を JSON 文字列で返す。失敗しても例外は投げず、エラー内容を返す。"""
        try:
            args = json.loads(arguments or "{}")
            handler = getattr(self, f"_t_{name}", None)
            if handler is None:
                raise ValueError(f"不明な関数です: {name}")
            result = await handler(args, ctx)
        except Exception as e:  # モデルに失敗を伝えて、言い直しや説明をさせる
            log.warning("ツール %s の実行に失敗しました: %s", name, e)
            result = {"error": str(e)}
        log.info("tool %s(%s) -> %s", name, arguments, str(result)[:200])
        return json.dumps(result, ensure_ascii=False)

    # ---- Web ----
    async def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        s = await self.session()
        async with s.post(url, json=body) as r:
            data = await r.json(content_type=None)
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {data}")
            return data

    async def _t_web_search(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        count = max(1, min(int(a.get("count") or 5), 10))
        data = await self._post(SEARCH_URL, {"query": str(a["query"])[:128], "count": count})
        return [
            {k: r.get(k) for k in ("title", "url", "date", "site_name", "snippet", "summary") if r.get(k)}
            for r in data.get("results", [])
        ]

    async def _t_read_url(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        data = await self._post(FETCH_URL, {"url": str(a["url"])})
        content = data.get("content") or ""
        if len(content) > FETCH_MAX_CHARS:
            content = content[:FETCH_MAX_CHARS] + "\n…（以下省略）"
        return {"title": data.get("title"), "content": content}

    # ---- 過去の発言 ----
    async def _t_search_messages(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        days = max(1, int(a.get("days") or 90))
        since = int(self.store.now().timestamp()) - days * 86400
        found = self.store.search_messages(ctx.guild_id, [str(k) for k in a.get("keywords") or []], since=since)
        return [
            {"when": self.store.fmt(m.created_at), "author": m.author_name, "channel_id": m.channel_id,
             "content": m.content[:500]}
            for m in found
        ]

    # ---- リマインダー・定期投稿 ----
    async def _t_create_reminder(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        due = self.store.parse_local(a["when"])
        if due <= self.store.now().timestamp():
            raise ValueError("過去の日時は指定できません")
        rid = self.store.add_reminder(
            guild_id=ctx.guild_id, channel_id=ctx.channel_id, user_id=ctx.user_id, text=a["text"],
            due_at=due, mention_user=bool(a.get("mention_user", True)),
        )
        out: dict[str, Any] = {"ok": True, "id": rid, "when": self.store.fmt(due)}
        if a.get("follow_up", True):
            self.store.add_followup(
                guild_id=ctx.guild_id, channel_id=ctx.channel_id, user_id=ctx.user_id, user_name=ctx.user_name,
                topic=a["text"], check_at=due + FOLLOWUP_AFTER_REMINDER_MIN * 60,
                context=f"{self.store.fmt(due)} にリマインドした内容", source=f"reminder:{rid}", created_by=ctx.user_id,
            )
        return out

    async def _t_create_recurring_post(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        hour, minute = (int(x) for x in str(a["time"]).split(":"))
        rule = Rule(
            freq=a["freq"], hour=hour, minute=minute,
            weekdays=a.get("weekdays") if a["freq"] == "weekly" else None,
            day_of_month=a.get("day_of_month") if a["freq"] == "monthly" else None,
        )
        rule.validate()
        first = rule.next_after(self.store.now())
        rid = self.store.add_reminder(
            guild_id=ctx.guild_id, channel_id=ctx.channel_id, user_id=ctx.user_id, text=a["text"],
            due_at=int(first.timestamp()), rule=rule, mention_user=bool(a.get("mention_user", False)),
        )
        return {"ok": True, "id": rid, "repeat": rule.describe(), "first": self.store.fmt(int(first.timestamp()))}

    async def _t_list_reminders(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        return self.store.list_reminders(ctx.guild_id)

    async def _t_cancel_reminder(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        ok = self.store.cancel_reminder(ctx.guild_id, int(a["id"]))
        if ok:
            self.store.cancel_followups_from(f"reminder:{int(a['id'])}")
        return {"ok": ok}

    # ---- 予定 ----
    async def _t_add_event(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        start = self.store.parse_local(a["start"])
        end = self.store.parse_local(a["end"]) if a.get("end") else None
        if end is not None and end < start:
            raise ValueError("終了日時が開始日時より前です")
        notify = a.get("notify_before_min")
        eid = self.store.add_event(
            guild_id=ctx.guild_id, channel_id=ctx.channel_id, user_id=ctx.user_id, title=a["title"],
            start_at=start, end_at=end, location=a.get("location"), notes=a.get("notes"),
            notify_before_min=int(notify) if notify is not None else None,
        )
        if a.get("follow_up", True):
            self.store.add_followup(
                guild_id=ctx.guild_id, channel_id=ctx.channel_id, user_id=ctx.user_id, user_name=ctx.user_name,
                topic=a["title"], check_at=end if end else start + FOLLOWUP_AFTER_EVENT_MIN * 60,
                context=f"{self.store.fmt(start)} からの予定" + (f"（場所: {a['location']}）" if a.get("location") else ""),
                source=f"event:{eid}", created_by=ctx.user_id,
            )
        return {"ok": True, "id": eid, "start": self.store.fmt(start)}

    async def _t_list_events(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        start = self.store.parse_local(a["from"])
        end = self.store.parse_local(a["to"])
        if end <= start:
            end = start + int(timedelta(days=1).total_seconds())
        return self.store.list_events(ctx.guild_id, start, end)

    async def _t_delete_event(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        ok = self.store.delete_event(ctx.guild_id, int(a["id"]))
        if ok:
            self.store.cancel_followups_from(f"event:{int(a['id'])}")
        return {"ok": ok}

    # ---- 人 ----
    def _resolve(self, about: Any, ctx: ToolContext) -> tuple[int, str]:
        """「誰について」の表示名を (ID, 表示名) にする。省略されたら発言した本人。"""
        name = str(about or "").strip().lstrip("@")
        if not name or name == ctx.user_name:
            return ctx.user_id, ctx.user_name
        for n, uid in ctx.participants.items():
            if n.lower() == name.lower():
                return uid, n
        partial = [(n, uid) for n, uid in ctx.participants.items() if name.lower() in n.lower()]
        if len(partial) == 1:
            return partial[0][1], partial[0][0]
        row = self.store.find_person(ctx.guild_id, name)
        if row:
            return row["user_id"], row["name"]
        known = "、".join(ctx.participants) or "なし"
        raise ValueError(f"「{name}」が誰か分かりません。会話に出てくる人: {known}")

    # ---- 気にかけること ----
    async def _t_add_followup(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        when = self.store.parse_local(a["when"])
        if when <= self.store.now().timestamp():
            raise ValueError("過去の日時は指定できません")
        uid, name = self._resolve(a.get("about"), ctx)
        if not self.store.wants_followups(ctx.guild_id, uid):
            return {"ok": False, "reason": f"{name} は声かけを受けない設定にしています"}
        fid = self.store.add_followup(
            guild_id=ctx.guild_id, channel_id=ctx.channel_id, user_id=uid, user_name=name,
            topic=str(a["topic"]), check_at=when, context=a.get("context"), created_by=ctx.user_id,
        )
        return {"ok": True, "id": fid, "who": name, "when": self.store.fmt(when)}

    async def _t_list_followups(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        return self.store.list_followups(ctx.guild_id)

    async def _t_cancel_followup(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        return {"ok": self.store.cancel_followup(ctx.guild_id, int(a["id"]))}

    async def _t_set_followups_for_me(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        enabled = bool(a.get("enabled"))
        cancelled = self.store.set_no_followups(ctx.guild_id, ctx.user_id, ctx.user_name, not enabled)
        return {"ok": True, "enabled": enabled, "cancelled": cancelled}

    async def _t_update_person_note(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        uid, name = self._resolve(a.get("about"), ctx)
        note = str(a.get("note") or "").strip()[:NOTE_MAX_CHARS]
        self.store.set_note(ctx.guild_id, uid, name, note or None)
        return {"ok": True, "who": name, "cleared": not note}

    # ---- 記憶・返信スタイル ----
    async def _t_remember(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        content = str(a["content"]).strip()
        if not content:
            raise ValueError("内容が空です")
        return {"ok": True, "id": self.store.add_memory(ctx.guild_id, "fact", content, ctx.user_id)}

    async def _t_forget(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        return {"ok": self.store.delete_memory(ctx.guild_id, int(a["id"]))}

    async def _t_set_reply_style(self, a: dict[str, Any], ctx: ToolContext) -> Any:
        style = str(a.get("style") or "").strip()
        self.store.set_style(ctx.guild_id, style or None)
        return {"ok": True, "style": style or None}
