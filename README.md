# oniya-bot

DeepSeek API を使って返事をする Discord bot。

## 動き方

- bot へのメンション、または bot のメッセージへの返信で会話が始まる。
- 会話中は、メンションがなくても、返事が必要なメッセージかを DeepSeek に判定させて返事をする。
- 会話は次のどちらかで終わる。終わった後は、またメンションするまで返事をしない。
  - bot が返事をしないまま `CONVERSATION_TIMEOUT` 秒（既定は10分）たった
  - お礼や締めの言葉などで、会話が終わったと判定された
- 最新の情報や事実確認が必要な質問では、DeepSeek のサーバー側 Web 検索を使い、末尾に参考リンクを付ける。
- 回答が出力上限で止まった場合は続きを生成し、2000文字ごとに分けて全て送る。

コマンド:
- `/ask question:質問` : スラッシュコマンドで質問する
- `/reset` : そのチャンネルの会話履歴を消して、会話を終える

## 起動

`.env` に `DISCORD_TOKEN` と `DEEPSEEK_API_KEY` を書いてから実行する（Railway では Variables に設定する）。

```bash
uv run main.py
```

PC と Railway で同時に動かすと、同じメッセージに2回返事をするので注意。

## 前提

- Discord Developer Portal の Bot タブで **Message Content Intent** をオンにしていること。
- DeepSeek の残高があること（残高がないと 402 エラーを返す）。

## 設定

省略可能な設定は `.env.example` を参照。回答の書き方の決まり（前置きを書かない、自分について触れない、調べてから答える）は `main.py` の `build_system_prompt` にある。

会話履歴はメモリ上にだけ持つので、bot を再起動すると消える。会話中のチャンネルのメッセージは、判定と回答のために DeepSeek に送られる。
