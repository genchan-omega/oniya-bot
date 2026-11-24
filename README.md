# oniya-bot

DeepSeek API を使って返事をする Discord bot。

## 使い方

1. `.env` に `DISCORD_TOKEN` と `DEEPSEEK_API_KEY` を書く（書き方は `.env.example`）。
2. 起動する。

   ```bash
   uv run main.py
   ```

3. Discord で次のどれかを送る。
   - `@bot名 質問` : メンションで質問する
   - `/ask question:質問` : スラッシュコマンドで質問する
   - `/reset` : そのチャンネルの会話履歴を消す

止めるときはターミナルで `Ctrl + C` を押す。

## 前提

- Discord Developer Portal の Bot タブで **Message Content Intent** をオンにしていること（オフだとメンションの本文を読めない）。
- DeepSeek の残高があること（残高がないと 402 エラーを返す）。

## 設定（`.env`）

| 名前 | 必須 | 既定値 | 内容 |
| --- | --- | --- | --- |
| `DISCORD_TOKEN` | 必須 | なし | Discord bot の Token |
| `DEEPSEEK_API_KEY` | 必須 | なし | DeepSeek の API Key |
| `DEEPSEEK_MODEL` | 任意 | `deepseek-flash` | 使うモデル。高性能版は `deepseek-v4-pro` |
| `MAX_TOKENS` | 任意 | `1024` | 1回の回答の最大トークン数（費用の上限になる） |
| `HISTORY_TURNS` | 任意 | `10` | チャンネルごとに記憶する往復数 |
| `SYSTEM_PROMPT` | 任意 | 親切なアシスタント | bot の性格・口調の指示 |

会話履歴はメモリ上にだけ持つので、bot を再起動すると消える。
