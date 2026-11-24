"""prompts/ と profiles/ にあるプロンプトのファイルを読み込む。

ファイル中の <!-- --> のコメントは取り除き、{{name}} のような部分を値に置き換える。
ファイルは呼び出すたびに読み直すので、手元で動かしているときはファイルを書き換えると次の返事から反映される。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT / "prompts"
PROFILES_DIR = ROOT / "profiles"

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def render(text: str, **values: str) -> str:
    text = _COMMENT.sub("", text)
    return _VAR.sub(lambda m: values.get(m.group(1), m.group(0)), text).strip()


def load(path: Path, **values: str) -> str:
    return render(path.read_text(encoding="utf-8"), **values)


@dataclass
class Profile:
    key: str  # profiles/ の下のディレクトリ名
    name: str  # 人格の名前

    @property
    def dir(self) -> Path:
        return PROFILES_DIR / self.key

    def soul(self, **values: str) -> str:
        """人格の指示を返す。まだ埋めていない TODO の行は渡さない。"""
        path = self.dir / "SOUL.md"
        if not path.exists():
            return ""
        text = load(path, **{**values, "name": self.name})
        lines = [line for line in text.splitlines() if "TODO" not in line]
        return _drop_empty_sections("\n".join(lines))


def _drop_empty_sections(text: str) -> str:
    """箇条書きが1つもない「## 見出し」の節を取り除く（TODO を消した後に見出しだけ残るのを防ぐ）。"""
    sections = re.split(r"(?m)^(?=## )", text)
    kept = [s for s in sections if not s.startswith("## ") or re.search(r"(?m)^\s*- ", s)]
    return "".join(kept).strip()


def load_profile(key: str) -> Profile:
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", key):
        raise ValueError(f"人格の名前が正しくありません: {key}")
    path = PROFILES_DIR / key / "profile.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    name = str(data.get("name", "")).strip()
    if not name:
        raise ValueError(f"{path} に name がありません")
    return Profile(key=key, name=name)
