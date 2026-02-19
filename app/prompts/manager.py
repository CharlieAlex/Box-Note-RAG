from pathlib import Path
from typing import Any

import yaml

MAIN_PATH = Path(__file__).parents[2]
PROMPTS_DIR = MAIN_PATH / "app" / "prompts"


class PromptManager:
    def __init__(self, prompts_dir=PROMPTS_DIR):
        self.prompts_dir = prompts_dir
        self.prompts = self._load_all()

    def get(self, name: str, version: str = None) -> str:
        if name not in self.prompts:
            available = list(self.prompts.keys())
            raise ValueError(
                f"❌ Prompt '{name}' 不存在！\n"
                f"可用 prompts: {available}\n"
                f"檢查 templates.yaml 的 'prompts' 鍵名"
            )

        p = self.prompts[name]
        version = version or p.get("default")
        if version not in p["versions"]:
            raise ValueError(f"❌ Version '{version}' 不存在於 '{name}'！可用: {list(p['versions'].keys())}")

        return p["versions"][version]

    def _load_all(self) -> dict[str, Any]:
        prompts = {}
        template_file = self.prompts_dir / "templates.yaml"

        if not template_file.exists():
            raise FileNotFoundError(f"❌ templates.yaml 不存在！路徑: {template_file}")

        prompts = yaml.safe_load(template_file.read_text(encoding="utf-8"))

        return prompts
