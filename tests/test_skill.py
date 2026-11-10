"""The shipped agent skill must not reference APIs that do not exist.

Stale skills are how an agent ends up importing a module that was deleted; this
checks every ``from mlx_... import ...`` line in ``skills/mlx-bertopic/SKILL.md``
against the package.
"""

import importlib
import re
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1] / "skills" / "mlx-bertopic"
SKILL = SKILL_DIR / "SKILL.md"


def _frontmatter(text):
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    end = text.index("\n---\n", 4)
    return text[4:end]


def test_frontmatter_names_the_skill_after_its_directory():
    fm = _frontmatter(SKILL.read_text(encoding="utf-8"))
    assert re.search(rf"^name: {re.escape(SKILL_DIR.name)}$", fm, re.M)
    assert re.search(r"^description: \S.{40,}$", fm, re.M)


def test_referenced_mlx_apis_import():
    text = SKILL.read_text(encoding="utf-8")
    lines = re.findall(r"^from (mlx_[A-Za-z_.]+) import (.+)$", text, re.M)
    assert lines, "no 'from mlx_... import ...' lines found in SKILL.md"

    missing = []
    checked = 0
    for module_name, names in lines:
        module = importlib.import_module(module_name)
        for name in names.split(","):
            name = name.split(" as ")[0].strip()
            if not name:
                continue
            checked += 1
            if not hasattr(module, name):
                missing.append(f"{module_name}.{name}")

    assert checked >= 10, f"only {checked} API names checked, parse broke"
    assert not missing, f"SKILL.md references missing APIs: {missing}"
