import ast
import sys
from pathlib import Path

import pytest

from config.paths import PROJECT_ROOT


@pytest.mark.parametrize(
    "layer,allowed",
    [
        ("domain", {"domain"}),
        ("port", {"port", "domain"}),
        ("application", {"application", "domain", "port"}),
    ],
)
def test_inner_layers_do_not_import_frameworks_or_adapters(layer: str, allowed: set[str]) -> None:
    for path in (PROJECT_ROOT / "src" / layer).rglob("*.py"):
        imports = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])
        assert set(imports) <= allowed | sys.stdlib_module_names, str(Path(path).relative_to(PROJECT_ROOT))
