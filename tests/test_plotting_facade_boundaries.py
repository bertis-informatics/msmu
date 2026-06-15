import ast
import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NON_PLOT_MODULES_WITH_PLOTTING_APIS = (
    REPO_ROOT / "msmu" / "_tools" / "_pyopenms.py",
    REPO_ROOT / "msmu" / "_statistics" / "_de_base.py",
)


def test_non_plot_modules_import_without_private_plotting_internals():
    importlib.import_module("msmu._tools._pyopenms")
    importlib.import_module("msmu._statistics._de_base")


def test_non_plot_modules_do_not_import_private_plotting_implementation_modules():
    forbidden_imports = []

    for path in NON_PLOT_MODULES_WITH_PLOTTING_APIS:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "_plotting._" in node.module:
                forbidden_imports.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} imports {node.module}")

    assert forbidden_imports == []
