# scripts/gen_ref_pages.py
from pathlib import Path
import mkdocs_gen_files
import msmu
import inspect
import subprocess

PACKAGE = "msmu"  # ./msmu 레이아웃 가정
src_dir = Path(msmu.__file__).parent  # msmu/ 디렉토리
nav = mkdocs_gen_files.Nav()


def map_alias(name):
    if name == "pp":
        return "preprocessing: <b><code>pp</code></b>"
    if name == "pl":
        return "plotting: <b><code>pl</code></b>"
    if name == "tl":
        return "tools: <b><code>tl</code></b>"
    if name == "dt":
        return "data: <b><code>dt</code></b>"
    if name == "pv":
        return "provenance: <b><code>pv</code></b>"
    return name


def iterate_modules(parent, parent_alias=[]):
    if not hasattr(parent, "__all__"):
        return

    for module_name in parent.__all__:
        child = getattr(parent, module_name)
        if module_name.startswith("_"):
            continue

        if inspect.ismodule(child):
            yield from iterate_modules(child, parent_alias + [module_name])

        if inspect.isfunction(child) or inspect.isclass(child) or callable(child):
            parts = parent_alias + [module_name]  # ['module', 'function']
            ident = ".".join([PACKAGE] + parts)  # msmu.module.function

            doc = Path("reference", *parts).with_suffix(".md")

            with mkdocs_gen_files.open(doc, "w") as f:
                f.write("---\n")
                f.write(f"title: '{module_name}'\n")
                f.write("hide:\n")
                f.write("  - toc\n")
                f.write("---\n\n")
                f.write(f"# `{ident}`\n\n::: {ident}\n")

            nav[[map_alias(p) for p in parts]] = Path("reference", *parts).with_suffix(".md").as_posix()

            yield child.__name__, child


list(iterate_modules(msmu))

# Add indents to the generated nav.md
nav_template = Path("docs", "nav.md").read_text()
if not nav_template.endswith("\n"):
    nav_template += "\n"


def format_api(line):
    return "    " + line.replace("\\", "")


api_nav = [format_api(line) for line in nav.build_literate_nav()]
with mkdocs_gen_files.open("nav.md", "w") as nav_file:
    nav_file.write(nav_template)
    nav_file.writelines(api_nav)

# Show the exact package/source used to generate this documentation.
revision = subprocess.check_output(["git", "describe", "--always", "--dirty", "--exclude=*"], text=True).strip()
version_notice = (
    '!!! info "Documentation version"\n'
    f'    Built with `msmu` **{msmu.__version__}**, source revision **{revision}**. '
    'Compare with `mm.__version__` in your environment. '
    'Repository documentation follows its checkout; the published site follows release tags.\n'
)
with mkdocs_gen_files.open("index.md", "w") as index_file:
    index_file.write(Path("docs/index.md").read_text().replace("<!-- documentation-version -->", version_notice))
