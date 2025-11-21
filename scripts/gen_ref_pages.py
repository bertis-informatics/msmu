# scripts/gen_ref_pages.py
from pathlib import Path
import mkdocs_gen_files

PACKAGE = "msmu"  # ./msmu 레이아웃 가정
src_dir = Path(PACKAGE)
nav = mkdocs_gen_files.Nav()

# 폴더(패키지 디렉토리) → index.md 생성하여 섹션 라벨 고정
dir_titles_done = set()

for py in sorted(src_dir.rglob("*.py")):
    if py.name == "__init__.py":
        continue

    rel = py.relative_to(src_dir).with_suffix("")  # e.g. plotting/_pdata
    parts = list(rel.parts)  # ['plotting', '_pdata']
    ident = ".".join([PACKAGE] + parts)  # msmu.plotting._pdata

    # 폴더별 index.md (once per dir) 생성
    for i in range(1, len(parts)):
        dir_parts = parts[:i]
        if tuple(dir_parts) in dir_titles_done:
            continue
        dir_titles_done.add(tuple(dir_parts))

        index_doc = Path("reference", *dir_parts, "index.md")
        dir_title = dir_parts[-1]
        print(f"Generating index for {dir_title} at {index_doc}")
        with mkdocs_gen_files.open(index_doc, "w") as f:
            f.write("---\n")
            f.write(f"title: '{dir_title}'\n")
            f.write("---\n")

    # 모듈 페이지 생성
    mod_doc = Path("reference", *parts).with_suffix(".md")
    page_title = parts[-1]
    with mkdocs_gen_files.open(mod_doc, "w") as f:
        f.write("---\n")
        f.write(f"title: '{page_title}'\n")
        f.write("---\n\n")
        f.write(f"# `{ident}`\n\n::: {ident}\n")

    # Nav 계층 유지: msmu / plotting / _pdata 처럼 트리로 구성
    nav[parts] = Path("reference", *parts).with_suffix(".md").as_posix()

# 기존 nav.md 템플릿에 API 트리(indent) 붙이기
nav_template = Path("mkdocs", "nav.md").read_text()
if not nav_template.endswith("\n"):
    nav_template += "\n"

def format_api(line):
    print(f"Processing line: {line}")
    return "    " + line.replace("\\", "")

api_nav = [format_api(line) for line in nav.build_literate_nav()]
with mkdocs_gen_files.open("nav.md", "w") as nav_file:
    nav_file.write(nav_template)
    nav_file.writelines(api_nav)
