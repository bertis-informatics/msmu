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

    # 상위 디렉토리들에 대해 index.md 생성 (섹션 제목 고정)
    for d in range(len(parts)):
        dir_key = tuple(parts[:d])
        dir_path = Path("reference", *dir_key, "index.md")
        title = parts[d - 1] if d > 0 else PACKAGE

        if dir_key not in dir_titles_done:
            with mkdocs_gen_files.open(dir_path, "w") as f:
                f.write("---\n")
                f.write(f'title: "{title}"\n')  # 섹션 라벨 고정
                f.write("---\n\n")
                f.write(f"# {title}\n")
            dir_titles_done.add(dir_key)

    # 모듈 페이지 생성
    mod_doc = Path("reference", *parts).with_suffix(".md")
    page_title = parts[-1]
    with mkdocs_gen_files.open(mod_doc, "w") as f:
        f.write("---\n")
        f.write(f'title: "{page_title}"\n')  # 페이지 라벨 고정
        f.write("---\n\n")
        f.write(f"# `{ident}`\n\n::: {ident}\n")

    # Nav 계층 유지: msmu / plotting / _pdata 처럼 트리로 구성
    nav[parts] = Path(*parts).with_suffix(".md").as_posix()

# # literate-nav용 SUMMARY 생성 (계층 유지)
# with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
#     nav_file.writelines(nav.build_literate_nav())
