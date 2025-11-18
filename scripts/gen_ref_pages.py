# scripts/gen_ref_pages.py
from pathlib import Path
import mkdocs_gen_files

PACKAGE = "msmu"          # ./msmu 레이아웃 가정
src_dir = Path(PACKAGE)
nav = mkdocs_gen_files.Nav()

def clean_label(s: str) -> str:
    out = []
    for ch in s:
        cp = ord(ch)
        if (0x0000 <= cp <= 0x001F or 0x007F <= cp <= 0x009F or  # control
            cp in (0x200D, 0xFE0E, 0xFE0F) or                    # ZWJ/VS
            0xE000 <= cp <= 0xF8FF or                            # PUA
            0x1F300 <= cp <= 0x1FAFF or                          # emoji block
            cp == 0xFFFD):                                       # replacement char
            continue
        out.append(ch)
    return " ".join("".join(out).split())

# 폴더(패키지 디렉토리) → index.md 생성하여 섹션 라벨 고정
dir_titles_done = set()

for py in sorted(src_dir.rglob("*.py")):
    if py.name == "__init__.py":
        continue

    rel = py.relative_to(src_dir).with_suffix("")  # e.g. plotting/_pdata
    parts = list(rel.parts)                        # ['plotting', '_pdata']
    ident = ".".join([PACKAGE] + parts)            # msmu.plotting._pdata

    # 상위 디렉토리들에 대해 index.md 생성 (섹션 제목 고정)
    for d in range(1, len(parts)+1):
        dir_parts = parts[:d-1]  # 상위 폴더 경로
        if d == 1 and not dir_parts:
            # 최상위 'msmu' 섹션용 인덱스
            dir_key = ()
            dir_path = Path("reference", PACKAGE, "index.md")
            title = clean_label(PACKAGE)
        elif d > 1:
            dir_key = tuple(parts[:d-1])
            dir_path = Path("reference", *dir_key, "index.md")
            title = clean_label(".".join([PACKAGE] + list(dir_key)))
        else:
            continue

        if dir_key not in dir_titles_done:
            with mkdocs_gen_files.open(dir_path, "w") as f:
                f.write("---\n")
                f.write(f'title: "{title}"\n')     # 섹션 라벨 고정
                f.write("---\n\n")
                f.write(f"# {title}\n")
            dir_titles_done.add(dir_key)

    # 모듈 페이지 생성
    mod_doc = Path("reference", *parts).with_suffix(".md")
    page_title = clean_label(".".join([PACKAGE] + parts))
    with mkdocs_gen_files.open(mod_doc, "w") as f:
        f.write("---\n")
        f.write(f'title: "{page_title}"\n')        # 페이지 라벨 고정
        f.write("---\n\n")
        f.write(f"# `{ident}`\n\n::: {ident}\n")

    # Nav 계층 유지: msmu / plotting / _pdata 처럼 트리로 구성
    nav[[PACKAGE] + parts] = Path(*parts).with_suffix(".md").as_posix()

# literate-nav용 SUMMARY 생성 (계층 유지)
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())