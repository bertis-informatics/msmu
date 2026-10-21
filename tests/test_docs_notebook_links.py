from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest

pytest.importorskip("mkdocs")

from mkdocs.structure.files import File, Files


on_page_content = run_path(Path(__file__).parents[1] / "docs" / "notebook_links.py")["on_page_content"]


@pytest.mark.parametrize("directory_urls", [True, False])
def test_notebook_links_follow_built_page_urls(directory_urls, tmp_path):
    notebook = File("tutorials/01.ipynb", str(tmp_path), str(tmp_path), directory_urls)
    target = File("tutorials/index.md", str(tmp_path), str(tmp_path), directory_urls)
    page = SimpleNamespace(file=notebook, url="tutorials/01/" if directory_urls else "tutorials/01.html")
    content = '<a href="index.md?x=1&amp;y=2#environment_setup">Environment</a><svg viewBox="0 0 10 10"/>'
    expected = "../" if directory_urls else "index.html"

    result = on_page_content(content, page, {}, Files([target]))

    assert result == f'<a href="{expected}?x=1&amp;y=2#environment_setup">Environment</a><svg viewBox="0 0 10 10"/>'


def test_notebook_links_preserve_other_targets(tmp_path):
    notebook = File("tutorials/01.ipynb", str(tmp_path), str(tmp_path), True)
    page = SimpleNamespace(file=notebook, url="tutorials/01/")
    content = """<a href="https://example.com/index.md">External</a><a href="#section">Section</a>
<a href="/absolute/">Absolute</a><a href="missing.ipynb">Missing</a>"""

    assert on_page_content(content, page, {}, Files([])) == content
    page.file = File("tutorials/index.md", str(tmp_path), str(tmp_path), True)
    assert on_page_content(content, page, {}, Files([])) == content
