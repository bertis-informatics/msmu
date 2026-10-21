"""Resolve source-relative notebook links after mkdocs-jupyter renders HTML."""

import html
import posixpath
import re
from urllib.parse import unquote, urlsplit, urlunsplit

from mkdocs.utils import get_relative_url


def on_page_content(content, page, config, files):
    if not page.file.src_uri.endswith(".ipynb"):
        return content

    def resolve_link(match):
        prefix, quote, href = match.groups()
        parts = urlsplit(html.unescape(href))
        if parts.scheme or parts.netloc or not parts.path or parts.path.startswith("/"):
            return match.group(0)
        source = posixpath.normpath(posixpath.join(posixpath.dirname(page.file.src_uri), unquote(parts.path)))
        target = files.get_file_from_path(source)
        if target is None or target.inclusion.is_excluded():
            return match.group(0)
        href = urlunsplit(("", "", get_relative_url(target.url, page.url), parts.query, parts.fragment))
        return prefix + quote + html.escape(href, quote=True) + quote

    # Patch link attributes only, preserving saved plots and the rest of the HTML.
    return re.sub(r"(<a\b[^>]*\bhref=)([\"'])(.*?)\2", resolve_link, content)
