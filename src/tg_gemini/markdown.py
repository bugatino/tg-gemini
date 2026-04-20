"""Markdown to Telegram HTML converter.

Ported from Go implementation in cc-connect/core/markdown_html.go.
Supports tags: <b>, <i>, <s>, <code>, <pre>, <a href="">, <blockquote>.
"""

import re
from re import Match

# Regex patterns for inline Markdown
_RE_INLINE_CODE = re.compile(r"`([^`]+)`")
_RE_BOLD_ITALIC = re.compile(r"\*\*\*(.+?)\*\*\*")
_RE_BOLD_AST = re.compile(r"\*\*(.+?)\*\*")
_RE_BOLD_UND = re.compile(r"__(.+?)__")
_RE_ITALIC_AST = re.compile(r"(?:^|[^*])\*([^*]+?)\*(?:[^*]|$)")
_RE_STRIKE = re.compile(r"~~(.+?)~~")
_RE_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_RE_WIKILINK = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]|\[\[([^\]]+)\]\]")

# Regex patterns for block-level Markdown
_RE_HEADING = re.compile(r"^#{1,6}\s*")
_RE_HORIZONTAL = re.compile(r"^(---+|\*\*\*+)$")
_RE_UNORDERED_LIST = re.compile(r"^(\s*)[-*]\s+(.*)$")
_RE_ORDERED_LIST = re.compile(r"^(\s*)\d+\.\s+(.*)$")
_RE_TABLE_SEP = re.compile(r"^\|[\s:|-]+\|$")
_RE_CALLOUT = re.compile(r"^\[!([^\]]+)\]\s*(.*)$")


def _escape_html(s: str) -> str:
    """Escape HTML special characters."""
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s.replace('"', "&quot;")


def _convert_inline_html(s: str) -> str:
    """Convert inline Markdown formatting to Telegram-compatible HTML.

    Uses placeholders to protect already-converted HTML from subsequent regex passes.
    """
    placeholders: list[tuple[str, str]] = []
    ph_idx = 0

    def next_ph(html: str) -> str:
        nonlocal ph_idx
        key = f"\x00PH{ph_idx}\x00"
        placeholders.append((key, html))
        ph_idx += 1
        return key

    # 1. Extract inline code → placeholder (content escaped)
    def replace_inline_code(m: Match[str]) -> str:
        inner = m.group(1)
        return next_ph(f"<code>{_escape_html(inner)}</code>")

    s = _RE_INLINE_CODE.sub(replace_inline_code, s)

    # 2. Extract links → placeholder (text & URL escaped)
    def replace_link(m: Match[str]) -> str:
        text = m.group(1)
        url = m.group(2)
        return next_ph(f'<a href="{_escape_html(url)}">{_escape_html(text)}</a>')

    s = _RE_LINK.sub(replace_link, s)

    # 3. Wikilinks: [[Link|Text]] → Text, [[Link]] → Link (plain text, no escape yet)
    def replace_wikilink(m: Match[str]) -> str:
        # m.group(1) is the link part, m.group(2) is the text part for [[Link|Text]]
        # m.group(3) is the link for [[Link]]
        if m.group(1) is not None and m.group(2) is not None:
            return m.group(2)
        if m.group(3) is not None:
            return m.group(3)
        return m.group(0)  # pragma: no cover

    s = _RE_WIKILINK.sub(replace_wikilink, s)

    # 4. HTML-escape the entire remaining text
    s = _escape_html(s)

    # 5. Bold-italic (***) → placeholder (must be before bold)
    def replace_bold_italic(m: Match[str]) -> str:
        inner = m.group(1)
        return next_ph(f"<b><i>{inner}</i></b>")

    s = _RE_BOLD_ITALIC.sub(replace_bold_italic, s)

    # 6. Bold → placeholder (so italic regex can't cross bold boundaries)
    def replace_bold_ast(m: Match[str]) -> str:
        inner = m.group(1)
        return next_ph(f"<b>{inner}</b>")

    s = _RE_BOLD_AST.sub(replace_bold_ast, s)

    def replace_bold_und(m: Match[str]) -> str:
        inner = m.group(1)
        return next_ph(f"<b>{inner}</b>")

    s = _RE_BOLD_UND.sub(replace_bold_und, s)

    # 7. Strikethrough → placeholder
    def replace_strike(m: Match[str]) -> str:
        inner = m.group(1)
        return next_ph(f"<s>{inner}</s>")

    s = _RE_STRIKE.sub(replace_strike, s)

    # 8. Italic (applied last, on text with bold/strike already protected)
    def replace_italic(m: Match[str]) -> str:
        # Find the first and last asterisk in the match
        full_match = m.group(0)
        first_star = full_match.find("*")
        last_star = full_match.rfind("*")
        if first_star < 0 or last_star <= first_star:
            return full_match  # pragma: no cover
        # The content is between first_star+1 and last_star
        prefix = full_match[:first_star]
        content = full_match[first_star + 1 : last_star]
        suffix = full_match[last_star + 1 :]
        return f"{prefix}<i>{content}</i>{suffix}"

    s = _RE_ITALIC_AST.sub(replace_italic, s)

    # 9. Restore all placeholders (may be nested, so iterate until stable)
    max_iterations = len(placeholders) + 1
    for _ in range(max_iterations):  # pragma: no branch
        changed = False
        for key, html in placeholders:
            if key in s:
                s = s.replace(key, html, 1)
                changed = True
        if not changed:
            break

    return s


def markdown_to_html(md: str) -> str:
    """Convert Markdown to Telegram-compatible HTML.

    Supported tags: <b>, <i>, <s>, <code>, <pre>, <a href="">, <blockquote>.

    Args:
        md: Markdown text to convert.

    Returns:
        HTML string suitable for Telegram.
    """
    # Normalize HTML line breaks from LLM output to actual newlines
    md = re.sub(r"<br\s*/?>", "\n", md, flags=re.IGNORECASE)

    lines = md.split("\n")
    result_parts: list[str] = []

    in_code_block = False
    code_lang = ""
    code_lines: list[str] = []

    in_blockquote = False
    bq_lines: list[str] = []

    in_table = False
    tbl_lines: list[str] = []

    def flush_blockquote() -> None:
        nonlocal in_blockquote
        if not bq_lines:
            return  # pragma: no cover

        result_parts.append("<blockquote>")
        start_idx = 0

        # Check for callout syntax in the first line
        if bq_lines:  # pragma: no branch
            m = _RE_CALLOUT.match(bq_lines[0])
            if m:
                callout_type = m.group(1)
                callout_title = m.group(2)
                if callout_title:
                    result_parts.append(
                        f"<b>{_escape_html(callout_type)}: {_escape_html(callout_title)}</b>"
                    )
                else:
                    result_parts.append(f"<b>{_escape_html(callout_type)}</b>")
                start_idx = 1
                if start_idx < len(bq_lines):
                    result_parts.append("\n")

        for j in range(start_idx, len(bq_lines)):
            if j > start_idx:
                result_parts.append("\n")
            result_parts.append(_convert_inline_html(bq_lines[j]))

        result_parts.append("</blockquote>")
        bq_lines.clear()
        in_blockquote = False

    def flush_table() -> None:
        nonlocal in_table
        if not tbl_lines:
            return  # pragma: no cover

        for j, tl_raw in enumerate(tbl_lines):
            if j > 0:
                result_parts.append("\n")
            tl = tl_raw.strip()
            if _RE_TABLE_SEP.match(tl):
                result_parts.append("——————————")
            else:
                # Extract content between outer | chars
                inner = tl[1:-1]
                cells = [cell.strip() for cell in inner.split("|")]
                row = " | ".join(cells)
                result_parts.append(_convert_inline_html(row))

        tbl_lines.clear()
        in_table = False

    for i, line in enumerate(lines):
        trimmed = line.strip()

        # Handle code blocks
        if trimmed.startswith("```"):
            if not in_code_block:
                # Flush any pending blockquotes or tables
                if in_blockquote:
                    flush_blockquote()
                    result_parts.append("\n")
                if in_table:
                    flush_table()
                    result_parts.append("\n")
                in_code_block = True
                code_lang = trimmed[3:]  # Remove "```" prefix
                code_lines = []
            else:
                in_code_block = False
                if code_lang:
                    result_parts.append(
                        f'<pre><code class="language-{_escape_html(code_lang)}">'
                    )
                else:
                    result_parts.append("<pre><code>")
                result_parts.append(_escape_html("\n".join(code_lines)))
                result_parts.append("</code></pre>")
                if i < len(lines) - 1:
                    result_parts.append("\n")
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        # Determine line type for blockquote/table buffering
        is_quote = trimmed.startswith("> ") or trimmed == ">"
        is_table = len(trimmed) > 2 and trimmed[0] == "|" and trimmed[-1] == "|"

        # Flush blockquote when leaving
        if not is_quote and in_blockquote:
            flush_blockquote()
            result_parts.append("\n")

        # Flush table when leaving
        if not is_table and in_table:
            flush_table()
            result_parts.append("\n")

        # Buffer blockquote lines into a single block
        if is_quote:
            quote_content = trimmed[2:] if trimmed.startswith("> ") else ""
            bq_lines.append(quote_content)
            in_blockquote = True
            continue

        # Buffer table lines
        if is_table:
            tbl_lines.append(trimmed)
            in_table = True
            continue

        # Headings → bold
        heading_match = _RE_HEADING.match(line)
        if heading_match:
            heading = heading_match.group(0)
            rest = line[len(heading) :]
            result_parts.append("<b>")
            result_parts.append(_convert_inline_html(rest))
            result_parts.append("</b>")
        elif _RE_HORIZONTAL.match(trimmed):
            result_parts.append("——————————")
        else:
            # Try unordered list
            m = _RE_UNORDERED_LIST.match(line)
            if m:
                indent = "  " * (len(m.group(1)) // 2)
                result_parts.append(indent + "• " + _convert_inline_html(m.group(2)))
            else:
                # Try ordered list
                m = _RE_ORDERED_LIST.match(line)
                if m:
                    indent = "  " * (len(m.group(1)) // 2)
                    num_dot = line[: len(line) - len(m.group(2))].strip()
                    result_parts.append(
                        indent
                        + _escape_html(num_dot)
                        + " "
                        + _convert_inline_html(m.group(2))
                    )
                else:
                    result_parts.append(_convert_inline_html(line))

        if i < len(lines) - 1:
            result_parts.append("\n")

    # Flush any remaining buffered state
    if in_blockquote:
        flush_blockquote()
    if in_table:
        flush_table()
    if in_code_block and code_lines:
        result_parts.append("<pre><code>")
        result_parts.append(_escape_html("\n".join(code_lines)))
        result_parts.append("</code></pre>")

    return "".join(result_parts)


def split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split text into chunks respecting code fence boundaries.

    When a chunk boundary falls inside a code block, the fence is closed at the end
    of the chunk and reopened at the start of the next chunk.

    Args:
        text: The text to split.
        max_len: Maximum length of each chunk (default 4096 for Telegram).

    Returns:
        List of text chunks.
    """
    if len(text) <= max_len:
        return [text]

    closing_fence = "\n```"

    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    open_fence = ""  # the ``` opening line, or "" if outside code block

    for line in lines:
        line_len = len(line) + 1  # +1 for newline

        # Reserve space for the closing fence when inside a code block
        limit = max_len
        if open_fence:
            limit -= len(closing_fence)

        # Check if this single line exceeds the limit
        if line_len > limit:
            # Flush current chunk first if it has content
            if current:
                chunk = "\n".join(current)
                if open_fence:
                    chunk += closing_fence
                chunks.append(chunk)
                current = []
                current_len = 0
                if open_fence:
                    current.append(open_fence)
                    current_len = len(open_fence) + 1
            # Split the long line into smaller pieces
            remaining_line = line
            while remaining_line:
                remaining = limit - current_len
                if remaining <= 0:
                    # Flush and start new chunk
                    chunk = "\n".join(current)
                    if open_fence:
                        chunk += closing_fence
                    chunks.append(chunk)
                    current = []
                    current_len = 0
                    if open_fence:
                        current.append(open_fence)
                        current_len = len(open_fence) + 1
                    remaining = limit - current_len
                take = min(len(remaining_line), remaining - 1)  # -1 for newline
                if take <= 0:
                    take = min(len(remaining_line), limit - 1)
                current.append(remaining_line[:take])
                current_len += take + 1
                remaining_line = remaining_line[take:]
            continue

        if current_len + line_len > limit and current:
            chunk = "\n".join(current)
            if open_fence:
                chunk += closing_fence
            chunks.append(chunk)

            current = []
            current_len = 0
            if open_fence:
                current.append(open_fence)
                current_len = len(open_fence) + 1

        current.append(line)
        current_len += line_len

        trimmed = line.strip()
        if trimmed.startswith("```"):
            open_fence = "" if open_fence else trimmed

    if current:  # pragma: no branch
        chunk = "\n".join(current)
        if open_fence:
            chunk += "\n```"
        chunks.append(chunk)

    return chunks
