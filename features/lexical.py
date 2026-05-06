"""
Language-agnostic lexical features extracted via regex and string analysis.
Works on any language — no AST required.
"""
import re
from typing import Any

# Patterns for comment detection per language
_COMMENT_PREFIXES = {
    "python":     re.compile(r"^\s*#"),
    "javascript": re.compile(r"^\s*//"),
    "java":       re.compile(r"^\s*//"),
    "c++":        re.compile(r"^\s*//"),
    "go":         re.compile(r"^\s*//"),
    "php":        re.compile(r"^\s*(//|#)"),
    "c#":         re.compile(r"^\s*//"),
    "c":          re.compile(r"^\s*//"),
}
_COMMENT_DEFAULT = re.compile(r"^\s*(//|#|--)")

_INLINE_COMMENT = re.compile(r".+\s+(#|//)(?!\S*[\"'])")
_TODO            = re.compile(r"#\s*(TODO|FIXME|HACK|XXX|NOTE)\b", re.IGNORECASE)
_SNAKE_CASE      = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_CAMEL_CASE      = re.compile(r"\b[a-z][a-z0-9]*(?:[A-Z][a-z0-9]+)+\b")
_LONG_IDENT      = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{15,}\b")
_STEP_COMMENT    = re.compile(r"#\s*(Step\s+\d|[Ss]tep\s+\d|\d+\.\s+[A-Z])", re.IGNORECASE)
_TYPE_ANNOTATION = re.compile(r":\s*(int|str|float|bool|list|dict|tuple|set|Optional|Union|Any|None)\b")
_TRY_BLOCK       = re.compile(r"\btry\s*[:{]")
_EXCEPT_BLOCK    = re.compile(r"\b(except|catch)\b")
_RAISE_THROW     = re.compile(r"\b(raise|throw)\b")

# Defensive coding patterns (language-agnostic)
_NULL_CHECK      = re.compile(r"\b(is\s+None|is\s+not\s+None|!= None|== None|== null|!= null|=== null|!== null)\b")
_NONE_RETURN     = re.compile(r"\breturn\s+None\b")
_ISINSTANCE      = re.compile(r"\bisinstance\s*\(")
_LOGGING         = re.compile(r"\b(logging\.|logger\.|log\.)(debug|info|warning|error|critical)\s*\(", re.IGNORECASE)
_PRINT_GUARD     = re.compile(r"^\s*print\s*\(", re.MULTILINE)
_EARLY_RETURN    = re.compile(r"^\s*(return|throw|raise)\b(?!.*\()", re.MULTILINE)
_GUARD_CLAUSE    = re.compile(r"if\s+.*(is\s+(None|not\s+None)|not\s+\w+|len\s*\(|== None|!= None).*:\s*$", re.MULTILINE)
_UNIFORM_EXCEPT  = re.compile(r"\bexcept\s+Exception\b")
_BARE_EXCEPT     = re.compile(r"\bexcept\s*:")
_DEFAULT_PARAM   = re.compile(r"=\s*(None|0|False|True|\[\]|\{\}|\"\")\s*[,)]")


def extract(code: str, language: str = "python") -> dict[str, Any]:
    lines = code.splitlines()
    total_lines = len(lines)
    total_chars = len(code)

    if total_lines == 0 or total_chars == 0:
        return _empty()

    non_blank = [l for l in lines if l.strip()]
    blank_count = total_lines - len(non_blank)
    line_lengths = [len(l) for l in lines]

    comment_re = _COMMENT_PREFIXES.get(language.lower(), _COMMENT_DEFAULT)
    comment_lines = sum(1 for l in lines if comment_re.match(l))
    inline_comments = sum(1 for l in non_blank if _INLINE_COMMENT.search(l))
    todo_count = len(_TODO.findall(code))
    step_comments = len(_STEP_COMMENT.findall(code))

    alpha = sum(c.isalpha() for c in code)
    digits = sum(c.isdigit() for c in code)
    whitespace = sum(c.isspace() for c in code)
    upper = sum(c.isupper() for c in code)

    snake = len(_SNAKE_CASE.findall(code))
    camel = len(_CAMEL_CASE.findall(code))
    long_idents = len(_LONG_IDENT.findall(code))
    type_annotations = len(_TYPE_ANNOTATION.findall(code))

    try_count = len(_TRY_BLOCK.findall(code))
    except_count = len(_EXCEPT_BLOCK.findall(code))
    raise_count = len(_RAISE_THROW.findall(code))

    # Defensive coding
    null_checks      = len(_NULL_CHECK.findall(code))
    none_returns     = len(_NONE_RETURN.findall(code))
    isinstance_count = len(_ISINSTANCE.findall(code))
    logging_count    = len(_LOGGING.findall(code))
    print_guards     = len(_PRINT_GUARD.findall(code))
    early_returns    = len(_EARLY_RETURN.findall(code))
    guard_clauses    = len(_GUARD_CLAUSE.findall(code))
    uniform_except   = len(_UNIFORM_EXCEPT.findall(code))
    bare_except      = len(_BARE_EXCEPT.findall(code))
    default_params   = len(_DEFAULT_PARAM.findall(code))

    # Indentation style: tabs vs spaces
    indented = [l for l in lines if l and l[0] in (" ", "\t")]
    tab_indent = sum(1 for l in indented if l[0] == "\t")
    space_indent = len(indented) - tab_indent
    indent_consistency = (
        max(tab_indent, space_indent) / len(indented) if indented else 1.0
    )

    # Trailing whitespace (sloppy human habit; AI rarely leaves these)
    trailing_ws = sum(1 for l in lines if l != l.rstrip())

    return {
        # Volume
        "char_count":            total_chars,
        "line_count":            total_lines,
        "non_blank_line_count":  len(non_blank),

        # Line length
        "avg_line_length":       sum(line_lengths) / total_lines,
        "max_line_length":       max(line_lengths),
        "p90_line_length":       sorted(line_lengths)[int(total_lines * 0.9)],

        # Ratios
        "blank_line_ratio":      blank_count / total_lines,
        "comment_line_ratio":    comment_lines / total_lines,
        "inline_comment_ratio":  inline_comments / len(non_blank) if non_blank else 0,
        "whitespace_ratio":      whitespace / total_chars,
        "digit_ratio":           digits / total_chars,
        "uppercase_ratio":       upper / alpha if alpha else 0,

        # Comments
        "todo_count":            todo_count,
        "step_comment_count":    step_comments,

        # Identifiers
        "snake_case_count":      snake,
        "camel_case_count":      camel,
        "long_ident_count":      long_idents,
        "naming_style_ratio":    snake / (snake + camel + 1),  # 1=pure snake, 0=pure camel

        # Type hints / annotations
        "type_annotation_count": type_annotations,

        # Error handling
        "try_count":             try_count,
        "except_count":          except_count,
        "raise_count":           raise_count,
        "error_handling_density": (try_count + except_count) / total_lines,

        # Defensive coding
        "null_check_count":       null_checks,
        "none_return_count":      none_returns,
        "isinstance_count":       isinstance_count,
        "logging_count":          logging_count,
        "print_guard_count":      print_guards,
        "early_return_count":     early_returns,
        "guard_clause_count":     guard_clauses,
        "uniform_except_count":   uniform_except,
        "bare_except_count":      bare_except,
        "default_param_count":    default_params,
        "defensive_density":      (null_checks + guard_clauses + isinstance_count) / total_lines,
        "error_uniformity":       uniform_except / (except_count + 1),

        # Style consistency
        "indent_consistency":    indent_consistency,
        "trailing_ws_ratio":     trailing_ws / total_lines,
    }


def _empty() -> dict[str, Any]:
    return {k: 0 for k in extract("x = 1")}
