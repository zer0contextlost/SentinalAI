"""
Python AST-level features. Only called when language == 'python' and the
code parses cleanly. Returns an empty dict on parse failure rather than
raising — callers should handle gracefully.
"""
import ast
from typing import Any


def extract(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}

    visitor = _Visitor()
    visitor.visit(tree)

    func_lengths = visitor.function_lengths
    return {
        "ast_function_count":       len(func_lengths),
        "ast_class_count":          visitor.class_count,
        "ast_import_count":         visitor.import_count,
        "ast_return_count":         visitor.return_count,
        "ast_assert_count":         visitor.assert_count,
        "ast_raise_count":          visitor.raise_count,
        "ast_try_count":            visitor.try_count,
        "ast_comprehension_count":  visitor.comprehension_count,
        "ast_lambda_count":         visitor.lambda_count,
        "ast_decorator_count":      visitor.decorator_count,
        "ast_docstring_count":      visitor.docstring_count,
        "ast_has_main_guard":       int(visitor.has_main_guard),
        "ast_max_nesting_depth":    visitor.max_depth,
        "ast_avg_function_length":  (
            sum(func_lengths) / len(func_lengths) if func_lengths else 0
        ),
        "ast_max_function_length":  max(func_lengths) if func_lengths else 0,
        "ast_type_annotation_count": visitor.type_annotation_count,
        "ast_fstring_count":        visitor.fstring_count,
        "ast_walrus_count":             visitor.walrus_count,
        "ast_global_count":             visitor.global_count,

        # Defensive coding (AST-precise)
        "ast_none_compare_count":       visitor.none_compare_count,
        "ast_isinstance_count":         visitor.isinstance_count,
        "ast_early_return_count":       visitor.early_return_count,
        "ast_broad_except_count":       visitor.broad_except_count,
        "ast_finally_count":            visitor.finally_count,
        "ast_default_arg_count":        visitor.default_arg_count,
        "ast_guarded_fn_ratio":         (
            visitor.guarded_functions / len(func_lengths) if func_lengths else 0
        ),
        "ast_except_per_try":           (
            visitor.except_handler_count / (visitor.try_count + 1)
        ),
        "ast_defensive_line_ratio":     visitor.defensive_lines / max(1, sum(func_lengths)),
    }


class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.class_count         = 0
        self.import_count        = 0
        self.return_count        = 0
        self.assert_count        = 0
        self.raise_count         = 0
        self.try_count           = 0
        self.comprehension_count = 0
        self.lambda_count        = 0
        self.decorator_count     = 0
        self.docstring_count     = 0
        self.type_annotation_count = 0
        self.fstring_count          = 0
        self.walrus_count           = 0
        self.global_count           = 0
        self.has_main_guard         = False
        self.function_lengths       = []
        self.max_depth              = 0
        self._depth                 = 0

        # Defensive coding
        self.none_compare_count     = 0
        self.isinstance_count       = 0
        self.early_return_count     = 0
        self.broad_except_count     = 0
        self.except_handler_count   = 0
        self.finally_count          = 0
        self.default_arg_count      = 0
        self.guarded_functions      = 0
        self.defensive_lines        = 0
        self._current_fn_has_guard  = False

    def _enter(self):
        self._depth += 1
        self.max_depth = max(self.max_depth, self._depth)

    def _exit(self):
        self._depth -= 1

    def _is_docstring(self, node) -> bool:
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )

    def visit_ClassDef(self, node):
        self.class_count += 1
        if node.body and self._is_docstring(node.body[0]):
            self.docstring_count += 1
        self._enter()
        self.generic_visit(node)
        self._exit()

    def visit_Import(self, node):      self.import_count += 1;  self.generic_visit(node)
    def visit_ImportFrom(self, node):  self.import_count += 1;  self.generic_visit(node)
    def visit_Assert(self, node):      self.assert_count += 1;  self.generic_visit(node)
    def visit_Raise(self, node):       self.raise_count += 1;   self.generic_visit(node)
    def visit_Lambda(self, node):      self.lambda_count += 1;  self.generic_visit(node)
    def visit_Global(self, node):      self.global_count += 1;  self.generic_visit(node)
    def visit_NamedExpr(self, node):   self.walrus_count += 1;  self.generic_visit(node)

    def visit_JoinedStr(self, node):
        self.fstring_count += 1
        self.generic_visit(node)

    def visit_ListComp(self, node):    self.comprehension_count += 1; self.generic_visit(node)
    def visit_SetComp(self, node):     self.comprehension_count += 1; self.generic_visit(node)
    def visit_DictComp(self, node):    self.comprehension_count += 1; self.generic_visit(node)
    def visit_GeneratorExp(self, node):self.comprehension_count += 1; self.generic_visit(node)

    def visit_If(self, node):
        # Detect `if __name__ == "__main__":`
        if (
            isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            self.has_main_guard = True
        self._enter()
        self.generic_visit(node)
        self._exit()

    def visit_For(self, node):   self._enter(); self.generic_visit(node); self._exit()
    def visit_While(self, node): self._enter(); self.generic_visit(node); self._exit()
    def visit_With(self, node):  self._enter(); self.generic_visit(node); self._exit()

    def visit_FunctionDef(self, node):
        # Track whether function opens with a guard clause (None/type check + return)
        prev = self._current_fn_has_guard
        self._current_fn_has_guard = False

        self.decorator_count += len(node.decorator_list)
        if node.returns or any(a.annotation for a in node.args.args):
            self.type_annotation_count += 1
        if node.body and self._is_docstring(node.body[0]):
            self.docstring_count += 1

        # Default argument count (None/[] defaults signal defensive style)
        for d in node.args.defaults + node.args.kw_defaults:
            if d is not None:
                self.default_arg_count += 1

        end = getattr(node, "end_lineno", node.lineno)
        self.function_lengths.append(end - node.lineno + 1)
        self._enter()
        self.generic_visit(node)
        self._exit()

        if self._current_fn_has_guard:
            self.guarded_functions += 1
        self._current_fn_has_guard = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ExceptHandler(self, node):
        self.except_handler_count += 1
        # `except Exception` or bare `except:` — broad catches
        if node.type is None:
            self.broad_except_count += 1
        elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
            self.broad_except_count += 1
        self.defensive_lines += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.try_count += 1
        if node.finalbody:
            self.finally_count += 1
        self._enter()
        self.generic_visit(node)
        self._exit()

    def visit_Compare(self, node):
        # Detect `x is None` / `x is not None`
        for op in node.ops:
            if isinstance(op, (ast.Is, ast.IsNot)):
                for comp in node.comparators:
                    if isinstance(comp, ast.Constant) and comp.value is None:
                        self.none_compare_count += 1
                        self.defensive_lines += 1
                        self._current_fn_has_guard = True
        self.generic_visit(node)

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
        ):
            self.isinstance_count += 1
            self.defensive_lines += 1
            self._current_fn_has_guard = True
        self.generic_visit(node)

    def visit_Return(self, node):
        self.return_count += 1
        # Early return = return inside an if at depth > function base
        if self._depth >= 2 and (node.value is None or (
            isinstance(node.value, ast.Constant) and node.value.value is None
        )):
            self.early_return_count += 1
        self.generic_visit(node)
