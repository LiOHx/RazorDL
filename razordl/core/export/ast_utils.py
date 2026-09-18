import ast
import re


def extract_class(source: str, class_name: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise ValueError(f"Class {class_name} not found in source")


def extract_function(source: str, function_name: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise ValueError(f"Function {function_name} not found in source")


def replace_ident(source: str, old: str, new: str) -> str:
    return re.sub(r"\b" + re.escape(old) + r"\b", new, source)


def extract_imports(source: str) -> str:
    """Extract all import statements from *source* as a string."""
    tree = ast.parse(source)
    lines = source.splitlines()
    result = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for i in range(node.lineno - 1, node.end_lineno):
                result.append(lines[i])
    return "\n".join(result)


def replace_class_docstring(class_source: str, new_docstring: str) -> str:
    """Replace (or insert) the docstring of the single class in *class_source*.

    Used when a preset re-exports another preset's class under a new name:
    the inherited docstring describes the *source* preset (e.g. "SFT preset:
    ...") and must not survive into the generated file.
    """
    tree = ast.parse(class_source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    lines = class_source.splitlines()
    first = cls.body[0]
    indent = " " * first.col_offset
    doc_line = f'{indent}"""{new_docstring}"""'
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        lines[first.lineno - 1 : first.end_lineno] = [doc_line]
    else:
        lines.insert(first.lineno - 1, doc_line)
    return "\n".join(lines)
