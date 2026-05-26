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


def extract_standalone_classes(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    lines = source.splitlines()
    result = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            result[node.name] = "\n".join(lines[node.lineno - 1 : node.end_lineno])
    return result


def replace_ident(source: str, old: str, new: str) -> str:
    return re.sub(r"\b" + re.escape(old) + r"\b", new, source)
