"""
CodeInsight AI - Code Validator

Validates Python code using AST parsing.
Never executes arbitrary user-submitted code.
"""

import ast
from typing import Tuple


def validate_python_code(code: str, max_length: int = 10000) -> Tuple[bool, str]:
    """
    Validate that the given string is syntactically valid Python.
    
    Returns:
        (is_valid, error_message) - error_message is empty if valid.
    """
    if not code or not code.strip():
        return False, "Code is empty or contains only whitespace."
    
    if len(code) > max_length:
        return False, f"Code exceeds maximum length of {max_length} characters."
    
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        line_info = f" at line {e.lineno}" if e.lineno else ""
        return False, f"Syntax error{line_info}: {e.msg}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def extract_functions(code: str) -> list:
    """Extract function names from valid Python code."""
    try:
        tree = ast.parse(code)
        return [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
    except SyntaxError:
        return []


def count_lines(code: str) -> int:
    """Count non-empty lines in code."""
    return len([line for line in code.splitlines() if line.strip()])
