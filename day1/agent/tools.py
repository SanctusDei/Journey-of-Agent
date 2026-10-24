"""Tool definitions and local implementations. All file access is sandboxed to workspace."""
import json
import operator
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent / 'workspace'
MAX_READ_CHARS = 12000

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "calculator", "description": "Calculate two numbers using add/subtract/multiply/divide.", "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}, "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]}}, "required": ["a", "b", "operation"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "list_files", "description": "List .md and .txt files under the research workspace (relative paths).", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read one .md or .txt file from the workspace by relative path.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}}},
]


def calculator(a: float, b: float, operation: str):
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError('a and b must be numbers')
    operations = {"add": operator.add, "subtract": operator.sub, "multiply": operator.mul, "divide": operator.truediv}
    if operation not in operations:
        raise ValueError('unsupported operation')
    if operation == 'divide' and b == 0:
        raise ValueError('division by zero')
    return {"result": operations[operation](a, b)}


def _safe_file(path: str) -> Path:
    if not isinstance(path, str):
        raise ValueError('path must be a string')
    base = WORKSPACE.resolve()
    candidate = (base / path).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError('path outside workspace')
    if candidate.suffix.lower() not in {'.md', '.txt'}:
        raise ValueError('only .md and .txt supported')
    if not candidate.is_file():
        raise FileNotFoundError('file not found')
    return candidate


def list_files():
    base = WORKSPACE.resolve()
    files = []
    for file in base.rglob('*'):
        if file.suffix.lower() in {'.md', '.txt'} and file.is_file() and file.resolve().is_relative_to(base):
            files.append(str(file.relative_to(base)))
    return {"files": sorted(files)[:100]}


def read_file(path: str):
    file = _safe_file(path)
    with file.open('r', encoding='utf-8') as stream:
        content = stream.read(MAX_READ_CHARS + 1)
    return {"path": str(file.relative_to(WORKSPACE.resolve())), "content": content[:MAX_READ_CHARS], "truncated": len(content) > MAX_READ_CHARS}


FUNCTIONS = {'calculator': calculator, 'list_files': list_files, 'read_file': read_file}


def execute_tool(name: str, arguments: str) -> str:
    """Never evaluate model generated Python. Dispatch only registered tools."""
    try:
        func = FUNCTIONS.get(name)
        if func is None:
            raise ValueError(f'unknown tool: {name}')
        args = json.loads(arguments)
        if not isinstance(args, dict):
            raise ValueError('arguments must be a JSON object')
        allowed = {
            'calculator': {'a', 'b', 'operation'},
            'list_files': set(),
            'read_file': {'path'},
        }
        if set(args) - allowed[name]:
            raise ValueError('unexpected tool arguments')
        result = func(**args)
        return json.dumps({"ok": True, "data": result}, ensure_ascii=False)
    except (ValueError, TypeError, KeyError, FileNotFoundError, UnicodeError, OSError, OverflowError) as exc:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
