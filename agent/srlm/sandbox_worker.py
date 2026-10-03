"""Sandboxed REPL worker process (one per candidate program).

Protocol (newline-delimited JSON over stdin/stdout):
  parent → {"type": "exec", "code": "..."}
  worker → {"type": "result", "stdout": str, "error": str|null, "final": any, "has_final": bool, "violation": str|null}
  worker → {"type": "subcall", "query": str, "text": str}   (only when sub_call is enabled)
  parent → {"type": "subcall_result", "output": str}

Safety: resource limits (address space, CPU seconds, file size, no core
dumps), cwd in a private temp dir, AST validation (only allow-listed imports,
no dunder or frame-introspection attributes, no dunder strings), restricted
builtins (no open/exec/eval/compile/input), socket/subprocess modules blocked.
"""
from __future__ import annotations

import ast
import builtins
import io
import json
import sys
import traceback
from contextlib import redirect_stdout

BANNED_ATTRS = {"gi_frame", "gi_code", "f_globals", "f_locals", "f_back", "f_builtins", "f_code", "tb_frame",
                "tb_next", "cr_frame", "ag_frame", "func_globals", "co_code", "mro", "func_code"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "input", "breakpoint", "globals", "locals", "vars",
                "__import__", "memoryview", "help", "exit", "quit", "getattr", "setattr", "delattr", "type",
                "object", "classmethod", "staticmethod", "super", "property"}
SAFE_BUILTINS = ["abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "format",
                 "frozenset", "hash", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max",
                 "min", "next", "print", "range", "repr", "reversed", "round", "set", "sorted", "str", "sum",
                 "tuple", "zip", "chr", "ord", "hex", "bin", "oct", "pow", "callable", "hasattr", "id",
                 "Exception", "ValueError", "KeyError", "IndexError", "TypeError", "ZeroDivisionError",
                 "StopIteration", "AttributeError", "RuntimeError", "ArithmeticError", "LookupError",
                 "True", "False", "None", "NotImplemented", "Ellipsis"]


class Violation(Exception):
    pass


def validate(code: str, allowed: set[str]) -> ast.AST:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for n in names:
                if n.split(".")[0] not in allowed:
                    raise Violation(f"import of '{n}' is not allowed (allowed: {sorted(allowed)})")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in BANNED_ATTRS:
                raise Violation(f"access to attribute '{node.attr}' is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES or node.id.startswith("__"):
                raise Violation(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise Violation("string constants containing '__' are not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise Violation("global/nonlocal statements are not allowed")
    return tree


def apply_limits(mem_mb: int, cpu_s: int, fsize_mb: int = 1) -> None:
    """POSIX resource limits. On Windows (no `resource` module) only the parent's wall-clock timeout applies."""
    try:
        import resource
    except ImportError:
        return
    resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_mb * 1024 * 1024, fsize_mb * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main() -> None:
    vars_path, cfg_path = sys.argv[1], sys.argv[2]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    with open(vars_path, encoding="utf-8") as f:
        variables = json.load(f)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    from agent.srlm.helpers import Helpers   # import before lockdown

    proto_out = sys.stdout
    proto_in = sys.stdin
    allowed = set(cfg.get("allowed_imports", []))
    max_out = int(cfg.get("max_output_chars", 4000))
    # Pre-import allowed modules so the restricted __import__ can hand them out.
    mods = {}
    for name in allowed:
        try:
            mods[name] = __import__(name)
        except ImportError:
            pass
    for blocked in ("socket", "subprocess", "ctypes", "multiprocessing", "urllib", "http", "requests", "shutil"):
        sys.modules[blocked] = None  # type: ignore[assignment]
    apply_limits(int(cfg.get("memory_limit_mb", 512)), int(cfg.get("cpu_limit_s", 600)))

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        top = name.split(".")[0]
        if top not in allowed or top not in mods:
            raise ImportError(f"import of '{name}' is not allowed")
        return __import__(name, globals, locals, fromlist, level)

    safe_builtins = {n: getattr(builtins, n) for n in SAFE_BUILTINS if hasattr(builtins, n)}
    safe_builtins["__import__"] = safe_import
    ns: dict = {"__builtins__": safe_builtins, "__name__": "__repl__"}
    ns.update(variables)
    helpers = Helpers(ns)
    ns.update(helpers.namespace())
    state = {"final": None, "has_final": False}

    def FINAL(value):  # noqa: N802 - name is part of the program protocol
        json.dumps(value)  # must be JSON-serialisable
        state["final"], state["has_final"] = value, True

    ns["FINAL"] = FINAL

    if cfg.get("use_subcalls"):
        def sub_call(query: str, text: str) -> str:
            proto_out.write(json.dumps({"type": "subcall", "query": str(query), "text": str(text)[:20000]}) + "\n")
            proto_out.flush()
            resp = json.loads(proto_in.readline())
            return resp.get("output", "")
        ns["sub_call"] = sub_call

    for line in proto_in:
        msg = json.loads(line)
        if msg.get("type") != "exec":
            continue
        buf = io.StringIO()
        err = violation = None
        state["final"], state["has_final"] = None, False   # FINAL applies to this step only
        try:
            tree = validate(msg["code"], allowed)
            with redirect_stdout(buf):
                exec(compile(tree, "<step>", "exec"), ns)  # noqa: S102 - sandboxed namespace
        except Violation as v:
            violation = str(v)
            err = f"SandboxViolation: {v}"
        except MemoryError:
            violation = "memory limit exceeded"
            err = "MemoryError: memory limit exceeded"
        except BaseException as e:  # noqa: BLE001 - report every program error to the model
            tb = traceback.format_exception_only(type(e), e)
            err = "".join(tb).strip()[-1500:]
        out = buf.getvalue()
        if len(out) > max_out:
            out = out[:max_out] + f"\n...[truncated {len(out) - max_out} chars]"
        try:
            final = state["final"]
            json.dumps(final)
        except (TypeError, ValueError):
            final, state["has_final"] = None, False
        proto_out.write(json.dumps({"type": "result", "stdout": out, "error": err, "final": final,
                                    "has_final": state["has_final"], "violation": violation}) + "\n")
        proto_out.flush()


if __name__ == "__main__":
    main()
