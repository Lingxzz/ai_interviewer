"""
CodeExecutor — 安全代码执行沙箱
- 子进程隔离，避免污染主进程
- 重定向 stdout/stderr，完整捕获 print 输出
- 硬超时保护，防止死循环卡死
- 简单静态检查，拦截高危操作
"""

import re
import subprocess
import sys
import textwrap


# 禁止执行的模式（静态扫描）
_BLOCKED_PATTERNS = [
    r"\bos\.system\b",
    r"\bsubprocess\b",
    r"\b__import__\b",
    r"\bopen\s*\(.*['\"]w",   # 写文件
    r"\bsocket\b",
    r"\burllib\b",
    r"\brequests\b",
    r"\bshutil\b",
]

MAX_OUTPUT = 4000   # 最大输出字符数


class CodeExecutor:
    """
    使用方式：
        executor = CodeExecutor(timeout=10)
        result = executor.run(code_str)
        # result: {"stdout": str, "stderr": str, "status": "ok"|"err"|"timeout"}
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    # ── 公开接口 ──────────────────────────────────────────────────

    def run(self, code_str: str) -> dict:
        """
        执行 Python 代码字符串，返回结构化结果。
        """
        safe, reason = self._check_safety(code_str)
        if not safe:
            return {"stdout": "", "stderr": reason, "status": "err"}

        wrapped = self._wrap(code_str)

        try:
            result = subprocess.run(
                [sys.executable, "-c", wrapped],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={"PATH": ""},   # 最小化环境变量继承
            )
            stdout, stderr = self._parse_output(result.stdout, result.stderr)
            return {
                "stdout": stdout[:MAX_OUTPUT],
                "stderr": stderr[:MAX_OUTPUT],
                "status": "err" if stderr else "ok",
            }

        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"执行超时（>{self.timeout}s），请检查是否有死循环。",
                "status": "timeout",
            }
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "status": "err"}

    # ── 内部方法 ──────────────────────────────────────────────────

    def _check_safety(self, code: str) -> tuple[bool, str]:
        for pat in _BLOCKED_PATTERNS:
            if re.search(pat, code):
                return False, f"沙箱拦截：检测到受限操作 `{pat}`，不允许执行。"
        return True, ""

    def _wrap(self, code: str) -> str:
        """将用户代码包裹在 stdout/stderr 重定向逻辑中。"""
        indented = textwrap.indent(code, "    ")
        return textwrap.dedent(f"""\
import sys, io, traceback
_out = io.StringIO()
_err = io.StringIO()
sys.stdout = _out
sys.stderr = _err
try:
{indented}
except Exception:
    _err.write(traceback.format_exc())
finally:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print("__STDOUT__:" + _out.getvalue(), end="")
    print("__STDERR__:" + _err.getvalue(), end="", file=sys.stderr)
""")

    def _parse_output(self, raw_stdout: str, raw_stderr: str) -> tuple[str, str]:
        stdout = ""
        stderr = ""
        if "__STDOUT__:" in raw_stdout:
            stdout = raw_stdout.split("__STDOUT__:", 1)[1]
        else:
            stdout = raw_stdout

        if "__STDERR__:" in raw_stderr:
            stderr = raw_stderr.split("__STDERR__:", 1)[1]
        else:
            stderr = raw_stderr

        return stdout.strip(), stderr.strip()