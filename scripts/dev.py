"""
scripts/dev.py —— 项目命令的统一入口。

[WHY] 常用命令各自拖着一串路径（`.venv/Scripts/python.exe src/app/eval/...`），
      记不住；漏掉 `PYTHONIOENCODING=utf-8` 还会中文乱码，看着像程序坏了。
[FIX] 把命令收成一张表按名字转发；子进程统一带上 utf-8 和项目根目录。
"""

import os
import subprocess
import sys
from pathlib import Path

# 输出被管道或重定向接住时（`| grep`、`> 文件`），Windows 默认按 GBK 编码，中文就花了。
# 真控制台本来就按 Unicode 走，所以只在这个分支里钉成 utf-8。
if not sys.stdout.isatty():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 项目根：脚本在 scripts/ 下，往上一级 —— 从哪个目录执行都能对
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "src" / "app" / "eval"

# 命令表：值是「解释器 + 脚本/模块」，额外参数运行时再拼上去。
# 一律用 `python -m`，不写 .venv/Scripts/*.exe 这种绝对路径。
COMMANDS = {
    "ask": [sys.executable, "-m", "app"],
    "eval": [sys.executable, str(EVAL_DIR / "run_eval.py")],
    "judge": [sys.executable, str(EVAL_DIR / "judge_facts.py")],
    "test": [sys.executable, "-m", "pytest", "-q"],
    "lint": [sys.executable, "-m", "ruff", "check", "."],
}

USAGE = """用法：python scripts/dev.py <命令> [参数]

  ask    "问题"        问一句（会真调模型）
  eval   [tag,...]     跑评测，给 tag 就只跑命中的题；不给 = 全量 31 道、约 7 分钟（会真调模型）
  judge  [结果文件]    判「答案召回率」；不给文件就判最新那份（会真调模型）
  test                 跑测试（不调模型、不联网，秒出）
  lint   [--fix]       风格检查，--fix 顺手改掉能自动改的

例子：
  python scripts/dev.py ask "火龙这个怪要怎么打？"
  python scripts/dev.py eval 查族名,俗称
  python scripts/dev.py judge
  python scripts/dev.py lint --fix

多余参数原样传给底层命令，命令本身还是那几条，没有被藏起来。
"""


def _warn_if_wrong_python() -> None:
    """用的不是项目 .venv 的解释器时提醒一句 —— 否则子进程会以 ImportError 收场。"""
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    # 没有 .venv 就没什么可提醒的，交给子进程自己报错
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() != venv_python.resolve():
        print(f"⚠️ 现在用的是 {sys.executable}", file=sys.stderr)
        print(f"   项目的依赖装在 .venv 里，改用：{venv_python} scripts/dev.py ...", file=sys.stderr)


def run_command(name: str, extra_args: list[str]) -> int:
    """把一条命令交给子进程跑，返回它的退出码（int）。"""
    argv = COMMANDS[name] + extra_args

    # 子进程的环境：中文别乱码 —— 这是最常踩的那个坑
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"

    # cwd 钉在项目根：data/、logs/、storage/ 都是相对根目录写的。
    # check=False 是故意的：非零退出码由调用方原样传出去，不在这里抛异常
    result = subprocess.run(argv, cwd=PROJECT_ROOT, env=env, check=False)
    return result.returncode


def main() -> int:
    """按第一个参数选命令、其余原样转发，返回退出码（int）。"""
    args = sys.argv[1:]

    # 没给命令、或者要帮助：打用法
    if not args or args[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    name, extra_args = args[0], args[1:]

    # 打错的命令名不该静默跑别的，列出来让人挑
    if name not in COMMANDS:
        print(f"没有这个命令：{name}\n")
        print(USAGE)
        return 2

    _warn_if_wrong_python()
    return run_command(name, extra_args)


if __name__ == "__main__":
    sys.exit(main())
