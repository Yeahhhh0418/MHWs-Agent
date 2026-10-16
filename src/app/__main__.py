"""让 `python -m app` 和 `mhws-agent` 命令等价。"""

from app.cli import main

if __name__ == "__main__":
    main()
