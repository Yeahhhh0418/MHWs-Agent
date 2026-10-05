"""让 `python -m app` 和 `mhwi-agent` 命令等价。"""

from app.cli import main

if __name__ == "__main__":
    main()
