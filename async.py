"""
兼容入口。

如果你习惯运行 `python async.py`，这里会直接复用 evaluate.py
里的异步评测主流程。
"""

import asyncio

from evaluate import run_tests_async


if __name__ == "__main__":
    asyncio.run(run_tests_async())
