"""一次跑全部单测(纯逻辑,不依赖外部服务)。

用法:
    python run_tests.py               # 跑所有
    python run_tests.py test_loop      # 只跑 test_loop.py

测的是控制流契约,不是端到端业务。所以即使 Neo4j/Qdrant 都没起,这套测试都能跑。
"""

import sys
import unittest
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_this = Path(__file__).resolve().parent
sys.path.insert(0, str(_this))

if __name__ == "__main__":
    pattern = sys.argv[1] + ".py" if len(sys.argv) > 1 else "test_*.py"
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(_this / "tests"), pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
