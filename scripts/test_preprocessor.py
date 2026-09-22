"""Runner for the preprocessing unit test suite (Stage 2, item 15).

Usage::

    python scripts/test_preprocessor.py            # run the suite
    python scripts/test_preprocessor.py -v         # verbose

The actual tests live in tests/preprocessing/test_preprocessor.py and run on
a synthetic mini-dataset, so they never touch the real 3.5M-row data.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    import pytest

    test_file = PROJECT_ROOT / "tests" / "preprocessing" / "test_preprocessor.py"
    args = [str(test_file), "-c", str(PROJECT_ROOT / "pytest.ini"), *sys.argv[1:]]
    raise SystemExit(pytest.main(args))


if __name__ == "__main__":
    main()
