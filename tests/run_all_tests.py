#!/usr/bin/env python3
"""
Run All Tests

Usage:
    python tests/run_all_tests.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 60)
    print("NanoNav Test Suite")
    print("=" * 60)

    from tests import test_a_star
    from tests import test_map_generator
    from tests import test_dataset
    from tests import test_model
    from tests import test_end_to_end

    all_passed = True

    try:
        test_a_star.run_all()
    except Exception as e:
        print(f"\n❌ A* tests failed: {e}")
        all_passed = False

    try:
        test_map_generator.run_all()
    except Exception as e:
        print(f"\n❌ Map generator tests failed: {e}")
        all_passed = False

    try:
        test_dataset.run_all()
    except Exception as e:
        print(f"\n❌ Dataset tests failed: {e}")
        all_passed = False

    try:
        test_model.run_all()
    except Exception as e:
        print(f"\n❌ Model tests failed: {e}")
        all_passed = False

    try:
        test_end_to_end.run_all()
    except Exception as e:
        print(f"\n❌ End-to-end tests failed: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nReady for training:")
        print("  python scripts/generate_dataset.py")
        print("  python -m trm_nav.train")
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
