"""Allow ``python -m dictate_mac`` invocation."""

from dictate_mac.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
