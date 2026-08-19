"""Source-checkout entry point; delegates to the installed CLI implementation."""

from civic_kernel.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
