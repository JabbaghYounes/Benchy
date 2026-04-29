# Allow running as: python -m benchmark
import sys

from benchmark.cli import main

if __name__ == "__main__":
    # Propagate the return code so failing subcommands (e.g. `compile`
    # with a model that fails at HAR generation) actually exit non-zero.
    # Without sys.exit() the interpreter discards main()'s int return
    # and exits 0 by default — which silently turns every failure into
    # a "PASS" for any script wrapping `python -m benchmark`.
    sys.exit(main() or 0)
