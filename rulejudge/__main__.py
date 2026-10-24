"""Entry point: python -m rulejudge adapt|eval  (Docker ENTRYPOINT)."""
import sys

from .pipeline import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
