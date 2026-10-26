"""Entry shim so ``python -m src.interfaces.cli`` executes the CLI."""

import sys

from src.interfaces.cli import main

sys.exit(main())
