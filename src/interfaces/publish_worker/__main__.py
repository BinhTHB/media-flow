"""Entry shim so ``python -m src.interfaces.publish_worker`` runs the worker."""

import sys

from src.interfaces.publish_worker import main

sys.exit(main())
