"""Allow ``python -m llmbench`` invocation."""

import sys
from .cli import main

sys.exit(main())
