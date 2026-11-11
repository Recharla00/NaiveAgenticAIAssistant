"""Optional, human-readable flow logs for the intermediate agent lessons."""

import logging
import sys


logger = logging.getLogger("agent_app.flow")


def configure_flow_logging(*, verbose: bool = False, debug: bool = False) -> None:
    """Send this application's flow logs to the terminal once per process."""
    if not (verbose or debug):
        return
    if not any(getattr(handler, "_agent_flow_handler", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[flow] %(levelname)s %(message)s"))
        handler._agent_flow_handler = True
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
