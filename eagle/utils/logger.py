"""
Eagle Utils: Structured Logging
=================================
Rich-formatted console logging with file rotation and structured
JSON output for production use.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


def setup_logging(config: Any) -> None:
    """
    Configure Eagle logging system.

    Sets up:
        - Rich console handler with color formatting
        - Rotating file handler with size limits
        - Structured format for both outputs
    """
    log_config = config.logging
    level = getattr(logging, config.app.log_level.upper(), logging.INFO)

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Clear any existing handlers
    root.handlers.clear()

    # Console handler
    if log_config.console:
        try:
            from rich.logging import RichHandler

            console_handler = RichHandler(
                level=level,
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
                tracebacks_show_locals=False,
            )
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        except ImportError:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(levelname)-8s %(name)-20s %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
        root.addHandler(console_handler)

    # File handler
    if log_config.file:
        log_path = Path(log_config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=log_config.max_file_size_mb * 1024 * 1024,
            backupCount=log_config.backup_count,
        )
        file_handler.setLevel(level)

        if log_config.structured:
            file_handler.setFormatter(
                logging.Formatter(
                    '{"time":"%(asctime)s","level":"%(levelname)s",'
                    '"module":"%(name)s","message":"%(message)s"}',
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            )
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(levelname)-8s %(name)-25s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ["urllib3", "asyncio", "yfinance", "peewee"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("eagle").info("Logging system initialized")
