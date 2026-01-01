"""Logging utilities for nano-vllm-jax.

Provides centralized logging configuration with support for:
- Environment variable control (NANO_VLLM_LOG_LEVEL)
- Per-module debug logging (NANO_VLLM_DEBUG_MODULES)
- Programmatic configuration

Example usage:
    # Setup logging at application entry point
    from nanovllm_jax.utils.logging_utils import setup_logging
    setup_logging('INFO')

    # Or use environment variables:
    # NANO_VLLM_LOG_LEVEL=WARNING python script.py
    # NANO_VLLM_DEBUG_MODULES=nanovllm_jax.layers.attention,nanovllm_jax.engine.model_runner python script.py

    # Get logger in modules
    from nanovllm_jax.utils.logging_utils import get_logger
    logger = get_logger(__name__)
    logger.info("This is an info message")
"""

import os
import logging
from typing import Optional

_LOG_INITIALIZED = False


def setup_logging(
    level: Optional[str] = None,
    enable_debug_modules: Optional[list[str]] = None,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    force: bool = True,
):
    """Setup global logging configuration.

    Args:
        level: Global log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
               Priority: arg > NANO_VLLM_LOG_LEVEL env var > default('INFO')
        enable_debug_modules: List of module names to enable DEBUG level.
                             Example: ['nanovllm_jax.layers.attention']
        format: Log message format string
        datefmt: Date/time format string
        force: If True, override existing logging configuration

    Environment variables:
        NANO_VLLM_LOG_LEVEL: Global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        NANO_VLLM_DEBUG_MODULES: Comma-separated list of modules to enable DEBUG logging
                                Example: "nanovllm_jax.layers.attention,nanovllm_jax.engine"
    """
    global _LOG_INITIALIZED

    if _LOG_INITIALIZED and not force:
        return

    # Determine global log level (priority: arg > env var > default)
    if level is None:
        level = os.getenv("NANO_VLLM_LOG_LEVEL", "INFO")

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format,
        datefmt=datefmt,
        force=force,
    )

    # Set specific modules to DEBUG level
    debug_modules = set()

    # From function argument
    if enable_debug_modules:
        debug_modules.update(enable_debug_modules)

    # From environment variable
    debug_modules_env = os.getenv("NANO_VLLM_DEBUG_MODULES", "")
    if debug_modules_env:
        modules = [m.strip() for m in debug_modules_env.split(",") if m.strip()]
        debug_modules.update(modules)

    # Apply DEBUG level to specified modules
    for module_name in debug_modules:
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(logging.DEBUG)
        logging.getLogger().info(f"Enabled DEBUG logging for module: {module_name}")

    _LOG_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name, typically __name__ of the module

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: str):
    """Dynamically change global log level.

    Args:
        level: Log level string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    logging.getLogger().setLevel(getattr(logging, level.upper()))


def set_module_log_level(module_name: str, level: str):
    """Set log level for a specific module.

    Args:
        module_name: Full module name (e.g., 'nanovllm_jax.layers.attention')
        level: Log level string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    logging.getLogger(module_name).setLevel(getattr(logging, level.upper()))


def disable_logging():
    """Completely disable all logging output."""
    logging.disable(logging.CRITICAL)


def enable_logging():
    """Re-enable logging after disable_logging()."""
    logging.disable(logging.NOTSET)
