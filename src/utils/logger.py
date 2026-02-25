import logging
import colorlog


def initialize_logger(level: int = logging.DEBUG) -> None:
    """
    Initialize global colored logging configuration.
    Call this ONCE in your entry-point script.
    """

    # Prevent duplicate handlers if called multiple times
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    handler = colorlog.StreamHandler()
    if level == logging.DEBUG:
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "white",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
    else:
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "white",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )

    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Silence noisy libraries 
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PIL.Image").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("fsspec.local").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    


# Showcase when run directly
if __name__ == "__main__":
    initialize_logger(logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")