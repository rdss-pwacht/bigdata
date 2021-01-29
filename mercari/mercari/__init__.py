import logging


__version__ = "0.1.0"

logging.basicConfig(
    level=logging.INFO,
    style="{",
    format="{asctime}.{msecs:03.0f} {levelname:>7s} {process:d} "
    + "--- [{threadName:>15.15}] {name:<40.40s} : {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
)
