import sys
import logging


class CustomFormatter(logging.Formatter):
    """Custom formatter for python logging module.

    Creates separate logging format for error, warning, info and debug logging
    levels.
    """

    err_str = (
        "[%(asctime)s.%(msecs)03d:%(processName)s:%(threadName)s:%(module)s.py:"
        "%(lineno)d]"
    )
    err_fmt = "*" * 100 + f"\n\tERROR {err_str}:\n%(msg)s\n" + "*" * 100
    warn_fmt = "*" * 20 + f" WARNING {err_str}: %(msg)s " + "*" * 20
    info_fmt = "[%(asctime)s.%(msecs)03d] %(message)s"
    dbg_fmt = f"DEBUG {err_str} %(msg)s"

    def __init__(self, fmt="[%(asctime)s.%(msecs)03d] %(message)s"):
        super().__init__(fmt=fmt, datefmt="%T", style="%")

    def format(self, record):
        format_orig = self._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._style._fmt = self.dbg_fmt
        elif record.levelno == logging.INFO:
            self._style._fmt = self.info_fmt
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warn_fmt
        elif record.levelno == logging.ERROR:
            self._style._fmt = self.err_fmt
        result = logging.Formatter.format(self, record)

        self._fmt = format_orig

        return result


CONSOLE = logging.StreamHandler()
CONSOLE.setLevel(logging.INFO)
CONSOLE.setFormatter(CustomFormatter())
ROOT_LOGGER = logging.getLogger("Remora")
ROOT_LOGGER.setLevel(logging.DEBUG)
ROOT_LOGGER.addHandler(CONSOLE)


def init_logger(log_fn=None, quiet=False):
    """Prepare logging output.

    Args:
        log_fn (str): Path to logging output file. All logging messages,
            including debug level, will be output to this file.
        quiet (bool): Set console logging level to warning. Default info.
    """
    log_fp = None
    if log_fn is not None:
        log_fp = logging.FileHandler(log_fn, mode="w")
        log_fp.setLevel(logging.DEBUG)
        log_fp.setFormatter(CustomFormatter())
    if log_fp is not None:
        ROOT_LOGGER.addHandler(log_fp)
    if quiet:
        CONSOLE.setLevel(logging.WARNING)
    logging.getLogger("Remora").debug(f'Command: """{" ".join(sys.argv)}"""')


def get_logger(module_name="Remora"):
    return logging.getLogger(module_name)


if __name__ == "__main__":
    sys.stderr.write("This is a module. See commands with `remora -h`")
    sys.exit(1)
