import argparse
import os

from remora import __version__

# from remora.common import logging
from remora.parsers import (
    register_train_model,
    register_infer,
    SubcommandHelpFormatter,
)

# LOGGER = logging.get_logger()

_DO_PROFILE = False
# None if environment var not set
_PROF_FN = os.getenv("REMORA_PROFILE_FILE")
if _PROF_FN:
    _DO_PROFILE = True
    # LOGGER.warning(f"Profiling remora. Saving profile data to {_PROF_FN}")


def run():
    """The main routine."""
    # prepare first level `remora -h` help, including description.
    desc = (
        "********** Remora *********\n\nModified base model training and "
        "application.\n\n"
    )
    parser = argparse.ArgumentParser(
        prog="remora",
        description=desc,
        formatter_class=SubcommandHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="Remora version: {}".format(__version__),
        help="Show Remora version and exit.",
    )
    parser.set_defaults(func=lambda x: parser.print_help())

    subparsers = parser.add_subparsers(title="sub-commands")
    register_train_model(subparsers)
    register_infer(subparsers)

    args = parser.parse_args()
    cmd_func = args.func
    if _DO_PROFILE:
        _func_wrapper = cmd_func

        def cmd_func(args):
            import cProfile

            prof = cProfile.Profile()
            retval = prof.runcall(_func_wrapper, args)
            prof.dump_stats(_PROF_FN)
            return retval

    cmd_func(args)


if __name__ == "__main__":
    run()
