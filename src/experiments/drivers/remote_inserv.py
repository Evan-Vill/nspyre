#!/usr/bin/env python
"""
Start up an instrument server to host drivers. For the purposes of this demo,
you can imagine that it is running on a remote system from the one that will
run experimental code, although, in practice, you can just as easily start it on
the local system.
"""
from pathlib import Path
import logging

from nspyre import InstrumentServer
from nspyre import serve_instrument_server_cli
from nspyre import nspyre_init_logger

_HERE = Path(__file__).parent

# log to the console as well as a file inside the logs folder
nspyre_init_logger(
    logging.INFO,
    log_path=_HERE / '../logs',
    log_path_level=logging.DEBUG,
    prefix='remote_inserv',
    file_size=10_000_000,
)

with InstrumentServer(port=42067) as inserv:
    # don't add drivers at this point since they will be handled by the subsystems driver
    serve_instrument_server_cli(inserv)
