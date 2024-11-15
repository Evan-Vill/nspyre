"""This driver can boot and shutdown other drivers."""
import logging
from pathlib import Path

from nspyre.extras import Subsystem

_HERE = Path(__file__).parent
_logger = logging.getLogger(__name__)

class SubsystemsDriver:
    """Handle boot sequencing for drivers that require it."""
    def __init__(self, local_inserv, remote_inserv):
        """
        Args:
            local_inserv: The local instrument server.
            remote_inserv: The remote instrument server (gateway connection).
        """
        # dictionary containing all of the Subsystem objects
        self.subsystems = []

        ps_sub = Subsystem(
            'ps',
            default_boot_inserv=remote_inserv,
            default_boot_add_args=['ps',
                                    _HERE / 'fake_ps_driver.py',
                                    'FakePSDriver'],
            default_boot_timeout=1
        )
        self.subsystems.append(ps_sub)

        io_sub = Subsystem(
            'io',
            default_boot_inserv=remote_inserv,
            default_boot_add_args=['io',
                                    _HERE / 'fake_io_driver.py',
                                    'FakeIODriver'],
            default_boot_timeout=1,
            dependencies=[ps_sub]
        )
        self.subsystems.append(io_sub)
