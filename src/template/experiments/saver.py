import logging
from functools import partial
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

from nspyre import DataSink
from nspyre import QThreadSafeObject

_HOME = Path.home()
_logger = logging.getLogger(__name__)

class DataSaver(QThreadSafeObject):
    """For automatic file saving at the conclusion of experiments."""

    def __init__(self):
        super().__init__()

    def save(
        self,
        filename: Union[str, Path],
        dataset: str,
        save_fun: Callable,
        timeout: Optional[float] = None,
    ):
        """
        Args:
            filename: The file to save data to.
            dataset: Data set on the data server to pull the data from.
            save_fun: Function that saves the data to a file. It should have
                the signature :code:`save(filename: Union[str, Path], data: Any)`.
            callback: Callback function to run (blocking, in the main thread)
                after the data is saved.
        """
        self.run_safe(
            self._save,
            filename=filename,
            dataset=dataset,
            save_fun=save_fun,
            timeout=timeout,
        )

    def _save(
        self,
        filename: Union[str, Path],
        dataset: str,
        save_fun: Callable,
        timeout: Optional[float] = None,
    ):
        """See save()."""
        try:
            try:
                # connect to the dataserver
                with DataSink(dataset) as sink:
                    # get the data from the dataserver
                    sink.pop(timeout=timeout)
                    save_fun(filename, sink.data)
            except TimeoutError as err:
                raise TimeoutError(
                    f'Timed out retreiving the data set [{dataset}] from data server.'
                ) from err
            else:
                _logger.info(f'Saved data set [{dataset}] to [{filename}].')
        except Exception as err:
            raise err
            



