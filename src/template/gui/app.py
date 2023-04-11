#!/usr/bin/env python
"""
This is an example script that demonstrates the basic functionality of nspyre.
"""
import logging
from pathlib import Path

# in order for dynamic reloading of code to work, you must pass the specifc
# module containing your class to MainWidgetItem, since the python reload()
# function does not recursively reload modules
import template.gui.elements

from nspyre.misc.stacktracer import trace_start, trace_stop
import nspyre.gui.widgets.save
import nspyre.gui.widgets.load
import nspyre.gui.widgets.flex_line_plot
from nspyre import MainWidget
from nspyre import MainWidgetItem
from nspyre import nspyre_init_logger
from nspyre import nspyreApp

HERE = Path(__file__).parent


def main():
    # Log to the console as well as a file inside the logs folder.
    nspyre_init_logger(
        log_level=logging.INFO,
        log_path=HERE / 'logs',
        log_path_level=logging.DEBUG,
        prefix=Path(__file__).stem,
        file_size=10_000_000,
    )

    # Create Qt application and apply nspyre visual settings.
    app = nspyreApp()

    # Create the GUI.
    main_widget = MainWidget(
        {
            'ODMR': MainWidgetItem(template.gui.elements, 'ODMRWidget', stretch=(1, 1)),
            'Plots': {
                'FlexLinePlotDemo': MainWidgetItem(
                    template.gui.elements,
                    'FlexLinePlotWidgetWithODMRDefaults',
                    stretch=(100, 100),
                ),
                'FlexLinePlot': MainWidgetItem(
                    nspyre.gui.widgets.flex_line_plot,
                    'FlexLinePlotWidget',
                    stretch=(100, 100),
                ),
            },
            'Save': MainWidgetItem(nspyre.gui.widgets.save, 'SaveWidget', stretch=(1, 1)),
            'Load': MainWidgetItem(nspyre.gui.widgets.load, 'LoadWidget', stretch=(1, 1)),
        }
    )
    main_widget.show()

    # TODO
    trace_start('trace.html', interval=5)
    # Run the GUI event loop.
    app.exec()
    trace_stop()


# if using the nspyre ProcessRunner, the main code must be guarded with if __name__ == '__main__':
# see https://docs.python.org/2/library/multiprocessing.html#windows
if __name__ == '__main__':
    main()
