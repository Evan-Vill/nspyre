"""
Example GUI elements.
"""
import numpy as np

from nspyre import FlexLinePlotWidget
from nspyre import ExperimentWidget
from nspyre import DataSink
from pyqtgraph import SpinBox
from pyqtgraph.Qt import QtWidgets

import template.experiments.odmr

class ODMRWidget(ExperimentWidget):
    def __init__(self):
        params_config = {
            'start_freq': {
                'display_text': 'Start Frequency',
                'widget': SpinBox(
                    value=3e9,
                    suffix='Hz',
                    siPrefix=True,
                    bounds=(100e3, 10e9),
                    dec=True,
                ),
            },
            'stop_freq': {
                'display_text': 'Stop Frequency',
                'widget': SpinBox(
                    value=4e9,
                    suffix='Hz',
                    siPrefix=True,
                    bounds=(100e3, 10e9),
                    dec=True,
                ),
            },
            'num_points': {
                'display_text': 'Number of Scan Points',
                'widget': SpinBox(value=101, int=True, bounds=(1, None), dec=True),
            },
            'iterations': {
                'display_text': 'Number of Experiment Repeats',
                'widget': SpinBox(value=50, int=True, bounds=(1, None), dec=True),
            },
            'dataset': {
                'display_text': 'Data Set',
                'widget': QtWidgets.QLineEdit('odmr'),
            },
        }

        super().__init__(params_config, 
                        template.experiments.odmr,
                        'SpinMeasurements',
                        'odmr_sweep',
                        title='ODMR')

def process_ODMR_data(sink: DataSink):
    """Subtract the signal from background trace and add it as a new 'diff' dataset."""
    diff_sweeps = []
    for s,_ in enumerate(sink.datasets['signal']):
        freqs = sink.datasets['signal'][s][0]
        sig = sink.datasets['signal'][s][1]
        bg = sink.datasets['background'][s][1]
        diff_sweeps.append(np.stack([freqs, sig - bg]))
    sink.datasets['diff'] = diff_sweeps

class FlexLinePlotWidgetWithODMRDefaults(FlexLinePlotWidget):
    """Add some default settings to the FlexSinkLinePlotWidget."""
    def __init__(self):
        super().__init__(data_processing_func=process_ODMR_data)
        # create some default signal plots
        self.add_plot('sig_avg',        series='signal',   scan_i='',     scan_j='',  processing='Average')
        self.add_plot('sig_latest',     series='signal',   scan_i='-1',   scan_j='',  processing='Average')
        self.add_plot('sig_first',      series='signal',   scan_i='0',    scan_j='1', processing='Average')
        self.add_plot('sig_latest_10',  series='signal',   scan_i='-10',  scan_j='',  processing='Average')
        self.hide_plot('sig_first')
        self.hide_plot('sig_latest_10')

        # create some default background plots
        self.add_plot('bg_avg',         series='background',   scan_i='',     scan_j='',  processing='Average')
        self.add_plot('bg_latest',      series='background',   scan_i='-1',   scan_j='',  processing='Average')

        # create some default diff plots
        self.add_plot('diff_avg',       series='diff',  scan_i='',      scan_j='',  processing='Average')
        self.add_plot('diff_latest',    series='diff',  scan_i='-1',    scan_j='',  processing='Average')

        # manually set the XY range
        self.line_plot.plot_item().setXRange(3.0, 4.0)
        self.line_plot.plot_item().setYRange(-3000, 4500)

        # retrieve legend object
        legend = self.line_plot.plot_widget.addLegend()
        # set the legend location
        legend.setOffset((-10, -50))

        self.datasource_lineedit.setText('odmr')
