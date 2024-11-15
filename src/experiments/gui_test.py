"""
Example GUI elements.
"""
# from tkinter.tix import ComboBox
from tracemalloc import stop
import numpy as np
from numpy.fft import fft, ifft
import logging
from functools import partial
from importlib import reload
from multiprocessing import Queue
from typing import Optional

from inspect import signature
from scipy.optimize import curve_fit
from rpyc.utils.classic import obtain

# from nspyre import FlexLinePlotWidget, LinePlotWidget
from nspyre.gui.widgets.flex_line_plot_2 import FlexLinePlotWidget
from nspyre import DataSink
from pyqtgraph import SpinBox, ComboBox
from pyqtgraph import PlotWidget
from pyqtgraph.Qt import QtWidgets

from PyQt6.QtWidgets import QLabel, QPushButton, QCheckBox, QComboBox, QLineEdit, QRadioButton, QFileDialog, QProgressBar, QSlider, QDial
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout
from PyQt6.QtWidgets import QStackedWidget, QWidget, QGraphicsOpacityEffect, QSizePolicy
from PyQt6.QtGui import QFont, QColor, QPainter, QBrush
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal, QRect, QSize

from nspyre.misc.misc import ProcessRunner
from nspyre.misc.misc import run_experiment
from nspyre import ParamsWidget
from nspyre import experiment_widget_process_queue
from nspyre import InstrumentManager

import nv_experiments_sideband

class Communicate(QObject):
    # Create a custom signal
    speak = pyqtSignal()

class ExpWidget(QWidget):
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle('NV Experiments')

        # parameter defaults for different experiments
        
        self.sideband_opts = ["Lower", "Upper"]
        self.params_defaults = [5e-5, 30e6, 0.45, self.sideband_opts, -0.004, -0.004]

        # experiment params label
        self.exp_label = QLabel("Experimental Parameters:")
        self.exp_label.setFixedHeight(20)
        self.exp_label.setStyleSheet("font-weight: bold")

        self.params_widget = ParamsWidget(
            params = {
                        'probe': {'display_text': 'CW ODMR MW Probe Time: ',
                                'widget': SpinBox(value = self.params_defaults[0], suffix = 's', siPrefix = True, bounds = (10e-9, None))},
                        'sideband_freq': {'display_text': 'Sideband Modulation Frequency: ',
                                'widget': SpinBox(value = self.params_defaults[1], suffix = 'Hz', siPrefix = True, bounds = (100, 100e6), dec = True)},
                        'sideband_power': {'display_text': 'Sideband Power: ',
                                        'widget': SpinBox(value = self.params_defaults[2], suffix = 'V', siPrefix = True)},
                        'sideband': {'display_text': 'Sideband Choice: ',
                                'widget': ComboBox(items = self.params_defaults[3])},
                        'i_offset': {'display_text': 'I Offset: ',
                                        'widget': SpinBox(value = self.params_defaults[4], suffix = 'V', siPrefix = True)},
                        'q_offset': {'display_text': 'Q Offset: ',
                                        'widget': SpinBox(value = self.params_defaults[5], suffix = 'V', siPrefix = True)}}
        )

        # save params button
        self.apply_params = QPushButton("Apply Parameters")
        self.apply_params.clicked.connect(lambda: self.apply_params_clicked())

        self.c = Communicate()        
        self.emitted_params: dict() = {}

        self.emitted_params['laser power'] = 1
        # unpack all keyword arg parameters to send to experiment process
        self.fun_kwargs = dict(**self.params_widget.all_params(), **self.emitted_params)

        self.gui_layout = QVBoxLayout()

        self.exp_frame = QFrame(self)
        self.exp_frame.setStyleSheet("background-color: #2b2b2b")
        self.exp_params_layout = QVBoxLayout(self.exp_frame)
        self.exp_params_layout.setSpacing(0)
        self.exp_params_layout.addWidget(self.exp_label)
        self.exp_params_layout.addWidget(self.params_widget)
        self.exp_params_layout.addWidget(self.apply_params)
     
        self.exp_widgets_layout = QVBoxLayout()
        self.exp_widgets_layout.addWidget(self.exp_frame)

        self.gui_layout.addLayout(self.exp_widgets_layout)


        self.setLayout(self.gui_layout)

    def apply_params_clicked(self):
        # Emit the params to the experiment widget for when "run" is pressed
        self.c.speak.emit(self.fun_kwargs)

class _Bar(QWidget):

    clickedValue = pyqtSignal(int)

    def __init__(self, steps, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding,
            QSizePolicy.Policy.MinimumExpanding
        )

        if isinstance(steps, list):
            # list of colors.
            self.n_steps = len(steps)
            self.steps = steps

        elif isinstance(steps, int):
            # int number of bars, defaults to red.
            self.n_steps = steps
            self.steps = ['green'] * steps

        else:
            raise TypeError('steps must be a list or int')

        self._bar_solid_percent = 0.8
        self._background_color = QColor('black')
        self._padding = 4.0  # n-pixel gap around edge.

    def paintEvent(self, e):
        painter = QPainter(self)

        brush = QBrush()
        brush.setColor(self._background_color)
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        rect = QRect(0, 0, painter.device().width(), painter.device().height())
        painter.fillRect(rect, brush)

        # Get current state.
        parent = self.parent()
        vmin, vmax = parent.minimum(), parent.maximum()
        value = parent.value()

        # Define our canvas.
        d_height = painter.device().height() - (self._padding * 2)
        d_width = painter.device().width() - (self._padding * 2)

        # Draw the bars.
        step_size = d_height / self.n_steps
        bar_height = step_size * self._bar_solid_percent
        bar_spacer = step_size * (1 - self._bar_solid_percent) / 2

        # Calculate the y-stop position, from the value in range.
        pc = (value - vmin) / (vmax - vmin)
        n_steps_to_draw = int(pc * self.n_steps)

        for n in range(n_steps_to_draw):
            brush.setColor(QColor(self.steps[n]))
            rect = QRect(
                int(self._padding),
                int(self._padding + d_height - ((1 + n) * step_size) + bar_spacer),
                int(d_width),
                int(bar_height)
            )
            painter.fillRect(rect, brush)

        painter.end()

    def sizeHint(self):
        return QSize(40, 120)

    def _trigger_refresh(self):
        self.update()

    def _calculate_clicked_value(self, e):
        parent = self.parent()
        vmin, vmax = parent.minimum(), parent.maximum()
        d_height = self.size().height() + (self._padding * 2)
        step_size = d_height / self.n_steps
        click_y = e.y() - self._padding - step_size / 2

        pc = (d_height - click_y) / d_height
        value = vmin + pc * (vmax - vmin)
        self.clickedValue.emit(value)

    def mouseMoveEvent(self, e):
        self._calculate_clicked_value(e)

    def mousePressEvent(self, e):
        self._calculate_clicked_value(e)


class PowerBar(QWidget):
    """
    Custom Qt Widget to show a power bar and dial.
    Demonstrating compound and custom-drawn widget.

    Left-clicking the button shows the color-chooser, while
    right-clicking resets the color to None (no-color).
    """

    colorChanged = pyqtSignal()

    def __init__(self, steps=25, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QVBoxLayout()
        self._bar = _Bar(steps)
        layout.addWidget(self._bar)

        self.laser_pc_label = QLabel("Laser power: 0%")
        layout.addWidget(self.laser_pc_label)
        # Create the QDial widget and set up defaults.
        # - we provide accessors on this class to override.
        self._slider = QSlider()
        self._slider.setMinimum(0)
        self._slider.setMaximum(110)
        self._slider.setValue(0)
        self._slider.setTickInterval(1)
        # self._slider.setNotchesVisible(True)
        # self._slider.setWrapping(False)
        self._slider.valueChanged.connect(self._bar._trigger_refresh)
        self._slider.valueChanged.connect(self.update_label)

        # Take feedback from click events on the meter.
        self._bar.clickedValue.connect(self._slider.setValue)

        layout.addWidget(self._slider)
        self.setLayout(layout)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self[name]

        return getattr(self._slider, name)

    def setColor(self, color):
        self._bar.steps = [color] * self._bar.n_steps
        self._bar.update()

    def setColors(self, colors):
        self._bar.n_steps = len(colors)
        self._bar.steps = colors
        self._bar.update()

    def setBarPadding(self, i):
        self._bar._padding = int(i)
        self._bar.update()

    def setBarSolidPercent(self, f):
        self._bar._bar_solid_percent = float(f)
        self._bar.update()

    def setBackgroundColor(self, color):
        self._bar._background_color = QColor(color)
        self._bar.update()

    def update_label(self):
        self.laser_power = self._slider.value()
        self.laser_pc_label.setText(f"Laser power: {self.laser_power}%")


