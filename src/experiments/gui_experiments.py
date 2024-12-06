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

from PyQt6.QtWidgets import QLabel, QPushButton, QCheckBox, QComboBox, QLineEdit, QRadioButton, QFileDialog, QProgressBar
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout
from PyQt6.QtWidgets import QStackedWidget, QWidget, QGraphicsOpacityEffect
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCore import Qt, QTimer, pyqtSlot

from nspyre.misc.misc import ProcessRunner
from nspyre.misc.misc import run_experiment
from nspyre import ParamsWidget
from nspyre import experiment_widget_process_queue
from nspyre import InstrumentManager

from gui_test import Communicate 

import nv_experiments_12_5_24

class ExpWidget(QWidget):
    
    QUEUE_CHECK_TIME = 50 # ms

    def __init__(self):
        super().__init__()

        self.setWindowTitle('NV Experiments')

        self.updateTimer = QTimer() #create a timer that will try to update that widget with messages from the from_exp_queue
        self.updateTimer.timeout.connect(lambda: self.check_queue_from_exp())
        self.updateTimer.start(self.QUEUE_CHECK_TIME)
            
        # parameter defaults for different experiments
        
        self.sideband_opts = ["Lower", "Upper"]
        self.sideband_cw_opts = ["Lower", "Upper"]
        self.detector_opts = ["APD", "PMT"]
        self.detector_cw_opts = ["APD", "PMT"]

        self.dig_ro_chan_opts = ["0", "1"]

        self.sigvstime_detector_opts = ["APD", "PMT"]

        self.rabi_axis_opts = ["y", "x"]
        self.rabi_device_opts = ["SRS", "AWG"]
       
        self.opt_t1_array_opts = ["geomspace", "linspace"]
        self.mw_t1_array_opts = ["geomspace", "linspace"]
        self.t2_array_opts = ["geomspace", "linspace"]
        self.dq_array_opts = ["geomspace", "linspace"]
        self.fid_array_opts = ["geomspace", "linspace"]
        self.fid_cd_array_opts = ["geomspace", "linspace"]
        self.t2_seq_opts = ["Ramsey", "Echo", "XY4", "YY4", "XY8", "YY8", "CPMG", "PulsePol"]
        self.deer_drive_opts = ["Pulsed", "Continuous"]
        self.fid_drive_opts = ["Pulsed", "Continuous"]
        self.corr_t1_array_opts = ["geomspace", "linspace"]
        # self.nmr_seq_opts = ["Correlation Spectroscopy", "CASR", "AERIS", "PROPI"]
        
        self.sigvstime_params_defaults = [100, self.sigvstime_detector_opts, 'sigvstime']

        self.laser_params_defaults = [30, 30e6, 0.45, self.sideband_opts, -0.002, -0.004, self.detector_opts]
        self.laser_cw_params_defaults = [30, 5e-5, 30e6, 0.15, self.sideband_cw_opts, -0.002, -0.004, self.detector_cw_opts]
        self.digitizer_defaults = [512, 5, 500e6, 0.5, self.dig_ro_chan_opts, 32]

        self.odmr_params_defaults = [12, 10, 2.87e9, 100e6, 30, 1e-9, 300, 'odmr']
        self.rabi_params_defaults = [12, 10, 0, 500e-9, 100, 2.87e9, 1e-9, self.rabi_axis_opts, self.rabi_device_opts, 300, 'rabi']    
        self.pulsed_odmr_params_defaults = [12, 10, 2.87e9, 100e6, 30, 1e-9, 100e-9, 300, 'odmr']
        self.opt_t1_params_defaults = [12, 10, 50e-9, 100e-6, 100, self.opt_t1_array_opts, 300, 't1']
        self.mw_t1_params_defaults = [12, 10, 50e-9, 100e-6, 100, self.mw_t1_array_opts, 2.87e9, 1e-9, 20e-9, 'y', 300, 't1']
        self.t2_params_defaults = [12, 10, 50e-9, 20e-6, 100, self.t2_array_opts, 2.87e9, 1e-9, 20e-9, 'y', self.t2_seq_opts, 1, 300, 't2']

        self.dq_params_defaults = [12, 10, 50e-9, 100e-6, 100, self.dq_array_opts, 2.87e9, 1e-9, 20e-9, 2.87e9, 1e-9, 20e-9, 'y', 300, 'dq']

        self.deer_params_defaults = [12, 10, 350e6, 750e6, 201, 800e-9, 2.87e9, 1e-9, 20e-9, 'y', 0.2, 40e-9, self.deer_drive_opts, 300, 'deer']      
        self.deer_rabi_params_defaults = [12, 10, 3e-9, 100e-9, 100, 800e-9, 2.87e9, 1e-9, 20e-9, 'y', 560e6, 0.2, 300, 'deer rabi']     
        self.deer_fid_params_defaults = [12, 10, 50e-9, 20e-6, 100, self.fid_array_opts, 2.87e9, 1e-9, 20e-9, 'y', 560e6, 0.2, 40e-9, 1, 300, 'fid']
        
        self.deer_fid_cd_params_defaults = [12, 10, 50e-9, 20e-6, 100, self.fid_cd_array_opts, 2.87e9, 1e-9, 20e-9, 'y', 560e6, 0.2, 40e-9, 0.1, 1, 300, 'fid cd']      
        
        self.deer_corr_rabi_params_defaults = [12, 10, 3e-9, 100e-9, 100, 800e-9, 1e-6, 2.87e9, 1e-9, 20e-9, 'y', 560e6, 40e-9, 0.2, 300, 'corr rabi']
        self.deer_corr_t1_params_defaults = [12, 10, 50e-9, 1e-6, 100, self.corr_t1_array_opts, 800e-9, 2.87e9, 1e-9, 20e-9, 'y', 560e6, 40e-9, 0.2, 300, 'corr t1']
        self.nmr_params_defaults = [12, 10, 50e-9, 100e-6, 100, 1e-6, 2.87e9, 1e-9, 20e-9, 'y', 1, 300, 'nmr']

        self.casr_params_defaults = [12, 10, 10, 1e-6, 2.87e9, 1e-9, 20e-9, 1, 300, 'casr']
        
        self.exp_dict = {"Signal vs Time": ["sigvstime_scan", self.sigvstime_params_defaults],
                    "CW ODMR": ["odmr_scan", self.odmr_params_defaults],
                    "Laser": [None, self.laser_params_defaults],
                    "CW Laser": [None, self.laser_cw_params_defaults],
                    "Digitizer": [None, self.digitizer_defaults],
                    "Pulsed ODMR": ["pulsed_odmr_scan", self.pulsed_odmr_params_defaults],
                    "Rabi": ["rabi_scan", self.rabi_params_defaults],
                    "Optical T1": ["OPT_T1_scan", self.opt_t1_params_defaults],
                    "MW T1": ["MW_T1_scan", self.mw_t1_params_defaults],
                    "T2": ["T2_scan", self.t2_params_defaults],
                    "DQ Relaxation": ["DQ_scan", self.dq_params_defaults],
                    "DEER": ["DEER_scan", self.deer_params_defaults],
                    "DEER Rabi": ["DEER_rabi_scan", self.deer_rabi_params_defaults],
                    "DEER FID": ["DEER_FID_scan", self.deer_fid_params_defaults],
                    "DEER FID Continuous Drive": ["DEER_FID_CD_scan", self.deer_fid_cd_params_defaults],
                    "DEER Correlation Rabi": ["DEER_corr_rabi_scan", self.deer_corr_rabi_params_defaults],
                    "DEER T1": ["DEER_corr_T1_scan", self.deer_corr_t1_params_defaults],
                    "DEER T2": ["DEER_corr_T1_scan", self.deer_corr_t1_params_defaults],
                    "NMR: Correlation Spectroscopy": ["Corr_Spec_scan", self.nmr_params_defaults],
                    "NMR: CASR": ["CASR_scan", self.casr_params_defaults]}
        
        self.experiments_label = QLabel("Experiment")
        self.experiments_label.setFixedHeight(25)
        self.experiments_label.setStyleSheet("font-weight: bold")        
        self.experiments = QComboBox()
        self.experiments.setFixedHeight(30)
        self.experiments.addItems(["Select an experiment from dropdown menu", 
                                 "Signal vs Time", 
                                 "CW ODMR", 
                                 "Pulsed ODMR",
                                 "Rabi", 
                                 "Optical T1",
                                 "MW T1", 
                                 "T2", 
                                 "DQ Relaxation",
                                 "DEER", 
                                 "DEER Rabi",
                                 "DEER FID",
                                 "DEER FID Continuous Drive",
                                 "DEER Correlation Rabi",
                                 "DEER T1",
                                 "DEER T2",
                                 "NMR: Correlation Spectroscopy",
                                 "NMR: CASR"])
        
        self.experiments.currentIndexChanged.connect(lambda: self.exp_selector())
  
        self.to_save = False
        self.extra_kwarg_params: dict() = {}

        # experiment params label
        self.exp_label = QLabel("Experiment Parameters")
        self.exp_label.setFixedHeight(24)
        self.exp_label.setStyleSheet("font-weight: bold")

        # print("EXP DICT: ", self.exp_dict['CW ODMR'][1])
        self.params_widget = ParamsWidget(self.create_params_widget('CW ODMR', self.exp_dict['CW ODMR'][1]))

        self.opacity_effects = []
        for i in range(11):
            self.opacity_effects.append(QGraphicsOpacityEffect())
            self.opacity_effects[i].setOpacity(0.3)

        self.exp_label.setGraphicsEffect(self.opacity_effects[0])
        self.params_widget.setGraphicsEffect(self.opacity_effects[1])
        self.params_widget.setEnabled(False)
        
        # save params button
        self.save_params = QPushButton("Save Parameters")
        self.save_params.clicked.connect(lambda: self.save_params_clicked())
        self.save_params.setGraphicsEffect(self.opacity_effects[2])
        self.save_params.setEnabled(False)

        # laser params widget & label
        self.laser_params_widget = ParamsWidget(self.create_params_widget('CW Laser', self.exp_dict['CW Laser'][1]))
        self.laser_label = QLabel("Laser & IQ Parameters")
        self.laser_label.setFixedHeight(24)
        self.laser_label.setStyleSheet("font-weight: bold")
        self.laser_label.setGraphicsEffect(self.opacity_effects[3])
        self.laser_params_widget.setGraphicsEffect(self.opacity_effects[4])
        self.laser_params_widget.setEnabled(False)

        self.dig_params_widget = ParamsWidget(self.create_params_widget('Digitizer', self.exp_dict['Digitizer'][1]))
        self.dig_label = QLabel("Digitizer Parameters")
        self.dig_label.setFixedHeight(24)
        self.dig_label.setStyleSheet("font-weight: bold")
        self.dig_label.setGraphicsEffect(self.opacity_effects[5])
        self.dig_params_widget.setGraphicsEffect(self.opacity_effects[6])
        self.dig_params_widget.setEnabled(False)

        # auto save checkbox
        self.auto_save_checkbox = QCheckBox("Auto Save")
        self.auto_save_checkbox.setChecked(False)
        self.auto_save_checkbox.stateChanged.connect(lambda: self.auto_save_changed())

        # select directory button
        self.select_dir_button = QPushButton("Select Directory")
        self.select_dir_button.setEnabled(False)
        self.select_dir_button.clicked.connect(lambda: self.select_directory())
        self.select_dir_button.setGraphicsEffect(self.opacity_effects[7])

        # selected directory display for saving
        self.chosen_dir = QLabel()
        self.chosen_dir.setGraphicsEffect(self.opacity_effects[8])
        self.chosen_dir.setStyleSheet("color: #ffa500")

        self.filename_label = QLabel("Filename: ")
        self.filename_label.setGraphicsEffect(self.opacity_effects[9])
        self.filename_label.setFixedHeight(20)

        self.filename_lineedit = QLineEdit()
        self.filename_lineedit.setGraphicsEffect(self.opacity_effects[10])
        self.filename_lineedit.setFixedHeight(30)
        self.filename_lineedit.setEnabled(False)

        # status label
        self.status = QLabel("Select parameters and press 'Run' to begin experiment.")
        self.status.setStyleSheet("color: black; background-color: #00b8ff; border: 4px solid black;")
        self.status.setFixedHeight(40)
        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        # run button
        run_button = QPushButton('Run')
        run_button.setStyleSheet("border: 2px solid limegreen")
        self.run_proc = ProcessRunner()
        run_button.clicked.connect(self.run)

        self.queue_to_exp: Queue = Queue()
        """multiprocessing Queue to pass to the experiment subprocess and use
        for sending messages to the subprocess."""
        self.queue_from_exp: Queue = Queue()
        """multiprocessing Queue to pass to the experiment subprocess and use
        for receiving messages from the subprocess."""

        # stop button
        stop_button = QPushButton('Stop')
        stop_button.setStyleSheet("border: 2px solid red")
        stop_button.clicked.connect(self.stop)
        # use a partial because the stop function may already be destroyed by the time
        # this is called
        self.destroyed.connect(partial(self.stop, log=False))

        self.gui_layout = QVBoxLayout()
        
        self.top_frame = QFrame(self)
        self.top_frame.setStyleSheet("background-color: #2b2b2b")
        self.top_layout = QVBoxLayout(self.top_frame)
        self.top_layout.setSpacing(0)
        self.top_layout.addWidget(self.experiments_label)
        self.top_layout.addWidget(self.experiments)

        self.exp_frame = QFrame(self)
        self.exp_frame.setStyleSheet("background-color: #2b2b2b")
        self.exp_params_layout = QVBoxLayout(self.exp_frame)
        self.exp_params_layout.setSpacing(0)
        self.exp_params_layout.addWidget(self.exp_label)
        self.exp_params_layout.addWidget(self.params_widget)
        self.exp_params_layout.addWidget(self.save_params)

        self.save_frame = QFrame(self)
        self.save_frame.setStyleSheet("background-color: #454545")
        self.save_layout = QGridLayout(self.save_frame)
        self.save_layout.setSpacing(0)
        self.save_layout.addWidget(self.auto_save_checkbox,1,1,1,1)
        self.save_layout.addWidget(self.select_dir_button,1,2,1,1)
        self.save_layout.addWidget(self.chosen_dir,2,1,1,2)
        self.save_layout.addWidget(self.filename_label,3,1,1,1)
        self.save_layout.addWidget(self.filename_lineedit,3,2,1,1)

        self.bottom_frame = QFrame(self)
        self.bottom_frame.setStyleSheet("background-color: black")
        self.bottom_layout = QGridLayout(self.bottom_frame)
        self.bottom_layout.setSpacing(0)
        self.bottom_layout.addWidget(self.status,1,1,1,2)
        self.bottom_layout.addWidget(self.progress_bar,2,1,1,2)
        self.bottom_layout.addWidget(run_button,3,1,1,1)
        self.bottom_layout.addWidget(stop_button,3,2,1,1)

        self.laser_frame = QFrame(self)
        self.laser_frame.setStyleSheet("background-color: #2b2b2b")
        self.laser_params_layout = QGridLayout(self.laser_frame)
        self.laser_params_layout.setSpacing(0)
        self.laser_params_layout.addWidget(self.laser_label,1,1,1,1)
        self.laser_params_layout.addWidget(self.laser_params_widget,2,1,1,1)
        self.laser_params_layout.addWidget(self.dig_label,1,2,1,1)
        self.laser_params_layout.addWidget(self.dig_params_widget,2,2,1,1)
        
        # self.gui_layout.addLayout(self.top_layout)
        # self.gui_layout.addLayout(self.exp_params_layout)
        # self.gui_layout.addLayout(self.laser_params_layout)
        # self.gui_layout.addLayout(self.bottom_layout)

        # self.stacked_widget = QStackedWidget()
        # self.stacked_widget.addWidget(self.top_frame)
        # self.stacked_widget.addWidget(self.exp_frame)
        # self.stacked_widget.addWidget(self.laser_frame)
        # self.stacked_widget.addWidget(self.bottom_frame)
        # self.gui_layout.addWidget(self.stacked_widget)

        self.top_widgets_layout = QVBoxLayout()
        self.top_widgets_layout.addWidget(self.top_frame)

        self.exp_widgets_layout = QVBoxLayout()
        self.exp_widgets_layout.addWidget(self.exp_frame)

        self.laser_widgets_layout = QVBoxLayout()
        self.laser_widgets_layout.addWidget(self.laser_frame)

        self.save_widgets_layout = QVBoxLayout()
        self.save_widgets_layout.addWidget(self.save_frame)

        self.bottom_widgets_layout = QVBoxLayout()
        self.bottom_widgets_layout.addWidget(self.bottom_frame)

        self.gui_layout.addLayout(self.top_widgets_layout)
        self.gui_layout.addLayout(self.exp_widgets_layout)
        self.gui_layout.addLayout(self.laser_widgets_layout)
        self.gui_layout.addLayout(self.save_widgets_layout)
        self.gui_layout.addLayout(self.bottom_widgets_layout)

        self.setLayout(self.gui_layout)

    def create_params_widget(self, tag, defaults):
        match tag:
            case 'Signal vs Time':
                params = {
                'sampling_rate': {'display_text': 'Sampling Rate: ',
                        'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (10, 1e6), dec = True)},
                'sigvstime_detector': {'display_text': 'Detector: ',
                                        'widget': ComboBox(items = defaults[1])},                
                'dataset': {'display_text': 'Data Set',
                        'widget': QtWidgets.QLineEdit(defaults[2])}}
            case 'CW ODMR':    
                params = {
                        'runs': {'display_text': 'Runs (per pt.): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'freq': {'display_text': 'Center Frequency: ',
                                'widget': SpinBox(value = defaults[2], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'max_sideband_freq': {'display_text': 'Half Frequency Span: ',
                                'widget': SpinBox(value = defaults[3], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'num_pts': {'display_text': '# Frequencies: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[5], suffix = 'W', siPrefix = True)},
                        'timeout': {'display_text': 'Exp. Timeout: ',
                                'widget': SpinBox(value = defaults[6], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'dataset': {'display_text': 'Data Set',
                                'widget': QtWidgets.QLineEdit(defaults[7])}}
            case 'Laser':    
                params = {
                        'laser_power': {'display_text': 'Power (%): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(0, 110), dec = True)},
                        'sideband_freq': {'display_text': 'Sideband Mod. Frequency: ',
                                        'widget': SpinBox(value = defaults[1], suffix = 'Hz', siPrefix = True, bounds = (100, 100e6), dec = True)},
                        'sideband_power': {'display_text': 'Sideband Power: ',
                                        'widget': SpinBox(value = defaults[2], suffix = 'V', siPrefix = True)},
                        'sideband': {'display_text': 'Sideband: ',
                                        'widget': ComboBox(items = defaults[3])},
                        'i_offset': {'display_text': 'I Offset: ',
                                        'widget': SpinBox(value = defaults[4], suffix = 'V', siPrefix = True)},
                        'q_offset': {'display_text': 'Q Offset: ',
                                        'widget': SpinBox(value = defaults[5], suffix = 'V', siPrefix = True)},
                        'detector': {'display_text': 'Detector: ',
                                        'widget': ComboBox(items = defaults[6])}}
            case 'CW Laser':    
                params = {
                        'laser_power': {'display_text': 'Laser Power %: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(0, 110), dec = True)},
                        'probe': {'display_text': 'ODMR MW Probe Time: ',
                                'widget': SpinBox(value = defaults[1], suffix = 's', siPrefix = True, bounds = (10e-9, None))},
                        'sideband_freq': {'display_text': 'Sideband Mod. Frequency: ',
                                'widget': SpinBox(value = defaults[2], suffix = 'Hz', siPrefix = True, bounds = (100, 100e6), dec = True)},
                        'sideband_power': {'display_text': 'Sideband Power: ',
                                        'widget': SpinBox(value = defaults[3], suffix = 'V', siPrefix = True)},
                        'sideband': {'display_text': 'Sideband Choice: ',
                                'widget': ComboBox(items = defaults[4])},
                        'i_offset': {'display_text': 'I Offset: ',
                                        'widget': SpinBox(value = defaults[5], suffix = 'V', siPrefix = True)},
                        'q_offset': {'display_text': 'Q Offset: ',
                                        'widget': SpinBox(value = defaults[6], suffix = 'V', siPrefix = True)},
                        'detector': {'display_text': 'Detector: ',
                                        'widget': ComboBox(items = defaults[7])}}
            
            case 'Digitizer':
                params = {
                        'segment_size': {'display_text': 'Segment Size: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(0, 1e9), dec = True)},
                        'dig_timeout': {'display_text': 'Card Timeout: ',
                                'widget': SpinBox(value = defaults[1], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'sampling_freq': {'display_text': 'Sampling Frequency: ',
                                'widget': SpinBox(value = defaults[2], suffix = 'Hz', siPrefix = True, bounds = (100, 500e6), dec = True)}, 
                        'amplitude': {'display_text': 'Amplitude: ',
                                'widget': SpinBox(value = defaults[3], suffix = 'V', siPrefix = True)},
                        'read_channel': {'display_text': 'Readout Channel: ',
                                'widget': ComboBox(items = defaults[4])},
                        'pretrig_size': {'display_text': '# Pretrigger Samples: ',
                                'widget': SpinBox(value = defaults[5], int = True, bounds=(0, 1024), dec = True)}}

            case 'Pulsed ODMR':    
                params = {
                        'runs': {'display_text': 'Runs (per pt.): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'freq': {'display_text': 'Center Frequency: ',
                                'widget': SpinBox(value = defaults[2], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'max_sideband_freq': {'display_text': 'Half Frequency Span: ',
                                'widget': SpinBox(value = defaults[3], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'num_pts': {'display_text': '# Frequencies: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[5], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[6], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'timeout': {'display_text': 'Exp. Timeout: ',
                                'widget': SpinBox(value = defaults[7], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'dataset': {'display_text': 'Data Set',
                                'widget': QtWidgets.QLineEdit(defaults[8])}}
            case 'Rabi':    
                params = {
                        'runs': {'display_text': 'Runs (per pt.): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start MW Pulse Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'End MW Pulse Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# Pulse Durations: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[5], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[6], suffix = 'W', siPrefix = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': ComboBox(items = defaults[7])},
                        'rabi_type': {'display_text': 'Rabi Type: ',
                                'widget': ComboBox(items = defaults[8])},
                        'timeout': {'display_text': 'Exp. Timeout: ',
                                'widget': SpinBox(value = defaults[9], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'dataset': {'display_text': 'Data Set',
                                'widget': QtWidgets.QLineEdit(defaults[10])}
                    }
            case 'Optical T1':
                params = {
                        'runs': {'display_text': 'Runs (per pt.): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])},
                        'timeout': {'display_text': 'Exp. Timeout: ',
                                'widget': SpinBox(value = defaults[6], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'dataset': {'display_text': 'Data Set',
                                'widget': QtWidgets.QLineEdit(defaults[7])}
                    }
            case 'MW T1':
                params = {
                        'runs': {'display_text': 'Runs (per pt.): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])},
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[6], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[7], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[8], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[9])},
                        'timeout': {'display_text': 'Exp. Timeout: ',
                                'widget': SpinBox(value = defaults[10], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'dataset': {'display_text': 'Data Set',
                                'widget': QtWidgets.QLineEdit(defaults[11])}
                    }
            case 'T2':
                params = {
                        'runs': {'display_text': 'Runs (per pt.): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])},
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[6], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[7], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[8], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[9])},
                        't2_seq': {'display_text': 'Sequence: ',
                                'widget': ComboBox(items = defaults[10])},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults[11], int = True, bounds=(1, None))},
                        'timeout': {'display_text': 'Exp. Timeout: ',
                                'widget': SpinBox(value = defaults[12], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'dataset': {'display_text': 'Data Set',
                                'widget': QtWidgets.QLineEdit(defaults[13])}
                    }
            case 'DQ Relaxation':
                params = {
                        'runs': {'display_text': 'Runs (per pt.): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])},
                        'freq_minus': {'display_text': 'NV Frequency |-1>: ',
                                'widget': SpinBox(value = defaults[6], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power_minus': {'display_text': 'NV MW Power |-1>: ',
                                'widget': SpinBox(value = defaults[7], suffix = 'W', siPrefix = True)},
                        'pi_minus': {'display_text': '\u03C0 Pulse |-1>: ',
                                'widget': SpinBox(value = defaults[8], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'freq_plus': {'display_text': 'NV Frequency |+1>: ',
                                'widget': SpinBox(value = defaults[9], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power_plus': {'display_text': 'NV MW Power |+1>: ',
                                'widget': SpinBox(value = defaults[10], suffix = 'W', siPrefix = True)},
                        'pi_plus': {'display_text': '\u03C0 Pulse |+1>: ',
                                'widget': SpinBox(value = defaults[11], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[12])},
                        'timeout': {'display_text': 'Exp. Timeout: ',
                                'widget': SpinBox(value = defaults[13], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'dataset': {'display_text': 'Data Set',
                                'widget': QtWidgets.QLineEdit(defaults[14])}
                    }
            case 'DEER':
                params = {
                        'runs': {'display_text': 'Runs (per pt.): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start Frequency: ',
                                'widget': SpinBox(value = defaults[2], suffix = 'Hz', siPrefix = True, bounds = (100e3, 750e6), dec = True)},
                        'stop': {'display_text': 'End Frequency: ',
                                'widget': SpinBox(value = defaults[3], suffix = 'Hz', siPrefix = True, bounds = (100e3, 750e6), dec = True)},
                        'num_pts': {'display_text': '# Frequencies: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'tau': {'display_text': 'Free Precession \u03C4: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[6], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = defaults[7], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[8], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[9])},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[10], suffix = 'V', siPrefix = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[11], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'drive_type': {'display_text': 'Dark MW Driving',
                                'widget': ComboBox(items = defaults[12])},
                        'timeout': {'display_text': 'Exp. Timeout: ',
                                'widget': SpinBox(value = defaults[13], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'dataset': {'display_text': 'Data Set',
                                'widget': QtWidgets.QLineEdit(defaults[14])}
                    }
            case 'DEER Rabi':
                params = {
                        'runs': {'display_text': 'Runs (per pt.): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start AWG Pulse Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (3e-9, None), dec = True)},
                        'stop': {'display_text': 'End AWG Pulse Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (3e-9, None), dec = True)},
                        'num_pts': {'display_text': '# Pulse Durations: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'tau': {'display_text': 'Free Precession \u03C4: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[6], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = defaults[7], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[8], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[9])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[10], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[11], suffix = 'V', siPrefix = True)},
                        'timeout': {'display_text': 'Exp. Timeout: ',
                                'widget': SpinBox(value = defaults[12], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'dataset': {'display_text': 'Data Set',
                                'widget': QtWidgets.QLineEdit(defaults[13])}
                    }
            case 'DEER FID':
                params = {
                        'runs': {'display_text': 'Runs (per pt.): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4 Points: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])},
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[6], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = defaults[7], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[8], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[9])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[10], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[11], suffix = 'V', siPrefix = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[12], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults[13], int = True, bounds=(1, None))},
                        'timeout': {'display_text': 'Exp. Timeout: ',
                                'widget': SpinBox(value = defaults[14], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'dataset': {'display_text': 'Data Set',
                                'widget': QtWidgets.QLineEdit(defaults[15])}
                    }  
            case 'DEER FID Continuous Drive':
                params = {
                        'runs': {'display_text': 'Runs (per pt.): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4 Points: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])},
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[6], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = defaults[7], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[8], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[9])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[10], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[11], suffix = 'V', siPrefix = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[12], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_cd_power': {'display_text': 'Continuous Drive (AWG) Power: ',
                                'widget': SpinBox(value = defaults[13], suffix = 'V', siPrefix = True)},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults[14], int = True, bounds=(1, None))},
                        'timeout': {'display_text': 'Exp. Timeout: ',
                                'widget': SpinBox(value = defaults[15], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'dataset': {'display_text': 'Data Set',
                                'widget': QtWidgets.QLineEdit(defaults[16])}
                    }               
            case 'DEER Correlation Rabi':
                params = {
                        'runs': {'display_text': 'Runs (per pt.): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start AWG Corr. Pulse Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (3e-9, None), dec = True)},
                        'stop': {'display_text': 'End AWG Corr. Pulse Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (3e-9, None), dec = True)},
                        'num_pts': {'display_text': '# Pulse Durations: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'tau': {'display_text': 'Free Precession \u03C4: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        't_corr': {'display_text': 'Correlation Time \u03C4_c: ',
                                'widget': SpinBox(value = defaults[6], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[7], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = defaults[8], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[9], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[10])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[11], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[12], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[13], suffix = 'V', siPrefix = True)},
                        'timeout': {'display_text': 'Exp. Timeout: ',
                                'widget': SpinBox(value = defaults[14], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'dataset': {'display_text': 'Data Set',
                                'widget': QtWidgets.QLineEdit(defaults[15])}
                    }
            case 'DEER T1':
                params = {
                        'runs': {'display_text': 'Runs (per pt.): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4_corr Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4_corr Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4 Points: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])},
                        'tau': {'display_text': 'Free Precession \u03C4: ',
                                'widget': SpinBox(value = defaults[6], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[7], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = defaults[8], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[9], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[10])},
                        # 't2_seq': {'display_text': 'Sequence: ',
                        #         'widget': ComboBox(items = defaults[9])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[11], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[12], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[13], suffix = 'V', siPrefix = True)},
                        # 'n': {'display_text': '# Seqs. (n): ',
                        #         'widget': SpinBox(value = defaults[14], int = True, bounds=(1, None))},
                        'timeout': {'display_text': 'Exp. Timeout: ',
                                'widget': SpinBox(value = defaults[14], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'dataset': {'display_text': 'Data Set',
                                'widget': QtWidgets.QLineEdit(defaults[15])}
                    }
            case 'DEER T2':
                params = {
                        'runs': {'display_text': 'Runs (per pt.): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4_corr Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4_corr Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4 Points: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])},
                        'tau': {'display_text': 'Free Precession \u03C4: ',
                                'widget': SpinBox(value = defaults[6], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[7], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = defaults[8], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[9], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[10])},
                        # 't2_seq': {'display_text': 'Sequence: ',
                        #         'widget': ComboBox(items = defaults[9])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[11], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[12], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[13], suffix = 'V', siPrefix = True)},
                        # 'n': {'display_text': '# Seqs. (n): ',
                        #         'widget': SpinBox(value = defaults[14], int = True, bounds=(1, None))},
                        'timeout': {'display_text': 'Exp. Timeout: ',
                                'widget': SpinBox(value = defaults[14], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'dataset': {'display_text': 'Data Set',
                                'widget': QtWidgets.QLineEdit(defaults[15])}
                    }
            case 'NMR: Correlation Spectroscopy':    
                params = {
                'runs': {'display_text': 'Runs (per pt.): ',
                        'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                'iters': {'display_text': '# Experiment Iterations: ',
                        'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                'start': {'display_text': 'Start \u03C4 Time: ',
                        'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                'stop': {'display_text': 'Stop \u03C4 Time: ',
                        'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                'num_pts': {'display_text': '# Frequencies: ',
                        'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                'tau': {'display_text': 'Free Precession Interval (\u03C4): ',
                        'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                'freq': {'display_text': 'NV Frequency: ',
                        'widget': SpinBox(value = defaults[6], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                'rf_power': {'display_text': 'NV MW Power: ',
                        'widget': SpinBox(value = defaults[7], suffix = 'W', siPrefix = True)},
                'pi': {'display_text': '\u03C0 Pulse: ',
                        'widget': SpinBox(value = defaults[8], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                'pulse_axis': {'display_text': 'Pulse Axis',
                        'widget': QtWidgets.QLineEdit(defaults[9])},
                # 'nmr_seq': {'display_text': 'Sequence: ',
                #         'widget': ComboBox(items = defaults[10])},
                'n': {'display_text': '# Seqs. (n): ',
                        'widget': SpinBox(value = defaults[10], int = True, bounds=(1, None))},
                'timeout': {'display_text': 'Exp. Timeout: ',
                        'widget': SpinBox(value = defaults[11], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                'dataset': {'display_text': 'Data Set',
                        'widget': QtWidgets.QLineEdit(defaults[12])}
            }
            case 'NMR: CASR':
                params = {
                'runs': {'display_text': 'Runs (avgs. per iteration): ',
                        'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                'iters': {'display_text': '# Experiment Iterations: ',
                        'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                'num_pts': {'display_text': 'n_sr (# synch. readout pts.): ',
                        'widget': SpinBox(value = defaults[2], int = True, bounds=(1, None), dec = True)},
                'tau': {'display_text': 'Free Precession Interval (\u03C4): ',
                        'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                'freq': {'display_text': 'NV Frequency: ',
                        'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                'rf_power': {'display_text': 'NV MW Power: ',
                        'widget': SpinBox(value = defaults[5], suffix = 'W', siPrefix = True)},
                'pi': {'display_text': '\u03C0 Pulse: ',
                        'widget': SpinBox(value = defaults[6], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                'n': {'display_text': 'XY8-N (# \u03C0 pulses): ',
                        'widget': SpinBox(value = defaults[7], int = True, bounds=(1, None))},
                'timeout': {'display_text': 'Exp. Timeout: ',
                        'widget': SpinBox(value = defaults[8], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                'dataset': {'display_text': 'Data Set',
                        'widget': QtWidgets.QLineEdit(defaults[9])}
            }
    
        return params

    def check_queue_from_exp(self):
        # queue checker to control progress bar display
        while not self.queue_from_exp.empty(): #if there is something in the queue
            queueText = self.queue_from_exp.get_nowait() #get it

            self.progress_bar.setValue(int(queueText[0]))
            
            if queueText[0] == '100':
                self.status.setStyleSheet("color: black; background-color: limegreen; border: 4px solid black;")
                self.status.setText(f"{self.experiments.currentText()} scan complete.")
                self.experiments.setEnabled(True)
            else:
                self.status.setStyleSheet("color: black; background-color: gold; border: 4px solid black;")
                self.status.setText(f"{self.experiments.currentText()} scan in progress... ({round(float(queueText[1]),2)} s/it)")
                self.experiments.setEnabled(False)
            # elif queueText[:9]=='SAVE_REQ:': #if it's a save request, save data in new proc we'll track
            #     saveType, self.datasetName, self.nameForAutosave = queueText.split(':', 1)[1].split(' ')[::2]

            #     self.saveProcs.append(saveInNewProc(datasetName=self.datasetName, expNameForAutosave=self.nameForAutosave, saveType=saveType))
            # else: #you gave an invalid response in the queue, bozo
            #     print(f'Invalid queue_from_exp text:{queueText}')

        self.updateTimer.start(self.QUEUE_CHECK_TIME)

    def auto_save_changed(self):
        if self.auto_save_checkbox.isChecked() == True:
            self.to_save = True
            self.select_dir_button.setEnabled(True)
            self.filename_lineedit.setEnabled(True)
            self.opacity_effects[4].setEnabled(False)
            self.opacity_effects[7].setEnabled(False)
            self.opacity_effects[8].setEnabled(False)
            self.opacity_effects[9].setEnabled(False)
            self.opacity_effects[10].setEnabled(False)

        else:
            self.to_save = False
            self.select_dir_button.setEnabled(False)
            self.filename_lineedit.setEnabled(False)
            self.opacity_effects[4].setEnabled(True)
            self.opacity_effects[7].setEnabled(True)
            self.opacity_effects[8].setEnabled(True)
            self.opacity_effects[9].setEnabled(True)
            self.opacity_effects[10].setEnabled(True)

    def save_params_clicked(self):
        params = dict(**self.params_widget.all_params())
        
        saved_params = list(params.values()) # set saved params for next time the experiment is selected

        # take chosen combobox parameter and place it first in the updated combobox item list
        match self.experiments.currentText():
            case 'Signal vs Time':
                self.sigvstime_detector_opts.insert(0, self.sigvstime_detector_opts.pop(self.sigvstime_detector_opts.index(saved_params[1])))
                saved_params[1] = self.sigvstime_detector_opts
            case 'Rabi':
                self.rabi_axis_opts.insert(0, self.rabi_axis_opts.pop(self.rabi_axis_opts.index(saved_params[7])))
                saved_params[7] = self.rabi_axis_opts   
                self.rabi_device_opts.insert(0, self.rabi_device_opts.pop(self.rabi_device_opts.index(saved_params[8])))
                saved_params[8] = self.rabi_device_opts                
            case 'Optical T1':
                self.opt_t1_array_opts.insert(0, self.opt_t1_array_opts.pop(self.opt_t1_array_opts.index(saved_params[5])))
                saved_params[5] = self.opt_t1_array_opts
            case 'MW T1':
                self.mw_t1_array_opts.insert(0, self.mw_t1_array_opts.pop(self.mw_t1_array_opts.index(saved_params[5])))
                saved_params[5] = self.mw_t1_array_opts
            case 'DQ Relaxation':
                self.dq_array_opts.insert(0, self.dq_array_opts.pop(self.dq_array_opts.index(saved_params[5])))
                saved_params[5] = self.dq_array_opts   
            case 'T2':
                self.t2_array_opts.insert(0, self.t2_array_opts.pop(self.t2_array_opts.index(saved_params[5])))
                saved_params[5] = self.t2_array_opts
                self.t2_seq_opts.insert(0, self.t2_seq_opts.pop(self.t2_seq_opts.index(saved_params[10])))
                saved_params[10] = self.t2_seq_opts
            case 'DEER':
                self.deer_drive_opts.insert(0, self.deer_drive_opts.pop(self.deer_drive_opts.index(saved_params[12])))
                saved_params[12] = self.deer_drive_opts
            case 'DEER FID':
                self.fid_array_opts.insert(0, self.fid_array_opts.pop(self.fid_array_opts.index(saved_params[5])))
                saved_params[5] = self.fid_array_opts
            case 'DEER FID Continuous Drive':
                self.fid_cd_array_opts.insert(0, self.fid_cd_array_opts.pop(self.fid_cd_array_opts.index(saved_params[5])))
                saved_params[5] = self.fid_cd_array_opts
            case 'DEER T1':
                self.corr_t1_array_opts.insert(0, self.corr_t1_array_opts.pop(self.corr_t1_array_opts.index(saved_params[5])))
                saved_params[5] = self.corr_t1_array_opts
        #     case 'NMR':
        #         self.nmr_seq_opts.insert(0, self.nmr_seq_opts.pop(self.nmr_seq_opts.index(saved_params[10])))
        #         saved_params[10] = self.nmr_seq_opts

        self.exp_dict[self.experiments.currentText()][1] = saved_params

    def select_directory(self):
        response = QFileDialog.getExistingDirectory(self, caption = "Select a folder")
        
        self.chosen_dir.setText(str(response) + "/")

    def get_lineedit_val(self, lineedit):
        return lineedit.text()
    
    def get_combobox_val(self, combobox):
        return str(combobox.value())
    
    def exp_selector(self):
        # reset params widgets each time new experiment selected
        self.status.setStyleSheet("color: black; background-color: #00b8ff; border: 4px solid black;")
        self.status.setText("Select parameters and press 'Run' to begin experiment.")
        self.progress_bar.setValue(0)

        self.exp_label.hide()
        self.params_widget.hide()
        self.save_params.hide()
        self.laser_label.hide()
        self.laser_params_widget.hide() 
        
        try:
        #     print("EXPERIMENT SELECTED: ", self.experiments.currentText())
            # print("DEFAULTS: ", self.exp_dict[self.experiments.currentText()][1])
            self.params_widget = ParamsWidget(self.create_params_widget(self.experiments.currentText(), self.exp_dict[self.experiments.currentText()][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})
        except KeyError: 
            # if "Select dropdown option" is selected, populate GUI with disabled ODMR widgets as filler
            self.params_widget = ParamsWidget(self.create_params_widget('CW ODMR', self.exp_dict['CW ODMR'][1]))
            self.params_widget.setGraphicsEffect(self.opacity_effects[1]) # reset opacity effects
            self.laser_params_widget = ParamsWidget(self.create_params_widget('CW Laser', self.exp_dict['CW Laser'][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})
            self.laser_params_widget.setGraphicsEffect(self.opacity_effects[4]) # reset opacity effects
            self.dig_params_widget = ParamsWidget(self.create_params_widget('Digitizer', self.exp_dict['Digitizer'][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})
            self.dig_params_widget.setGraphicsEffect(self.opacity_effects[6]) # reset opacity effects
            for i in range(7):
                self.opacity_effects[i].setEnabled(True)

            self.params_widget.setEnabled(False)
            self.save_params.setEnabled(False)
            self.laser_params_widget.setEnabled(False)
            self.dig_params_widget.setEnabled(False)
        else:
            self.params_widget.setEnabled(True)
            self.save_params.setEnabled(True)
            self.laser_params_widget.setEnabled(True)
            self.dig_params_widget.setEnabled(True)
            for i in range(7):
                self.opacity_effects[i].setEnabled(False)

            if self.experiments.currentText() == 'CW ODMR':
                self.laser_params_widget = ParamsWidget(self.create_params_widget('CW Laser', self.exp_dict['CW Laser'][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})
                self.dig_params_widget = ParamsWidget(self.create_params_widget('Digitizer', self.exp_dict['Digitizer'][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})
                self.extra_kwarg_params['odmr_type'] = "CW"
            else:
                if self.experiments.currentText() == 'Pulsed ODMR':
                    self.extra_kwarg_params['odmr_type'] = "Pulsed"

                self.laser_params_widget = ParamsWidget(self.create_params_widget('Laser', self.exp_dict['Laser'][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})
                self.dig_params_widget = ParamsWidget(self.create_params_widget('Digitizer', self.exp_dict['Digitizer'][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})

        finally:
            self.exp_params_layout.insertWidget(0, self.exp_label)
            self.exp_params_layout.insertWidget(1, self.params_widget)
            self.exp_params_layout.insertWidget(2, self.save_params)
            self.laser_params_layout.addWidget(self.laser_label,1,1,1,1)
            self.laser_params_layout.addWidget(self.laser_params_widget,2,1,1,1)
            self.laser_params_layout.addWidget(self.dig_label,1,2,1,1)
            self.laser_params_layout.addWidget(self.dig_params_widget,2,2,1,1)

            self.exp_label.show()
            self.save_params.show()
            self.params_widget.show()
            if self.experiments.currentText() != 'Signal vs Time':
                self.laser_label.show()
                self.laser_params_widget.show()
                self.dig_label.show()
                self.dig_params_widget.show()
            else:
                self.laser_label.hide()
                self.laser_params_widget.hide()
                self.dig_label.hide()
                self.dig_params_widget.hide()

#     @pyqtSlot(dict)
#     def retrieve_exp_params(self, params):
#         return params
    
    def run(self):
        """Run the experiment function in a subprocess."""
        
        if self.run_proc.running():
            logging.info(
                'Not starting the experiment process because it is still running.'
            )
            
            return

        # reload the module at runtime in case any changes were made to the code
        reload(nv_experiments_12_5_24)
        
        self.status.setStyleSheet("color: black; background-color: gold; border: 4px solid black;")
        self.status.setText(f"{self.experiments.currentText()} scan in progress...")

        # self.communicator = Communicate()
        # self.communicator_params = self.communicator.speak.connect(self.retrieve_exp_params)

        self.extra_kwarg_params['save'] = self.to_save
        self.extra_kwarg_params['filename'] = self.filename_lineedit.text()
        self.extra_kwarg_params['directory'] = self.chosen_dir.text()
        self.extra_kwarg_params['seq'] = self.experiments.currentText()

        # unpack all keyword arg parameters to send to experiment process
        fun_kwargs = dict(**self.params_widget.all_params(), **self.laser_params_widget.all_params(), **self.dig_params_widget.all_params(), **self.extra_kwarg_params)
        # fun_kwargs = dict(self.communicator_params, **self.params_widget.all_params(), **self.extra_kwarg_params)

        # if self.experiments.currentText() == 'ODMR':
        #     fun_kwargs = dict(**self.params_widget.all_params())
        #     fun_kwargs['odmr_type'] = self.odmr_type
        # else:
        #     fun_kwargs = dict(**self.params_widget.all_params(), **self.laser_params_widget.all_params())

        # call the function in a new process
        self.run_proc.run(
            run_experiment,
            exp_cls = nv_experiments_12_5_24.SpinMeasurements,
            fun_name = self.exp_dict[self.experiments.currentText()][0],
            constructor_args = list(),
            constructor_kwargs = dict(),
            queue_to_exp = self.queue_to_exp,
            queue_from_exp = self.queue_from_exp,
            fun_args = list(),
            fun_kwargs = fun_kwargs,
        )
        
    def stop(self, log: bool = True):
        """Request the experiment subprocess to stop by sending the string :code:`stop`
        to :code:`queue_to_exp`.

        Args:
            log: if True, log when stop is called but the process isn't running.
        """

        self.experiments.setEnabled(True)

        self.status.setStyleSheet("color: black; background-color: red; border: 4px solid black;")
        self.status.setText(f"{self.experiments.currentText()} scan stopped.")
                
        if self.run_proc.running():
        #     self.queue_to_exp.put('stop')
            self.run_proc.kill() # kill process. Commented out previous line as this gave DAQ invalid task errors when used alone.  When both lines uncommented, every other scan wouldn't run.
        else:
            if log:
                logging.info(
                    'Not stopping the experiment process because it is not running.'
                )
        

        with InstrumentManager() as mgr:
            try:
                mgr.daq.close_do_task()
                mgr.daq.close_ao_task()
                mgr.daq.close_ai_task()

            except Exception as e:
                print(e)
            finally:
                mgr.sg.set_rf_toggle(0)
                mgr.sg.set_mod_toggle(0)
                mgr.laser.laser_off()
                mgr.laser_shutter.close_shutter()
                mgr.ps.Pulser.reset()
                mgr.awg.set_disabled()
                mgr.dig.stop_card()
                mgr.dig.reset()


class FlexLinePlotWidgetAllDefaults(FlexLinePlotWidget):
    """Add some default settings to the FlexSinkLinePlotWidget."""
    def __init__(self):
        super().__init__()

        # manually set the XY range
        self.line_plot.plot_item().setXRange(3.0, 4.0)
        self.line_plot.plot_item().setYRange(-3000, 4500)

        # retrieve legend object
        legend = self.line_plot.plot_widget.addLegend()
        # set the legend location
        legend.setOffset((-10, -50))

        self.datasource_lineedit.setText('odmr')


#Bunch of helper fxns for fitter
#Fit Fxns
def customSingleLorentzian(x, x0, FWHM, A, B):
    '''Defines a Lorentzian of x with additional args
        x0 (center), FWHM, A (depth), B (offset)'''
    return( B + A/(1+( (x-x0)/(FWHM/2) )**2) )

def customDoubleLorentzian(x, x1, FWHM1, A1, x2, FWHM2, A2, B):
    '''Defines a Double Lorentzian of x with additional args
        x1 (center), FWHM1, A1 (depth), x2, FWHM2, A2, B (offset)'''
    return( B + A1/(1+( (x-x1)/(FWHM1/2) )**2) + A2/(1+( (x-x2)/(FWHM2/2) )**2) )

def customCosine(t, f, t0, A, B):
    '''Defines a cosine of t with additional args
    f (plain freq), t0 (offset), A (amplitude), B (offset)'''
    return( B + A*np.cos(2*np.pi*f*(t-t0)) )

def customDecayExp(t, T, A, B):
    '''Defines a exp decay of t with additional args
    t0 (time-const), A (amplitude), B (offset)'''
    return( B + A*np.exp(-t/T) )

def customDecayCos(t, f, t0, T, A, B):
    '''Defines a decaying cosine of t with additional args
    f (plain freq), t0 (offset), T (decay time-const), A (amplitude), B (offset)'''
    return( B + A*np.exp(-t/T)*np.cos(2*np.pi*f*(t-t0)) )

def customGaussian(x, x0, sigma, A, B):
    '''Defines a Gaussian function of x with parameters x0 (x of peak), sigma (std dev), A (vertical stretch factor), and B (y-offset)'''
    return( B + A*np.exp(-(x-x0)**2/(2*sigma**2)) )



#Val accessor fxns
def getQCheckBoxVal(QCheckBoxObj):
    return(QCheckBoxObj.isChecked())

def getComboBoxVal(ComboBoxObj):
    return(ComboBoxObj.currentText())

def getLineEditVal(lineEditObj):
    return(lineEditObj.text())

class FitterWidget(QtWidgets.QWidget): 

    FIT_FXNS = {'Single Lorentzian': customSingleLorentzian,
                'Double Lorentzian': customDoubleLorentzian, 
                'Cos': customCosine,
                'Decay Exp': customDecayExp,
                'Decay Cos': customDecayCos,
                'Gaussian': customGaussian,}

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Fitter')

        #Set-up main layout with two parts: 
        #Left box for taking dataset name, ifDiffMeas, and boxes for selecting IV/DV for plotting
        self.fullLayout = QtWidgets.QHBoxLayout()
        self.leftBox = QtWidgets.QVBoxLayout()
        self.fullLayout.addLayout(self.leftBox)
        self.rightBox = QtWidgets.QVBoxLayout()
        self.fullLayout.addLayout(self.rightBox)

        self.datasetNameWidget =  ParamsWidget({ #button to specify which data to pull from DataServ
                'datasetName': {
                    'display_text': 'Dataset Name',
                    'widget': QtWidgets.QLineEdit('CW_ODMR'),
                },
                })  #self.datasetNameBox.textChanged.connect(self._updateDataParams)
        self.pullDatasetButton = QtWidgets.QPushButton('Pull') #Pressing will pull specified dataset from DataServ (which takes a couple seconds)
        self.pullDatasetButton.clicked.connect(self._pullData)
        self.leftBox.addWidget(self.datasetNameWidget)
        self.leftBox.addWidget(self.pullDatasetButton)

        self.ifDiffMeas = QtWidgets.QCheckBox('Differential Measurement?')
        self.ifDiffMeas.stateChanged.connect(self._updateDatasetParams) #will draw 2 boxes for IV,DV for normal meas, and 3 for IV,DV1-DV2 for diff meas
        self.leftBox.addWidget(self.ifDiffMeas)


        #Now for right box for fitter params
        self.fitFxnComboBox = ComboBox(items=self.FIT_FXNS) #combo box to select fit fxn
        self.fitFxnComboBox.currentTextChanged.connect(self._updateFitterParams)
        self.fitFxnWidget = ParamsWidget({
                'fitFxn': {
                    'display_text': 'Fit Fxn',
                    'widget': self.fitFxnComboBox,
                },

            }, get_param_value_funs={ComboBox: getComboBoxVal}) #QLineEdit: getLineEditVal,  QCheckBox: getQCheckBoxVal} )

        self.rightBox.addWidget(self.fitFxnWidget)

        self.fitDataButton = QtWidgets.QPushButton('Fit') #Button to run fitter
        self.fitDataButton.clicked.connect(self._fit)
        self.plotDataButton = QtWidgets.QPushButton('Plot') #Button to plot a fit
        self.plotDataButton.clicked.connect(self._fitThenPlot)
        self._updateFitterParams() #takes care of creating a params widget to get guess params for fitter, also adds fit+plot buttons to GUI

        #make layout active
        self.setLayout(self.fullLayout)

        '''self.datasetParams = ParamsWidget({
                'datasetName': {
                    'display_text': 'Dataset Name',
                    'widget': self.datasetNameBox,
                },

                'diffMeas': {
                    'display_text': 'Differential Measurement?',
                    'widget': self.diffMeas,
                },
                
                'fitFxn': {
                    'display_text': 'Fit Fxn',
                    'widget': self.fitFxnComboBox,
                },

            }, 
            get_param_value_funs={QLineEdit: getLineEditVal, ComboBox: getComboBoxVal, QCheckBox: getQCheckBoxVal}
        )'''


    def _pullData(self):
        try: #jank, but good enough so this works for init and updating #TODO: Dejank
            self.ds = DataSink(self.datasetNameWidget.datasetName) #create a connection to this DS so we can use it later
            self.ds.__enter__()
            self.ds.pop(0.1) #makes sure we have data (or errors out after waiting for 100ms)
            self._updateDatasetParams() #update dataset params widget for new dataset
        except Exception as e:
            print(e)


    def _updateDatasetParams(self):

        datasets = list(self.ds.datasets.keys())

        if hasattr(self, 'datasets'): #if we have previously assigned a set of datasets to these boxes
            if  self.datasets != datasets or self.diffMeas != self.ifDiffMeas.isChecked(): #if the keys changed, redraw the box, otherwise do nothing because we're already set
                self.datasets = datasets
                self.diffMeas = self.ifDiffMeas.isChecked()
                self.datasetParams.hide()
                self._drawDatasetParams()
        else:
            self.datasets = datasets
            self.diffMeas = self.ifDiffMeas.isChecked()
            self._drawDatasetParams()   
        #hasattr(self, 'datasetParams')
       

    def _drawDatasetParams(self):
        
        #print('Debug:', self.ifDiffMeas.isChecked()) #DEBUG
        if self.ifDiffMeas.isChecked(): #taking a diff meas
            self.datasetParams = ParamsWidget({
                'IV': {
                    'display_text': 'IV',
                    'widget': ComboBox(items=self.datasets),
                },

                'DV1': {
                    'display_text': 'DV1',
                    'widget': ComboBox(items=self.datasets),
                },
                
                'DV2': {
                    'display_text': '-DV2',
                    'widget': ComboBox(items=self.datasets),
                },
            }, get_param_value_funs={ComboBox: getComboBoxVal})
        else: #so non-diff measurement
            self.datasetParams = ParamsWidget({
                'IV': {
                    'display_text': 'IV',
                    'widget': ComboBox(items=self.datasets),
                },

                'DV': {
                    'display_text': 'DV',
                    'widget': ComboBox(items=self.datasets),
                },
            }, get_param_value_funs={ComboBox: getComboBoxVal})
        
        self.leftBox.addWidget(self.datasetParams)
        self.leftBox.update()
        


    def _updateFitterParams(self):
        try: #kinda jank, but good enough so this works for init and updating #TODO: Dejank
            self.fitterParams.hide()
            #self.fitDataButton.hide()
            #self.plotDataButton.hide()
        except AttributeError:
            pass

        paramNames = str(signature(self.FIT_FXNS[self.fitFxnWidget.fitFxn]))[1:-1].split(',')[1:] #take the arg names as a string and make into a list ignoring the first arg (which is for data)
        self.fitterParams = ParamsWidget(
            dict([[paramName, {'display_text': paramName, 'widget': SpinBox(value=1, dec=True)}, ] for paramName in paramNames])
            )
        
        self.rightBox.addWidget(self.fitterParams)
        self.rightBox.addWidget(self.fitDataButton) #redoing this here because I want this order, and I don't know how to do this properly. #TODO: Dejank
        self.rightBox.addWidget(self.plotDataButton) #redoing this here because I want this order, and I don't know how to do this properly. #TODO: Dejank
        self.rightBox.update()


    def _fit(self):
        fitFxn = self.FIT_FXNS[self.fitFxnWidget.fitFxn]
        if self.ifDiffMeas.isChecked():
            xs = self.ds.datasets[self.datasetParams.IV]
            #print('debug1')
            xs, y1s = self._unpackData(self.ds.datasets[self.datasetParams.IV], self.ds.datasets[self.datasetParams.DV1])
            y2s = self._unpackData(self.ds.datasets[self.datasetParams.IV], self.ds.datasets[self.datasetParams.DV2])[1]
            #print('debug2')
            ys = [y1s[i]-y2s[i] for i in range(len(xs))]
            yMeans = [np.mean(y) for y in ys]
        else: 
            xs, ys = self._unpackData(self.ds.datasets[self.datasetParams.IV], self.ds.datasets[self.datasetParams.DV])
            yMeans = [np.mean(y) for y in ys]

        #print(list(self.fitterParams.all_params().values())) #DEBUG
        guessParamNames = list(self.fitterParams.all_params().keys()) #get a dict of param name, param val pairs, so take only the values to pass to the fitter (hopefully order is correct)
        guessParams = list(self.fitterParams.all_params().values()) #get a dict of param name, param val pairs, so take only the values to pass to the fitter (hopefully order is correct)

        #might want to try-catch this in case optimization fails, but hopefully everything breaks for now. #TODO: Implement this
        optParams, pcov = curve_fit(fitFxn, xs, yMeans, guessParams)
        
        print(guessParamNames, '\nGuess Params:', guessParams, '\nFit Params:', optParams)
        if self.fitFxnWidget.fitFxn in ['Cos', 'Decay Cos']:
            f, t0 = optParams[0], optParams[1] 
            print(f'Pi/2-Time: {1/(4*f)+t0}, Pi-Time: {1/(2*f)+t0}, 3Pi/2-Time: {3/(4*f)+t0}')
        return(optParams, lambda x: fitFxn(x, *optParams), xs, yMeans) #should unpack params, then pass list of optimal params+lambda-defined fit fxn 


    def _unpackData(self, xs, ys):
        xs = obtain(xs)
        ys = obtain(ys)
        sortedXs = np.sort(xs)
        if isinstance(ys, dict): #data is in a dict of lists, so need xs to unpack data
            sortedData = [np.array(ys[x], dtype=float).flatten() for x in sortedXs]
        else: #so data is in list of lists of data
            sortedData = [np.array(y, dtype=float).flatten() for y in ys]
        return(sortedXs, sortedData)


    def _plotDataAndFit(self, xs, ys, fitFxn):
        try: #kinda jank, but good enough so this works for init and updating #TODO: Dejank
            self.plotWidget.hide()
        except AttributeError:
            pass

        self.plotWidget = PlotWidget() #NFI if this will work
        self.plotWidget.plot(xs, ys, label='data')
        denseXs = np.linspace(min(xs), max(xs), len(xs)*10 ) #in-fill xs by a factor of 10x
        self.plotWidget.plot(denseXs, [fitFxn(x) for x in denseXs], label='fit')
        #self.plotWidget.legend()
        self.fullLayout.addWidget(self.plotWidget)
        self.fullLayout.update()


    def _fitThenPlot(self):
        fitParams, fitFxn, xs, ys = self._fit()
        self._plotDataAndFit(xs, ys, fitFxn)
