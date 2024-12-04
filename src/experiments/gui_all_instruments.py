"""
Copyright (c) 2021, Jacob Feder
All rights reserved.

This work is licensed under the terms of the 3-Clause BSD license.
For a copy, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

"""
Magnet Mount Motion and Magnet Alignment GUI
Evan Villafranca - 2022
"""

from functools import partial
from importlib import reload

import numpy as np
import nnmr_magnet_motion as mag
from b_field_calibrator import BFieldCalibrate

from nspyre import DataSink
from nspyre import LinePlotWidget
from nspyre import ParamsWidget
from nspyre import ProcessRunner
from nspyre import InstrumentManager

from pyqtgraph import SpinBox, ComboBox

from multiprocessing import Queue
from nspyre import SaveWidget
from PyQt6.QtWidgets import QLabel, QPushButton, QCheckBox, QComboBox, QLineEdit, QInputDialog, QRadioButton, QSlider, QDoubleSpinBox
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout
from PyQt6.QtWidgets import QWidget, QGraphicsOpacityEffect
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, QTimer

import logging

logger = logging.getLogger(__name__)

class InstWidget(QWidget):
    """
    Qt widget subclass that generates an interface for operating magnet mount.
    """

    QUEUE_CHECK_TIME = 50 # ms

    def __init__(self):
        super().__init__()

        self.setWindowTitle('NanoNMR Magnet Alignment')

        self.updateTimer = QTimer() #create a timer that will try to update that widget with messages from the from_exp_queue
        self.updateTimer.timeout.connect(lambda: self.check_queue_from_mag())
        self.updateTimer.start(self.QUEUE_CHECK_TIME)

        # self.calibrator = BFieldCalibrate('B_field_cal.xlsx')

        # current stage positions
        self.curr_r = 0 
        self.curr_theta = 0
        self.curr_phi = 0
        self.r_offset = 0

        self.r_min = 0
        self.r_max = 100
        self.theta_min = -50
        self.theta_max = 50
        self.phi_min = 0
        self.phi_max = 115

        self.white_border = "border: 1px solid white"
        self.font = "Arial"
        
        self.q = Queue()
        
        # magnet stage widgets
        self.magnet_label = QLabel("Magnet Stage Control")
        self.magnet_label.setFixedHeight(30)
        # self.magnet_label.setStyleSheet("color: white; background-color:  #2b2b2b; border: 1px solid black;")
        self.magnet_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.magnet_label.setFont(QFont(self.font, 20))

        self.all_disable_button = QPushButton("Enable All")
        self.all_disable_proc = ProcessRunner()
        self.all_disable_button.clicked.connect(self.all_disable_clicked)
        
        self.all_standby_button = QPushButton("Standby")
        self.all_standby_proc = ProcessRunner()
        self.all_standby_button.clicked.connect(self.all_standby_clicked)

        self.status_label = QLabel("Magnets status here")
        self.status_label.setStyleSheet("color: white; background-color: black; border: 4px solid black;")
        self.status_label.setFixedHeight(40)

        # self.all_move_checkbox = QCheckBox("Move")
        # self.all_move_checkbox.setChecked(False)
        # self.all_move_checkbox.setStyleSheet("QCheckBox::indicator:hover" "{""background-color : yellow;""}"
        #                                     "QCheckBox::indicator:pressed" "{""background-color : lightgreen;""}")
        # self.all_move_checkbox.stateChanged.connect(lambda:self.move_button_checked(self.all_move_checkbox))
        
        # self.all_move_types = QComboBox()
        # self.all_move_types.addItems(["", "Absolute", "B-Field"])
        # self.all_move_types.currentTextChanged.connect(self.all_move_types_changed)
        
        # self.all_positions = QLabel("")
        
        # self.clear_all_positions = QPushButton("Clear Positions")
        # self.clear_all_positions.clicked.connect(lambda:self.clear_positions_clicked())
        
        # self.all_execute_move_button = QPushButton("Execute Move")
        # self.all_execute_move_button_proc = ProcessRunner()
        # self.all_execute_move_button.clicked.connect(lambda:self.all_move_clicked())

        # self.all_move_types.setEnabled(False)
        # self.all_positions.setEnabled(False)
        # self.clear_all_positions.setEnabled(False)
        # self.all_execute_move_button.setEnabled(False)

        # self.all_move_types.setGraphicsEffect(self.all_opacity_effects[0])
        # self.all_positions.setGraphicsEffect(self.all_opacity_effects[1])
        # self.clear_all_positions.setGraphicsEffect(self.all_opacity_effects[2])
        # self.all_execute_move_button.setGraphicsEffect(self.all_opacity_effects[3])

        # self.all_stop_button = QPushButton("Stop All")
        # self.all_stop_proc = ProcessRunner()
        # self.all_stop_button.clicked.connect(self.all_stop_button_clicked)

        self.init_r_widgets()
        self.init_polar_widgets()
        self.init_azi_widgets()
        self.init_b_field_calc_widgets()
        self.init_sg396_widgets()
        self.init_laser_widgets()
        self.init_flipper_widgets()
        self.init_pmt_shutter_widgets()

        # self.obtain_current_positions() # update stage positions on GUI startup 
        self.layouts()

        self.park_all()

    '''
    CREATE WIDGETS
    '''
    def init_r_widgets(self):
        '''
        r stage widgets
        '''
        self.r_opacity_effects = []
        for i in range(4):
            self.r_opacity_effects.append(QGraphicsOpacityEffect())
            self.r_opacity_effects[i].setOpacity(0.3)

        self.r_label = QLabel("R")
        self.r_label.setFixedHeight(30)
        self.r_label.setFont(QFont(self.font, 20))
        # self.r_pos_label = QLabel(f"r position = {self.curr_r} mm (mag. sep. = {2*self.curr_r + self.r_offset} mm)")
        # self.r_pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.r_pos_label.setStyleSheet(self.white_border)

        # self.r_park_button = QPushButton("Unpark")
        # self.r_park_button_proc = ProcessRunner()
        # self.r_park_button.clicked.connect(lambda:self.park_button_clicked('zaber'))
        self.r_move_checkbox = QCheckBox("R Move Type: ")
        self.r_move_checkbox.setChecked(False)
        self.r_move_checkbox.setStyleSheet("QCheckBox::indicator:hover" "{""background-color : yellow;""}"
                                            "QCheckBox::indicator:pressed" "{""background-color : lightgreen;""}")
        self.r_move_checkbox.stateChanged.connect(lambda:self.move_button_checked(self.r_move_checkbox))
        
        self.r_move_types = QComboBox()
        self.r_move_types.addItems(["Absolute", "Relative", "B-Field"])
        self.r_move_types.currentIndexChanged.connect(lambda: self.r_move_type_changed())
        self.r_move_option = "Absolute"
        self.r_move_position = QLineEdit()       
        self.r_move_position_units = QLabel("")     
        self.r_execute_move_button = QPushButton("Execute Move")
        self.r_execute_move_button_proc = ProcessRunner()
        self.r_execute_move_button.clicked.connect(lambda:self.single_move_clicked('zaber'))

        self.r_move_types.setEnabled(False)
        self.r_move_position.setEnabled(False)
        self.r_move_position_units.setEnabled(False)
        self.r_execute_move_button.setEnabled(False)

        self.r_move_types.setGraphicsEffect(self.r_opacity_effects[0])
        self.r_move_position.setGraphicsEffect(self.r_opacity_effects[1])
        self.r_move_position_units.setGraphicsEffect(self.r_opacity_effects[2])
        self.r_execute_move_button.setGraphicsEffect(self.r_opacity_effects[3])

        self.r_stop_button = QPushButton("Stop Motion")
        self.r_stop_button_proc = ProcessRunner()
        self.r_stop_button.clicked.connect(lambda:self.stop_button_clicked('zaber'))
        # self.r_home_button = QPushButton("Home")
        # self.r_home_button_proc = ProcessRunner()
        # self.r_home_button.clicked.connect(lambda:self.home_button_clicked('zaber'))

    def init_polar_widgets(self):
        '''
        polar stage widgets
        '''
        self.polar_opacity_effects = []
        for i in range(4):
            self.polar_opacity_effects.append(QGraphicsOpacityEffect())
            self.polar_opacity_effects[i].setOpacity(0.3)

        self.polar_label = QLabel("\u03B8")    
        self.polar_label.setFixedHeight(30)
        self.polar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.polar_label.setFont(QFont(self.font, 20))    
        # self.polar_pos_label = QLabel("\u03B8 position = ")
        # self.polar_pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.polar_pos_label.setStyleSheet(self.white_border)

        # self.polar_enable_button = QPushButton("Enable")
        # self.polar_enable_button_proc = ProcessRunner()
        # self.polar_enable_button.clicked.connect(lambda:self.park_button_clicked('thor_polar'))
        self.polar_move_checkbox = QCheckBox("\u03B8 Move Type: ")
        self.polar_move_checkbox.setChecked(False)
        self.polar_move_checkbox.setStyleSheet("QCheckBox::indicator:hover" "{""background-color : yellow;""}"
                                            "QCheckBox::indicator:pressed" "{""background-color : lightgreen;""}")
        self.polar_move_checkbox.stateChanged.connect(lambda:self.move_button_checked(self.polar_move_checkbox))
        
        self.polar_move_types = QComboBox()
        self.polar_move_types.addItems(["Absolute", "Jog"])
        self.polar_move_position = QLineEdit()
        self.polar_move_position_units = QLabel("")
        self.polar_execute_move_button = QPushButton("Execute Move")
        self.polar_execute_move_button_proc = ProcessRunner()
        self.polar_execute_move_button.clicked.connect(lambda:self.single_move_clicked('thor_polar'))
        
        self.polar_move_types.setEnabled(False)
        self.polar_move_position.setEnabled(False)
        self.polar_move_position_units.setEnabled(False)
        self.polar_execute_move_button.setEnabled(False)

        self.polar_move_types.setGraphicsEffect(self.polar_opacity_effects[0])
        self.polar_move_position.setGraphicsEffect(self.polar_opacity_effects[1])
        self.polar_move_position_units.setGraphicsEffect(self.polar_opacity_effects[2])
        self.polar_execute_move_button.setGraphicsEffect(self.polar_opacity_effects[3])

        self.polar_stop_button = QPushButton("Stop Motion")
        self.polar_stop_button_proc = ProcessRunner()
        self.polar_stop_button.clicked.connect(lambda:self.stop_button_clicked('thor_polar'))
        # self.polar_home_button = QPushButton("Home")
        # self.polar_home_button_proc = ProcessRunner()
        # self.polar_home_button.clicked.connect(lambda:self.home_button_clicked('thor_polar'))

    def init_azi_widgets(self):
        '''
        azimuthal stage widgets
        '''
        self.azi_opacity_effects = []
        for i in range(4):
            self.azi_opacity_effects.append(QGraphicsOpacityEffect())
            self.azi_opacity_effects[i].setOpacity(0.3)

        self.azi_label = QLabel("\u03C6")
        self.azi_label.setFixedHeight(30)
        self.azi_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.azi_label.setFont(QFont(self.font, 20))
        # self.azi_pos_label = QLabel("\u03C6 position = ")
        # self.azi_pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.azi_pos_label.setStyleSheet(self.white_border)

        # self.azi_enable_button = QPushButton("Enable")
        # self.azi_enable_button_proc = ProcessRunner()
        # self.azi_enable_button.clicked.connect(lambda:self.park_button_clicked('thor_azi'))
        self.azi_move_checkbox = QCheckBox("\u03C6 Move Type: ")
        self.azi_move_checkbox.setChecked(False)
        self.azi_move_checkbox.setStyleSheet("QCheckBox::indicator:hover" "{""background-color : yellow;""}"
                                            "QCheckBox::indicator:pressed" "{""background-color : lightgreen;""}")
        self.azi_move_checkbox.stateChanged.connect(lambda:self.move_button_checked(self.azi_move_checkbox))
        
        self.azi_move_types = QComboBox()
        self.azi_move_types.addItems(["Absolute", "Jog"])
        self.azi_move_position = QLineEdit()
        self.azi_move_position_units = QLabel("")    
        self.azi_execute_move_button = QPushButton("Execute Move")
        self.azi_execute_move_button_proc = ProcessRunner()
        self.azi_execute_move_button.clicked.connect(lambda:self.single_move_clicked('thor_azi'))

        self.azi_move_types.setEnabled(False)
        self.azi_move_position.setEnabled(False)
        self.azi_move_position_units.setEnabled(False)
        self.azi_execute_move_button.setEnabled(False)

        self.azi_move_types.setGraphicsEffect(self.azi_opacity_effects[0])
        self.azi_move_position.setGraphicsEffect(self.azi_opacity_effects[1])
        self.azi_move_position_units.setGraphicsEffect(self.azi_opacity_effects[2])
        self.azi_execute_move_button.setGraphicsEffect(self.azi_opacity_effects[3])

        self.azi_stop_button = QPushButton("Stop Motion")
        self.azi_stop_button_proc = ProcessRunner()
        self.azi_stop_button.clicked.connect(lambda:self.stop_button_clicked('thor_azi'))
        # self.azi_home_button = QPushButton("Home")
        # self.azi_home_button_proc = ProcessRunner()
        # self.azi_home_button.clicked.connect(lambda:self.home_button_clicked('thor_azi'))

    def init_b_field_calc_widgets(self):
        '''
        B-field calculation widgets
        '''
        self.bfield_calc_label = QLabel("B-Field Resonance Calculator")


    def get_combobox_val(self, combobox):
        return str(combobox.value())
    

    def init_sg396_widgets(self):
        # magnet stage widgets
        self.sg396_label = QLabel("Signal Generator Control")
        self.sg396_label.setFixedHeight(30)
        # self.sg396_label.setStyleSheet("color: white; background-color:  #2b2b2b; border: 1px solid black;")
        self.sg396_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sg396_label.setFont(QFont(self.font, 20))

        self.sg396_checkbox = QCheckBox("Enable SG396 output")
        self.sg396_checkbox.setChecked(False)
        self.sg396_checkbox.setStyleSheet("QCheckBox::indicator:hover" "{""background-color : yellow;""}"
                                            "QCheckBox::indicator:pressed" "{""background-color : lightgreen;""}")
        self.sg396_checkbox.stateChanged.connect(lambda:self.sg396_checked(self.sg396_checkbox))

        self.sg396_opacity_effects = []
        for i in range(5):
            self.sg396_opacity_effects.append(QGraphicsOpacityEffect())
            self.sg396_opacity_effects[i].setOpacity(0.3)


        self.sideband_opts = ["Lower", "Upper"]
        self.channel_opts = ["IQ", "DEER"]

        self.sg396_params_widget_1 = ParamsWidget(
            { 
                'channel': {'display_text': 'AWG Channel: ',
                                        'widget': ComboBox(items = self.channel_opts)},

                'RF_Frequency': {
                    'display_text': 'MW Frequency: ',
                    'widget': SpinBox(
                        value = 2.87e9,
                        suffix = 'Hz',
                        siPrefix = True,
                        bounds = (100e3, 6e9),
                        dec = True,
                    ),
                },

                'RF_Power': {
                    'display_text': 'MW Power: ',
                    'widget': SpinBox(
                        value = 1e-9,
                        suffix = 'W',
                        siPrefix = True,
                    )
                },

                'deer_power': {'display_text': 'DEER RF Power: ',
                                        'widget': SpinBox(value = 0.2, suffix = 'V', siPrefix = True)},
            },
            get_param_value_funs = {ComboBox: self.get_combobox_val}
        )

        self.sg396_params_widget_2 = ParamsWidget(
            {
                'sideband_freq': {'display_text': 'Sideband Modulation Frequency: ',
                                        'widget': SpinBox(value = 30e6, suffix = 'Hz', siPrefix = True, bounds = (100, 100e6), dec = True)},
                
                'sideband_power': {'display_text': 'Sideband Power: ',
                                        'widget': SpinBox(value = 0.45, suffix = 'V', siPrefix = True)},
                
                'sideband': {'display_text': 'Sideband: ',
                                        'widget': ComboBox(items = self.sideband_opts)},
                
                'i_offset': {'display_text': 'I Offset: ',
                                        'widget': SpinBox(value = -0.002, suffix = 'V', siPrefix = True)},
                
                'q_offset': {'display_text': 'Q Offset: ',
                                        'widget': SpinBox(value = -0.004, suffix = 'V', siPrefix = True)},
            },
            get_param_value_funs = {ComboBox: self.get_combobox_val}
        )


        self.sg396_status_label = QLabel("SRS SG396 status here")
        self.sg396_status_label.setStyleSheet("color: white; background-color: black; border: 4px solid black;")
        self.sg396_status_label.setFixedHeight(40)

        self.sg396_emit_button = QPushButton("Emit")
        self.sg396_emit_button_proc = ProcessRunner()
        self.sg396_emit_button.clicked.connect(lambda:self.sg396_emit_button_clicked())

        self.sg396_stop_button = QPushButton("Stop")
        self.sg396_stop_button_proc = ProcessRunner()
        self.sg396_stop_button.clicked.connect(lambda:self.sg396_stop_button_clicked())
        
        self.sg396_params_widget_1.setEnabled(False)
        self.sg396_params_widget_2.setEnabled(False)
        self.sg396_status_label.setEnabled(False)
        self.sg396_emit_button.setEnabled(False)
        self.sg396_stop_button.setEnabled(False)

        self.sg396_params_widget_1.setGraphicsEffect(self.sg396_opacity_effects[0])
        self.sg396_params_widget_2.setGraphicsEffect(self.sg396_opacity_effects[1])
        self.sg396_status_label.setGraphicsEffect(self.sg396_opacity_effects[2])
        self.sg396_emit_button.setGraphicsEffect(self.sg396_opacity_effects[3])
        self.sg396_stop_button.setGraphicsEffect(self.sg396_opacity_effects[4])

    def init_laser_widgets(self):
        self.laser_label = QLabel("Laser")
        self.laser_label.setFixedHeight(20)
        self.laser_label.setStyleSheet("font-weight: bold")

        self.laser_b1 = QRadioButton("CW ON")
        self.laser_b1.toggled.connect(lambda:self.toggle_laser(self.laser_b1))
            
        self.laser_b2 = QRadioButton("CW OFF")
        self.laser_b2.setChecked(True)
        self.laser_b2.toggled.connect(lambda:self.toggle_laser(self.laser_b2))
    
        self.laser_power_slider = QSlider()
        self.laser_power_slider.setOrientation(Qt.Orientation.Horizontal)
        # self.laser_power_slider.setTickPosition(QSlider.TicksBelow)
        self.laser_power_slider.setTickInterval(1)
        self.laser_power_slider.setMinimum(0)
        self.laser_power_slider.setMaximum(110)
        self.laser_power_slider.valueChanged.connect(lambda: self.laser_power_changed())
        self.laser_power_slider.setValue(0)
        # VALUE of laser power (in units of % diode current)
        self.laser_power = self.laser_power_slider.value()

        self.laser_power_label = QLabel("")
        self.laser_power_label.setFont(QFont("Sanserif", 15))

        self.laser_shutter_button = QPushButton("Open laser shutter")
        self.laser_shutter_button.clicked.connect(lambda: self.laser_shutter_status_changed())

        self.laser_status_label = QLabel("Laser status here")
        self.laser_status_label.setStyleSheet("color: white; background-color: black; border: 4px solid black;")
        self.laser_status_label.setFixedHeight(40)

    def init_flipper_widgets(self):
        self.flipper_label = QLabel("Flip Mirror")
        self.flipper_label.setFixedHeight(20)
        self.flipper_label.setStyleSheet("font-weight: bold")

        self.flipper_b1 = QRadioButton("APD")
        self.flipper_b1.toggled.connect(lambda:self.toggle_flipper(self.flipper_b1))
            
        self.flipper_b2 = QRadioButton("PMT")
        self.flipper_b2.toggled.connect(lambda:self.toggle_flipper(self.flipper_b2))

    def init_pmt_shutter_widgets(self):
        self.pmt_shutter_label = QLabel("PMT Shutter")
        self.pmt_shutter_label.setFixedHeight(20)
        self.pmt_shutter_label.setStyleSheet("font-weight: bold")

        self.pmt_shutter_button = QPushButton("Open PMT shutter")
        self.pmt_shutter_button.clicked.connect(lambda: self.pmt_shutter_status_changed())

    def layouts(self):
        '''
        GUI Layout Structure
        '''
        self.gui_layout = QVBoxLayout()

        self.magnet_frame = QFrame(self)
        self.magnet_frame.setStyleSheet("background-color: #2b2b2b")

        self.magnet_layout = QGridLayout(self.magnet_frame)
        self.magnet_layout.setSpacing(0)
        self.magnet_layout.addWidget(self.magnet_label,1,1,1,3)
        self.magnet_layout.addWidget(self.r_label,2,1,1,1, Qt.AlignmentFlag.AlignCenter)
        self.magnet_layout.addWidget(self.polar_label,2,2,1,1, Qt.AlignmentFlag.AlignCenter)
        self.magnet_layout.addWidget(self.azi_label,2,3,1,1, Qt.AlignmentFlag.AlignCenter)

        self.zaber_frame = QFrame(self)
        self.zaber_frame.setStyleSheet("background-color: #2b2b2b")
        # self.zaber_frame.setFrameShape(QFrame.StyledPanel)
        # self.zaber_frame.setFrameShadow(QFrame.Raised)

        self.r_layout = QGridLayout(self.zaber_frame)
        self.r_layout.setSpacing(0)
        self.r_layout.addWidget(self.r_label,1,1,1,2, Qt.AlignmentFlag.AlignCenter)
        # self.r_layout.addWidget(self.r_park_button,2,1,1,2)
        self.r_layout.addWidget(self.r_move_checkbox,2,1)
        self.r_layout.addWidget(self.r_move_types,2,2)
        self.r_layout.addWidget(self.r_move_position,3,1)
        self.r_layout.addWidget(self.r_move_position_units,3,2)
        self.r_layout.addWidget(self.r_execute_move_button,4,1,1,2)
        self.r_layout.addWidget(self.r_stop_button,5,1,1,2)

        self.thor1_frame = QFrame(self)
        self.thor1_frame.setStyleSheet("background-color: #2b2b2b")
        # self.thor1_frame.setFrameShape(QFrame.StyledPanel)
        # self.thor1_frame.setFrameShadow(QFrame.Raised)
        
        self.polar_layout = QGridLayout(self.thor1_frame)
        self.polar_layout.setSpacing(0)
        self.polar_layout.addWidget(self.polar_label,1,1,1,2)
        # self.polar_layout.addWidget(self.polar_enable_button,2,1,1,2)
        self.polar_layout.addWidget(self.polar_move_checkbox,2,1)
        self.polar_layout.addWidget(self.polar_move_types,2,2)
        self.polar_layout.addWidget(self.polar_move_position,3,1)
        self.polar_layout.addWidget(self.polar_move_position_units,3,2)
        self.polar_layout.addWidget(self.polar_execute_move_button,4,1,1,2)
        self.polar_layout.addWidget(self.polar_stop_button,5,1,1,2)
        
        self.thor2_frame = QFrame(self)
        self.thor2_frame.setStyleSheet("background-color: #2b2b2b")
        # self.thor2_frame.setFrameShape(QFrame.StyledPanel)
        # self.thor2_frame.setFrameShadow(QFrame.Raised)

        self.azi_layout = QGridLayout(self.thor2_frame)
        self.azi_layout.setSpacing(0)
        self.azi_layout.addWidget(self.azi_label,1,1,1,2)
        # self.azi_layout.addWidget(self.azi_enable_button,2,1,1,2)
        self.azi_layout.addWidget(self.azi_move_checkbox,2,1)
        self.azi_layout.addWidget(self.azi_move_types,2,2)
        self.azi_layout.addWidget(self.azi_move_position,3,1)
        self.azi_layout.addWidget(self.azi_move_position_units,3,2)
        self.azi_layout.addWidget(self.azi_execute_move_button,4,1,1,2)
        self.azi_layout.addWidget(self.azi_stop_button,5,1,1,2)

        self.status_frame = QFrame(self)
        self.status_frame.setStyleSheet("background-color: #2b2b2b")

        self.status_bar_layout = QGridLayout(self.status_frame)
        self.status_bar_layout.setSpacing(0)
        self.status_bar_layout.addWidget(self.all_disable_button,1,1,1,1)
        self.status_bar_layout.addWidget(self.all_standby_button,1,2,1,1)
        self.status_bar_layout.addWidget(self.status_label,2,1,1,2)

        self.individual_cmds_layout = QGridLayout()
        self.individual_cmds_layout.setSpacing(0)

        self.individual_cmds_layout.addWidget(self.magnet_frame,1,1,1,3)
        self.individual_cmds_layout.addWidget(self.zaber_frame,2,1,1,1)
        self.individual_cmds_layout.addWidget(self.thor1_frame,2,2,1,1)
        self.individual_cmds_layout.addWidget(self.thor2_frame,2,3,1,1)
        self.individual_cmds_layout.addWidget(self.status_frame,3,1,1,3)



        self.sg396_frame = QFrame(self)
        self.sg396_frame.setStyleSheet("background-color: #2b2b2b")
        self.sg396_layout = QGridLayout(self.sg396_frame)
        self.sg396_layout.setSpacing(0)
        self.sg396_layout.addWidget(self.sg396_label,1,1,1,2)
        self.sg396_layout.addWidget(self.sg396_checkbox,2,1,1,2)
        self.sg396_layout.addWidget(self.sg396_params_widget_1,3,1,1,1)
        self.sg396_layout.addWidget(self.sg396_params_widget_2,3,2,1,1)
        self.sg396_layout.addWidget(self.sg396_status_label,4,1,1,2)
        self.sg396_layout.addWidget(self.sg396_emit_button,5,1,1,1)
        self.sg396_layout.addWidget(self.sg396_stop_button,5,2,1,1)

        self.sig_gens_layout = QGridLayout()
        self.sig_gens_layout.setSpacing(0)

        self.sig_gens_layout.addWidget(self.sg396_frame,1,1,1,1)



        self.laser_frame = QFrame(self)
        self.laser_frame.setStyleSheet("background-color: #454545")
        self.laser_layout = QGridLayout(self.laser_frame)
        self.laser_layout.setSpacing(0)
        self.laser_layout.addWidget(self.laser_label,1,1,1,2)
        self.laser_layout.addWidget(self.laser_b1,2,1,1,1)
        self.laser_layout.addWidget(self.laser_b2,2,2,1,1)
        self.laser_layout.addWidget(self.laser_power_slider,3,1,1,2)
        self.laser_layout.addWidget(self.laser_power_label,4,1,1,2)
        self.laser_layout.addWidget(self.laser_shutter_button,5,1,1,2)
        self.laser_layout.addWidget(self.laser_status_label,6,1,1,2)

        self.flipper_frame = QFrame(self)
        self.flipper_frame.setStyleSheet("background-color: #454545")
        self.flipper_layout = QGridLayout(self.flipper_frame)
        self.flipper_layout.setSpacing(0)
        self.flipper_layout.addWidget(self.flipper_label,1,1,1,2)
        self.flipper_layout.addWidget(self.flipper_b1,2,1,1,1)
        self.flipper_layout.addWidget(self.flipper_b2,2,2,1,1)


        self.pmt_shutter_frame = QFrame(self)
        self.pmt_shutter_frame.setStyleSheet("background-color: #454545")
        self.pmt_shutter_layout = QGridLayout(self.pmt_shutter_frame)
        self.pmt_shutter_layout.setSpacing(0)
        self.pmt_shutter_layout.addWidget(self.pmt_shutter_label,1,1,1,2)
        self.pmt_shutter_layout.addWidget(self.pmt_shutter_button,2,1,1,2)


        # self.srs_frame = QFrame(self)
        # self.srs_frame.setStyleSheet("background-color: #454545")
        # self.srs_layout = QGridLayout(self.srs_frame)
        # self.srs_layout.setSpacing(0)
        # self.srs_layout.addWidget(self.laser_label,1,1,1,2)

        self.other_widgets_layout = QHBoxLayout()
        self.other_widgets_layout.addWidget(self.laser_frame)
        self.other_widgets_layout.addWidget(self.flipper_frame)
        self.other_widgets_layout.addWidget(self.pmt_shutter_frame)

        # bfield_layout = QGridLayout()
        # bfield_layout.addWidget(self.bfield_calc_label,1,1)

        self.gui_layout.addLayout(self.individual_cmds_layout)
        self.gui_layout.addLayout(self.status_bar_layout)
        self.gui_layout.addLayout(self.sig_gens_layout)
        self.gui_layout.addLayout(self.other_widgets_layout)

        self.setLayout(self.gui_layout)
    
    def park_all(self):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()
        
        self.all_disable_proc.run(mag_control.disable_all)
        
        logger.info("All magnet stages have been parked upon startup of magnet alignment widget.")
    
    '''
    INTERACTIVE WIDGET FUNCTIONS
    '''

    def obtain_current_positions(self, positions):
        
        self.curr_r_left = round(positions[0][0], 2)
        self.curr_r_right = round(positions[0][1], 2)
        self.curr_theta = round(positions[1], 2)
        self.curr_phi = round(positions[2], 2)
        
        # print("ZBER CURRENT POSITIONS: ", self.curr_r_left, self.curr_r_right)
        # self.r_pos_label.setText(f"r position = {self.curr_r_left} mm")
        # self.r_pos_label.setText(f"r positions: L = {self.curr_r[0]} mm, R = {self.curr_r[1]} mm")
        # (f"(mag. sep. = {2*self.curr_r + self.r_offset} mm)")
        # self.polar_pos_label.setText(f"\u03B8 position = {self.curr_theta}\N{DEGREE SIGN}")
        # self.azi_pos_label.setText(f"\u03C6 position = {self.curr_phi}\N{DEGREE SIGN}")

    def r_move_type_changed(self):
        self.r_move_option = self.r_move_types.currentText()
        
        match self.r_move_types.currentIndex():
                    case 2:
                        self.r_move_position_units.setText("G")
                    case _: 
                        self.r_move_position_units.setText("mm")

    def move_button_checked(self, box):
        if box.text() == "R Move Type: ":
            if box.isChecked() == True:
                self.r_move_types.setEnabled(True)
                self.r_move_position.setEnabled(True)
                self.r_move_position_units.setEnabled(True)
                self.r_execute_move_button.setEnabled(True)
                [self.r_opacity_effects[i].setEnabled(False) for i in range(4)]

                match self.r_move_types.currentIndex():
                    case 2:
                        self.r_move_position_units.setText("G")
                    case _: 
                        self.r_move_position_units.setText("mm")
            else:
                self.r_move_types.setEnabled(False)
                self.r_move_position.setEnabled(False)
                self.r_move_position_units.setEnabled(False)
                self.r_move_position_units.setText("")
                self.r_execute_move_button.setEnabled(False)
                [self.r_opacity_effects[i].setEnabled(True) for i in range(4)]

        elif box.text() == "\u03B8 Move Type: ":
            if box.isChecked() == True:
                self.polar_move_types.setEnabled(True)
                self.polar_move_position.setEnabled(True)
                self.polar_move_position_units.setEnabled(True)
                self.polar_move_position_units.setText("\N{DEGREE SIGN}")
                self.polar_execute_move_button.setEnabled(True)
                [self.polar_opacity_effects[i].setEnabled(False) for i in range(4)]
            else:
                self.polar_move_types.setEnabled(False)
                self.polar_move_position.setEnabled(False)
                self.polar_move_position_units.setEnabled(False)
                self.polar_move_position_units.setText("")
                self.polar_execute_move_button.setEnabled(False)
                [self.polar_opacity_effects[i].setEnabled(True) for i in range(4)]
        
        elif box.text() == "\u03C6 Move Type: ":
            if box.isChecked() == True:
                self.azi_move_types.setEnabled(True)
                self.azi_move_position.setEnabled(True)
                self.azi_move_position_units.setEnabled(True)
                self.azi_move_position_units.setText("\N{DEGREE SIGN}")
                self.azi_execute_move_button.setEnabled(True)
                [self.azi_opacity_effects[i].setEnabled(False) for i in range(4)]
            else:
                self.azi_move_types.setEnabled(False)
                self.azi_move_position.setEnabled(False)
                self.azi_move_position_units.setEnabled(False)
                self.azi_move_position_units.setText("")
                self.azi_execute_move_button.setEnabled(False)
                [self.azi_opacity_effects[i].setEnabled(True) for i in range(4)]

        # else:
        #     if box.isChecked() == True:
        #         self.all_move_types.setEnabled(True)
        #         self.all_positions.setEnabled(True)
        #         self.all_execute_move_button.setEnabled(True)
        #         for i in range(4):
        #             if i == 2:
        #                 continue
        #             self.all_opacity_effects[i].setEnabled(False)
        #     else:
        #         self.all_move_types.setCurrentIndex(0)
        #         self.all_move_types.setEnabled(False)
        #         self.all_positions.setEnabled(False)
        #         self.clear_all_positions.setEnabled(False)
        #         self.all_execute_move_button.setEnabled(False)
        #         [self.all_opacity_effects[i].setEnabled(True) for i in range(4)]

    # def park_button_clicked(self, stage):
    #     reload(mag)
    #     mag_control = mag.NanoNMRMagnetMotion()
    #     stage = stage

    #     if stage == "zaber":
    #         if self.r_park_button.text() == "Unpark":
    #             self.r_park_button_proc.run(mag_control.enable, stage)
    #             self.r_park_button.setText("Park")
    #         else:
    #             self.r_park_button_proc.run(mag_control.disable, stage)
    #             self.r_park_button.setText("Unpark")
                
    #     elif stage == "thor_polar":
    #         if self.polar_enable_button.text() == "Enable":
    #             self.polar_enable_button_proc.run(mag_control.enable, stage)
    #             self.polar_enable_button.setText("Disable")
    #         else:
    #             self.polar_enable_button_proc.run(mag_control.disable, stage)
    #             self.polar_enable_button.setText("Enable")

    #     elif stage == "thor_azi":
    #         if self.azi_enable_button.text() == "Enable":
    #             self.azi_enable_button_proc.run(mag_control.enable, stage)
    #             self.azi_enable_button.setText("Disable")
    #         else:
    #             self.azi_enable_button_proc.run(mag_control.disable, stage)
    #             self.azi_enable_button.setText("Enable")
    #     else:
    #         pass

    #     return 0

    def single_move_clicked(self, stage):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()
        stage = stage

        if self.all_disable_button.text() == "Disable All":    
            if stage == 'zaber':
                type_idx = self.r_move_types.currentIndex()
                try:
                    new_position = float(self.r_move_position.text())
                    if type_idx == 0:
                        self.r_execute_move_button_proc.run(mag_control.move_to_orientation,
                                                    **{'stage': stage, 'new_pos': new_position, 'queue': self.q, 'abs': True})
                    elif type_idx == 1:
                        self.r_execute_move_button_proc.run(mag_control.move_to_orientation,
                                                    **{'stage': stage, 'new_pos': new_position, 'queue': self.q, 'abs': False})
                    else:
                        pass

                    # self.position_tracker = Worker(self.q, True)
                    # self.position_tracker.position.connect(self.obtain_current_positions)
                    # self.position_tracker.start()

                except ValueError:
                    logger.debug("Invalid entry for new position. Try again.")

            elif stage == 'thor_polar':
                type_idx = self.polar_move_types.currentIndex()
                try:
                    new_position = float(self.polar_move_position.text())
                    if type_idx == 0:
                        self.polar_execute_move_button_proc.run(mag_control.move_to_orientation,
                                                    **{'stage': stage, 'new_angle': new_position, 'queue': self.q, 'abs': True})
                    elif type_idx == 1:
                        self.polar_execute_move_button_proc.run(mag_control.move_to_orientation,
                                                    **{'stage': stage, 'new_angle': new_position, 'queue': self.q, 'abs': False})
                    else:
                        pass

                    # self.position_tracker = Worker(self.q, True)
                    # self.position_tracker.position.connect(self.obtain_current_positions)
                    # self.position_tracker.start()

                except ValueError:
                    logger.debug("Invalid entry for new angle. Try again.")

            elif stage == 'thor_azi':
                type_idx = self.azi_move_types.currentIndex()
                try:
                    new_position = float(self.azi_move_position.text())
                    if type_idx == 0:
                        self.azi_execute_move_button_proc.run(mag_control.move_to_orientation,
                                                    **{'stage': stage, 'new_angle': new_position, 'queue': self.q, 'abs': True})
                    elif type_idx == 1:
                        self.azi_execute_move_button_proc.run(mag_control.move_to_orientation,
                                                    **{'stage': stage, 'new_angle': new_position, 'queue': self.q, 'abs': False})
                    else:
                        pass
                    
                    # self.position_tracker = Worker(self.q, True)
                    # self.position_tracker.position.connect(self.obtain_current_positions)
                    # self.position_tracker.start()

                except ValueError:
                    logger.debug("Invalid entry for new angle. Try again.")
                
            else:
                pass

        else:
            self.status_label.setStyleSheet("color: black; background-color: red; border: 4px solid black;")
            self.status_label.setText("Magnet Status: CANNOT MOVE STAGE IS PARKED")
            
        return 0

    def stop_button_clicked(self, stage):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()
        stage = stage

        if stage == "zaber":
            self.r_stop_button_proc.run(mag_control.stop_motion, stage)
        elif stage == "thor_polar":
            self.polar_stop_button_proc.run(mag_control.stop_motion, stage)
        elif stage == "thor_azi":
            self.azi_stop_button_proc.run(mag_control.stop_motion, stage)     
        else:
            pass

        return 0

    # def home_button_clicked(self, stage):
    #     reload(mag)
    #     mag_control = mag.NanoNMRMagnetMotion()
    #     stage = stage
    #     if stage == "zaber":
    #         self.r_home_button_proc.run(mag_control.move_to_home, stage)
    #     elif stage == "thor_polar":
    #         self.polar_home_button_proc.run(mag_control.move_to_home, stage)
    #     elif stage == "thor_azi":
    #         self.azi_home_button_proc.run(mag_control.move_to_home, stage)     
    #     else:
    #         pass

    #     return 0

    def all_disable_clicked(self):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()
        
        if self.all_disable_button.text() == "Disable All":    
            self.all_disable_proc.run(mag_control.disable_all)
            self.all_disable_button.setText("Enable All")
            self.status_label.setStyleSheet("color: white; background-color: black; border: 4px solid black;")
            self.status_label.setText("Magnet Status: STAGES DISABLED")
            # self.r_park_button.setText("Unpark")
            # self.polar_enable_button.setText("Enable")
            # self.azi_enable_button.setText("Enable")          
        else:
            self.all_disable_proc.run(mag_control.enable_all)
            self.all_disable_button.setText("Disable All")
            self.status_label.setStyleSheet("color: black; background-color: white; border: 4px solid black;")
            self.status_label.setText("Magnet Status: STAGES ENABLED")
            # self.r_park_button.setText("Park")
            # self.polar_enable_button.setText("Disable")
            # self.azi_enable_button.setText("Disable")

    def all_standby_clicked(self):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()
        
        if self.all_disable_button.text() == "Disable All":
            self.all_standby_proc.run(mag_control.standby_all, **{'queue': self.q})
        else:
            self.status_label.setStyleSheet("color: black; background-color: red; border: 4px solid black;")
            self.status_label.setText("Magnet Status: CANNOT MOVE STAGE IS PARKED")
            
    def all_move_types_changed(self, state):
        if state == "Absolute":
            self.r, self.rPressed = QInputDialog.getDouble(self, "Set r", "Position (mm):", 100, 0, 100, 0.01)
            if self.rPressed:
                new_positions = f"r = {self.r} mm"
                self.all_positions.setText(new_positions)

            self.theta, self.thetaPressed = QInputDialog.getDouble(self, "Set \u03B8", "Angle (\N{DEGREE SIGN}):", 0, -50, 50, 0.01)
            if self.thetaPressed:
                new_positions += f"\n\u03B8 = {self.theta}\N{DEGREE SIGN}"
                self.all_positions.setText(new_positions)
            
            self.phi, self.phiPressed = QInputDialog.getDouble(self, "Set \u03C6", "Angle (\N{DEGREE SIGN}):", 0, 0, 130, 0.01)
            if self.phiPressed:
                new_positions += f"\n\u03C6 = {self.phi}\N{DEGREE SIGN}"
                self.all_positions.setText(new_positions)
            
            self.all_opacity_effects[2].setEnabled(False)
            self.clear_all_positions.setEnabled(True)

        elif state == "Jog":
            pass

        else:
            pass

    def clear_positions_clicked(self):
        self.all_positions.clear()
        self.all_move_types.setCurrentIndex(0)

    def all_move_clicked(self):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()
        
        move_type_idx = self.all_move_types.currentIndex()
        

        if move_type_idx == 1:
            '''Absolute movements'''
            self.all_execute_move_button_proc.run(mag_control.move_to_orientation,
                                        **{'stage': 'all', 
                                           'new_pos': [self.phi, self.theta, self.r], 
                                           'queue': self.q,
                                           'abs': True})
            
            # self.position_tracker = Worker(self.q, True)
            # self.position_tracker.position.connect(self.obtain_current_positions)
            # self.position_tracker.start()

        elif move_type_idx == 2:
            '''Move to certain B-field configuration'''
            self.all_execute_move_button_proc.run(mag_control.move_to_orientation,
                                        **{'stage': 'all', 
                                           'new_pos': [self.calculated_phi, self.calculated_theta, self.calculated_r], 
                                           'queue': self.q,
                                           'abs': True})
            
            # self.position_tracker = Worker(self.q, True)
            # self.position_tracker.position.connect(self.obtain_current_positions)
            # self.position_tracker.start()

    def all_stop_button_clicked(self):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()
        self.all_stop_proc.run(mag_control.stop_all_motion)

    def check_queue_from_mag(self):
        # queue checker to control progress bar display
        while not self.q.empty(): #if there is something in the queue
            queueText = self.q.get_nowait() #get it
            
            if queueText[0] == 'done':
                self.status_label.setStyleSheet("color: black; background-color: limegreen; border: 4px solid black;")
                self.status_label.setText("Magnet Status: MOVEMENT COMPLETED")

                self.r_label.setText(f"R = {queueText[1][0]}")
                self.polar_label.setText(f"\u03B8 = {round(queueText[2],1)}\N{DEGREE SIGN}")
                self.azi_label.setText(f"\u03C6 = {round(queueText[3],1)}\N{DEGREE SIGN}")

            elif queueText[0] == 'start standby':
                self.status_label.setStyleSheet("color: black; background-color: gold; border: 4px solid black;")
                self.status_label.setText("Magnet Status: MOVING TO STANDBY POSITION...")

            else:
                self.status_label.setStyleSheet("color: black; background-color: gold; border: 4px solid black;")
                self.status_label.setText("Magnet Status: MOVEMENT IN PROGRESS...")

            # elif queueText[:9]=='SAVE_REQ:': #if it's a save request, save data in new proc we'll track
            #     saveType, self.datasetName, self.nameForAutosave = queueText.split(':', 1)[1].split(' ')[::2]

            #     self.saveProcs.append(saveInNewProc(datasetName=self.datasetName, expNameForAutosave=self.nameForAutosave, saveType=saveType))
            # else: #you gave an invalid response in the queue, bozo
            #     print(f'Invalid queue_from_exp text:{queueText}')

        self.updateTimer.start(self.QUEUE_CHECK_TIME)





    # output sg396 frequency without sideband modulation
    # def sg396_emit_button_clicked(self):
    #     with InstrumentManager() as mgr:
    #         fun_kwargs = dict(**self.sg396_params_widget_1.all_params(), **self.sg396_params_widget_2.all_params())
    #         # print("power to emit:", fun_kwargs['RF_Power'])
    #         # print("freq to emit: ", fun_kwargs['RF_Frequency'])
    #         mgr.sg.set_frequency(fun_kwargs['RF_Frequency'])
    #         mgr.sg.set_rf_amplitude(fun_kwargs['RF_Power'])
    #         mgr.sg.set_mod_type('QAM')
    #         mgr.sg.set_mod_toggle(1)
    #         mgr.sg.set_rf_toggle(1)
    #         mgr.sg.set_mod_function('external')

    def sg396_checked(self, box):
        if box.isChecked() == True:
            self.sg396_params_widget_1.setEnabled(True)
            self.sg396_params_widget_2.setEnabled(True)
            self.sg396_status_label.setEnabled(True)
            self.sg396_emit_button.setEnabled(True)
            self.sg396_stop_button.setEnabled(True)
            [self.sg396_opacity_effects[i].setEnabled(False) for i in range(5)]

        else:
            self.sg396_params_widget_1.setEnabled(False)
            self.sg396_params_widget_2.setEnabled(False)
            self.sg396_status_label.setEnabled(False)
            self.sg396_emit_button.setEnabled(False)
            self.sg396_stop_button.setEnabled(False)
            [self.sg396_opacity_effects[i].setEnabled(True) for i in range(5)]

    def choose_sideband(self, opt, nv_freq, side_freq, pulse_axis='x'):
        match pulse_axis:
            case 'y':
                delta = 90
            case _: 
                delta = 0

        match opt:
            case 'Upper':
                frequency = nv_freq - side_freq
                iq_phases = [delta+90, delta+0]
            case 'Both':
                frequency = nv_freq - side_freq
                iq_phases = [delta+0, delta+90, delta+90, delta+0] # lower sideband phases + upper sideband phases
            case _:
                frequency = nv_freq + side_freq
                iq_phases = [delta+0, delta+90]

        return frequency, iq_phases
    
    def sg396_emit_button_clicked(self):
        with InstrumentManager() as mgr:
            sig_gen = mgr.sg
            hdawg = mgr.awg

            fun_kwargs = dict(**self.sg396_params_widget_1.all_params(), **self.sg396_params_widget_2.all_params())
            # print("power to emit:", fun_kwargs['RF_Power'])
            # print("freq to emit: ", fun_kwargs['RF_Frequency'])
            output_freq, iq_phases = self.choose_sideband(fun_kwargs['sideband'], fun_kwargs['RF_Frequency'], fun_kwargs['sideband_freq'], 'y')

            sig_gen.set_frequency(output_freq)
            sig_gen.set_rf_amplitude(fun_kwargs['RF_Power'])
            sig_gen.set_mod_type('QAM')
            sig_gen.set_mod_function('external')
            sig_gen.set_mod_toggle(1)

            hdawg.set_sequence(**{'seq': 'Test',
                                    'channel': fun_kwargs['channel'],
                                    'RF_Frequency': fun_kwargs['RF_Frequency'],
                                    'deer_power': fun_kwargs['deer_power'],
                                    'i_offset': fun_kwargs['i_offset'],
                                    'q_offset': fun_kwargs['q_offset'],
                                    'sideband_power': fun_kwargs['sideband_power'],
                                    'sideband_freq': fun_kwargs['sideband_freq'], 
                                    'iq_phases': iq_phases})
            
            sig_gen.set_rf_toggle(1)

            self.sg396_status_label.setStyleSheet("color: black; background-color: gold; border: 4px solid black;")
            self.sg396_status_label.setText(f"SRS SG396 Status: ON ({output_freq*1e-9} GHz at {fun_kwargs['RF_Power']*1e6} uW)")
            
    def sg396_stop_button_clicked(self):
        with InstrumentManager() as mgr:
            mgr.sg.set_rf_toggle(0)
            mgr.sg.set_mod_toggle(0)
            mgr.awg.set_disabled()

            self.sg396_status_label.setStyleSheet("color: white; background-color: black; border: 4px solid black;")
            self.sg396_status_label.setText("SRS SG396 Status: OFF")

    def toggle_laser(self, b):
        with InstrumentManager() as mgr:
            mgr.laser.set_modulation_state('cw')
            mgr.laser.set_analog_control_mode('current')
            mgr.laser.set_diode_current_realtime(self.laser_power)

            match b.text():
                case 'CW ON':
                    if b.isChecked() == True:
                        mgr.laser.laser_on()
                    else:
                        mgr.laser.laser_off()
                case 'CW OFF':
                    if b.isChecked() == True:
                        mgr.laser.laser_off()
                    else:
                        mgr.laser.laser_on()

    def toggle_flipper(self, b):
        with InstrumentManager() as mgr:
            daq = mgr.daq
            daq.open_do_task('flip mirror')
            daq.start_do_task()
            
            match b.text():
                case 'APD':
                    daq.write_do_task('flip mirror', detector='apd')
                case 'PMT':
                    daq.write_do_task('flip mirror', detector='pmt')

            daq.stop_do_task()
            daq.close_do_task()

    def pmt_shutter_status_changed(self):
        with InstrumentManager() as mgr:
            daq = mgr.daq
            daq.open_do_task('shutter')
            daq.start_do_task()

            if self.pmt_shutter_button.text() == "Open PMT shutter":
                self.pmt_shutter_button.setText("Close PMT shutter")
                daq.write_do_task('shutter', shutter_status='open')
            else:
                self.pmt_shutter_button.setText("Open PMT shutter")
                daq.write_do_task('shutter', shutter_status='close')

            daq.stop_do_task()
            daq.close_do_task()
        
    def laser_power_changed(self):
        self.laser_power = self.laser_power_slider.value()
        self.laser_power_label.setText(str(self.laser_power) + "%")
        with InstrumentManager() as mgr:
            mgr.laser.set_diode_current_realtime(self.laser_power)

    def laser_shutter_status_changed(self):
        with InstrumentManager() as mgr:
            laser_shutter = mgr.laser_shutter
            
            if self.laser_shutter_button.text() == "Open laser shutter":
                self.laser_shutter_button.setText("Close laser shutter")
                laser_shutter.open_shutter()
                self.laser_status_label.setStyleSheet("color: black; background-color: white; border: 4px solid black;")
                self.laser_status_label.setText("Shutter Status: OPEN")
            else:
                self.laser_shutter_button.setText("Open laser shutter")
                laser_shutter.close_shutter()
                self.laser_status_label.setStyleSheet("color: white; background-color: black; border: 4px solid black;")
                self.laser_status_label.setText("Shutter Status: CLOSED")

    # def field_calibration_button_clicked(self):
    #     """Runs when the 'run' button is pressed."""

    #     # reload the spin measurements module at runtime in case any changes were made to the code
    #     reload(mag)

    #     # create an instance of the ODMR class that implements the experimental logic.
    #     mag_control = mag.NanoNMRMagnetMotion()

    #     # run the run function in a new thread.
    #     self.run_f_map_cal.run(
    #         mag_control.field_calibration
    #         # self.params_widget.start_freq,
    #         # self.params_widget.stop_freq,
    #         # self.params_widget.num_points,
    #     )



    def kill_process(self):
        """Stop the run process."""
        self.run_proc.kill()



