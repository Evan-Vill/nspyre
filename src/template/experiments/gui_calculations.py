"""
Example GUI elements.
"""
# from tkinter.tix import ComboBox
from tracemalloc import stop
import numpy as np
import logging
from functools import partial
from importlib import reload
from multiprocessing import Queue
from typing import Optional

from nspyre import DataSink
from pyqtgraph import SpinBox, ComboBox
from PyQt6.QtWidgets import QLabel, QPushButton, QCheckBox, QComboBox, QLineEdit, QRadioButton, QSlider
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout
from PyQt6.QtWidgets import QStackedWidget, QWidget, QGraphicsOpacityEffect
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCore import Qt, QTimer

from nspyre.misc.misc import ProcessRunner
from nspyre.misc.misc import run_experiment
from nspyre import experiment_widget_process_queue
from nspyre import InstrumentManager

import nv_experiments

class CalcWidget(QWidget):
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle('NV Widefield Microscope Instruments')

        # experiment params label
        self.laser_label = QLabel("Laser Settings:")
        self.laser_label.setFixedHeight(20)
        self.laser_label.setStyleSheet("font-weight: bold")

        self.laser_b1 = QRadioButton("ON")
        self.laser_b1.toggled.connect(lambda:self.toggle_laser(self.laser_b1))
            
        self.laser_b2 = QRadioButton("OFF")
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

        self.gui_layout = QVBoxLayout()
        
        self.laser_frame = QFrame(self)
        self.laser_frame.setStyleSheet("background-color: #454545")
        self.laser_layout = QGridLayout(self.laser_frame)
        self.laser_layout.setSpacing(0)
        self.laser_layout.addWidget(self.laser_label,1,1,1,2)
        self.laser_layout.addWidget(self.laser_b1,2,1,1,1)
        self.laser_layout.addWidget(self.laser_b2,2,2,1,1)
        self.laser_layout.addWidget(self.laser_power_slider,3,1,1,2)
        self.laser_layout.addWidget(self.laser_power_label,4,1,1,2)

        # self.srs_frame = QFrame(self)
        # self.srs_frame.setStyleSheet("background-color: #454545")
        # self.srs_layout = QGridLayout(self.srs_frame)
        # self.srs_layout.setSpacing(0)
        # self.srs_layout.addWidget(self.laser_label,1,1,1,2)

        self.laser_widgets_layout = QVBoxLayout()
        self.laser_widgets_layout.addWidget(self.laser_frame)

        # self.srs_widgets_layout = QVBoxLayout()
        # self.srs_widgets_layout.addWidget(self.srs_frame)

        self.gui_layout.addLayout(self.laser_widgets_layout)
        # self.gui_layout.addLayout(self.srs_widgets_layout)

        self.setLayout(self.gui_layout)

    def toggle_laser(self, b):
        with InstrumentManager() as mgr:
            mgr.OXXIUS_Laser.set_modulation_state('cw')
            mgr.OXXIUS_Laser.set_analog_control_mode('current')
            mgr.OXXIUS_Laser.set_diode_current_realtime(self.laser_power)

            match b.text():
                case 'ON':
                    if b.isChecked() == True:
                        mgr.OXXIUS_Laser.laser_on()
                    else:
                        mgr.OXXIUS_Laser.laser_off()
                case 'OFF':
                    if b.isChecked() == True:
                        mgr.OXXIUS_Laser.laser_off()
                    else:
                        mgr.OXXIUS_Laser.laser_on()

    def convert_to_gauss(self, B_field, units):
        match units: # convert all B field units to Gauss for calculation
            case 'T':
                B_field = B_field/1e4
            case 'mT':
                B_field = B_field/10
            case _:
                pass # default in G 

        return B_field
        
    def calc_nv_resonance(self, B_field, units):
        ### computes NV resonances in [GHz] from a B field
        nv_zfs = 2.87 # GHz
        gyro_e = 2.8025 # MHz/G
        
        B_field = self.convert_to_gauss(B_field, units)

        nv_minus1 = abs(nv_zfs - (gyro_e * B_field)/1000)
        nv_plus1 = nv_zfs + (gyro_e * B_field)/1000

        return nv_minus1, nv_plus1
    
    def calc_nuclear_resonance(self, B_field, units, spin):
        ### computes nuclear spin resonance [MHz] for a B field
        B_field = self.convert_to_gauss(B_field, units) 

        match spin: # gyromagnetic ratio in [MHz/T]
            case 'H1':
                gyro_n = 42.5775
            case 'N15':
                gyro_n = abs(-4.316)
            case 'N14':
                gyro_n = 3.077
            case 'C13':
                gyro_n = 10.7084
            case 'F19':
                gyro_n = 40.078
            case 'Al27':
                gyro_n = 11.103
            case 'P31':
                gyro_n = 17.235

        gyro_n = gyro_n * 1e-4 # convert to [MHz/G]

        larmor = gyro_n * B_field # nuclear Larmor frequency 

        period = 1/larmor # period in [us]
        half_period = period/2

        return larmor, period, half_period

        