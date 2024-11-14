import numpy as np
import time
import math
from itertools import cycle
import logging
import scipy as sp
from scipy import signal
import datetime as Dt

from rpyc.utils.classic import obtain 

# nidaqmx STAYS SAME
import nidaqmx

#CHECK FOR ADDITIONAL CONSTANTS
from nidaqmx.constants import (AcquisitionType, Edge, TerminalConfiguration, VoltageUnits,
    READ_ALL_AVAILABLE, TaskMode, TriggerType, LineGrouping)
# from nidaqmx.constants import (AcquisitionType, CountDirection, Edge,
#     READ_ALL_AVAILABLE, TaskMode, TriggerType)

# from nidaqmx._task_modules.channels.ai_channel import AIChannel
from nidaqmx.stream_readers import AnalogSingleChannelReader
from nidaqmx.stream_writers import AnalogSingleChannelWriter

class NIDAQ():

    def __init__(self):
        
        # cDAQ-9185
        
        self.apd_channel = 'cDAQ9185-214A4EDMod3/ai0' # NI 9223
        self.pmt_channel = 'cDAQ9185-214A4EDMod3/ai2' # NI 9223

        self.clk_channel = '/cDAQ9185-214A4EDMod4/PFI0'
        self.shutter_channel = '/cDAQ9185-214A4EDMod4/PFI4' # NI 9401
        self.flip_mirror_channel = '/cDAQ9185-214A4EDMod4/PFI7' # NI 9401
        
        self.pmt_ctrl_voltage_channel = '/cDAQ9185-214A4EDMod1/ao0' # NI 9263

        self.sampling_rate = 50e3

        # self.task = None
        self.do_task = None
        self.ao_task = None
        self.ai_task = None
        self.ao_writer = None
        self.ai_reader = None
        
        self.daq_do_channel_mapping = {
            'shutter': self.shutter_channel,
            'flip mirror': self.flip_mirror_channel}
        
        self.daq_ai_channel_mapping = {
            'APD': self.apd_channel,
            'PMT': self.pmt_channel}
        

    def open_do_task(self, device):

        self.do_task = nidaqmx.Task()
        
        self.do_task.do_channels.add_do_chan(self.daq_do_channel_mapping.get(device))
        
    def open_ao_task(self):

        self.ao_task = nidaqmx.Task()
      
        self.ao_task.ao_channels.add_ao_voltage_chan(
                                self.pmt_ctrl_voltage_channel)
        
        self.ao_writer = AnalogSingleChannelWriter(self.ao_task.out_stream, auto_start = False)

    def open_ai_task(self, device, buffer_len):
            
        self.ai_task = nidaqmx.Task()
        
        self.ai_task.ai_channels.add_ai_voltage_chan(
                                self.daq_ai_channel_mapping.get(device),
                                terminal_config = TerminalConfiguration.DEFAULT,
                                min_val = -10,
                                max_val = 10,
                                units = VoltageUnits.VOLTS
        )

        self.ai_task.timing.cfg_samp_clk_timing(
                                self.sampling_rate,
                                source = self.clk_channel,
                                active_edge = Edge.FALLING,
                                sample_mode = AcquisitionType.FINITE,
                                samps_per_chan = buffer_len
        )
        
        self.ai_reader = AnalogSingleChannelReader(self.ai_task.in_stream)
        

    ### Digital output task management
    def start_do_task(self):
        self.do_task.start()

    def write_do_task(self, device, **kwargs):
        match device:
            case 'flip mirror':
                if kwargs['detector'] == 'apd':
                    data = [False]
                else: 
                    data = [True]
            case 'shutter':
                if kwargs['shutter_status'] == 'open':
                    data = [True]
                else:
                    data = [False]
        
        self.do_task.write(data)

    def stop_do_task(self):  
        print(f"{self.do_task} task stopped.")
        self.do_task.stop()

    def close_do_task(self):
        print(f"{self.do_task} task closed.")
        try:
            self.do_task.close()
        except AttributeError:
            print("'NoneType' task disregarded.")
        else:
            self.do_task = None


    ### Analog output task management
    def start_ao_task(self):
        self.ao_task.start()

    def write_ao_task(self, voltage):
        voltage = obtain(voltage)
        # print("VOLTAGE TO SEND TO AO0: ", voltage)
        data = np.empty(100)
        data[:] = voltage
        self.ao_writer.write_many_sample(data)

    def stop_ao_task(self):  
        print(f"{self.ao_task} Analog output task stopped.")
        self.ao_task.stop()

    def close_ao_task(self):
        print(f"{self.ao_task} task closed.")
        try:
            self.ao_task.close()
        except AttributeError:
            print("'NoneType' task disregarded.")
        else:
            self.ao_task = None
    

    ### Analog input task management
    def start_ai_task(self):
        self.ai_task.start()

    def stop_ai_task(self):  
        print(f"{self.ai_task} task stopped.")
        self.ai_task.stop()

    def close_ai_task(self):
        print(f"{self.ai_task} task closed.")
        try:
            self.ai_task.close()
        except AttributeError:
            print("'NoneType' task disregarded.")
        else:
            self.ai_task = None
            self.ai_reader = None
    






    def read_samples(self, buffer, buffer_len, timeout):
        
        try:
            buffer = obtain(buffer)
            print("DAQ received empty buffer. Reading samples...")
            # print(type(obtain(buffer)))
            # print(buffer.dtype)
            self.ai_reader.read_many_sample(
                                    buffer,
                                    number_of_samples_per_channel = buffer_len,
                                    timeout = timeout
            )
            
            return buffer
            
        except Exception as e:
            print(e)
            self.stop_ai_task()
            self.close_ai_task()
    
