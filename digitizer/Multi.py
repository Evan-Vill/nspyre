"""
Spectrum Instrumentation GmbH (c)

3_acq_multi.py

Shows a simple Standard multiple recording mode example using only the few necessary commands
- connect a function generator that generates a sine wave with 10-100 kHz frequency and 200 mV amplitude to channel 0
- triggering is done with a channel trigger on channel 0

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units 

import numpy as np
import matplotlib.pyplot as plt
import time as time 

from rpyc.utils.classic import obtain

import logging

logger = logging.getLogger(__name__)

class Multi(): 
    def __init__(self,ip_address):
       
        # initialize the default configuration 
        self._init_config = dict(
                                ACCOUPLE = 1,
                                AMP = 500 , # in mV
                                card_timeout = 5 * units.s,
                                DCCOUPLE =1,
                                HF_INPUT_50OHM = 1,
                                mw_num = None ,
                                mem_size = None, 
                                num_segment = None,
                                pretrig_size = 32 * units.Sa, 
                                posttrig_size=None,
                                readout_ch = None,
                                ip_address = ip_address,
                                runs=None, 
                                sampling_frequency = 0.4 * units.GHz,
                                segment_size = 512 * units.Sa,
                                TERMINATE50OHM = 1)
       
        # assign the configuration to self
        for key, value in self._init_config.items():
            setattr(self, key, value)
        
        # # open the connection with the digitizer
        self.connect()

    def connect(self):
        self.card = spcm.Card(self.ip_address) 
        self.card.__enter__()

        if self.card._closed==False:
            print("Successfully connected to the digitizer")
            if self.card._closed==True:
                assert print("Connection unsuccessful")

    def disconnect(self):
        self.card.__exit__()
        if self.card._closed==True:
            print("Card dissconection successful")
            # reset the configuraiton 
            for key, value in self._init_config.items():
                setattr(self, key, value)
        if self.card._closed==False:
            print("Card dissconection unsuccessful")

    def stop_card(self):
        self.card.stop(spcm.M2CMD_DATA_STOPDMA)
        print("Card stopped")

    def restart(self):
        self.card.reset()

    def assign_param(self,settings_dict):
        # assign the configuration to self
        for key, value in settings_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            ### TODO: PRINT THE PASSED CONFIG VALUE TO FIX THE UNIT PROBLEM 
   
    def config(self):
        
        # handling memory assignment 

        self.posttrig_size = (self.segment_size - self.pretrig_size)
        
        self.num_segment = self.runs*self.mw_num
        self.mem_size = self.num_segment * self.segment_size
        # print(obtain(self.mem_size), type(obtain(self.mem_size)))
        # setup clock engine
        clock = spcm.Clock(self.card)
        clock.mode(spcm.SPC_CM_INTPLL)
        clock.sample_rate(self.sampling_frequency)
        
        CH_mapping = {
            0 : spcm.CHANNEL0,
            1 : spcm.CHANNEL1
        }

        # self.channel = spcm.Channels(self.card, card_enable=CH_mapping.get(self.readout_ch))

        ACDC_mapping = {
            0: spcm.SPC_ACDC0,
            1: spcm.SPC_ACDC1}
        self.ACDC=ACDC_mapping.get(self.readout_ch)
        
        PATH_mapping={
            0: spcm.SPC_PATH0,
            1: spcm.SPC_PATH1}
        self.PATH = PATH_mapping.get(self.readout_ch)
        
        AMP_mapping={
            0: spcm.SPC_AMP0,
            1: spcm.SPC_AMP1}
        self.AMP_ch=AMP_mapping.get(self.readout_ch)

        # setup card mode
        self.card.card_mode(spcm.SPC_REC_STD_MULTI) # multiple recording mode
        self.card.timeout(self.card_timeout)

        # set Analog input parameters
        self.card.set_i(self.PATH, self.HF_INPUT_50OHM)
        self.card.set_i(self.AMP_ch, int(self.AMP))
        # self.card.set_i(spcm.SPC_OFFS0, int(50)) # offset by how much percent
        self.card.set_i(self.ACDC, self.ACCOUPLE)

        # set triggering
        self.card.set_i(spcm.SPC_TRIG_EXT0_MODE,  spcm.SPC_TM_POS)
        self.card.set_i(spcm.SPC_TRIG_ORMASK, spcm.SPC_TMASK_EXT0)
        self.card.set_i(spcm.SPC_TRIG_TERM, self.TERMINATE50OHM)
        self.card.set_i(spcm.SPC_TRIG_EXT0_ACDC, spcm.COUPLING_DC)
        self.card.set_i(spcm.SPC_TRIG_EXT0_LEVEL0, int(2000))
        
        self.multiple_recording = spcm.Multi(self.card)
        
        self.multiple_recording.memory_size(self.mem_size)
        self.multiple_recording.allocate_buffer(self.segment_size, num_segments=self.num_segment)
        self.multiple_recording.post_trigger(self.posttrig_size)
        self.multiple_recording.notify_samples(0)
        self.multiple_recording.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)

    def check_connection(self):
         if self.card._closed==False:
              pass
         elif self.card._closed==True:
            assert print("Digitizer is not connected")
 
    
    def start_buffer(self):
        self.card.start(spcm.M2CMD_CARD_ENABLETRIGGER)

    def acquire(self):
        try:
            
            self.card.set_i(spcm.SPC_M2CMD, spcm.M2CMD_DATA_WAITDMA)
            self.raw_data = np.copy(self.multiple_recording.buffer)
            self.card.stop(spcm.M2CMD_DATA_STOPDMA)
            self.card.reset()

        except spcm.SpcmTimeout as timeout:
            # logger.critical("Shape of digitizer data array in acquire: ", np.shape(self.data))
            # logger.critical()
            self.card.stop(spcm.M2CMD_DATA_STOPDMA)
            self.card.__exit__()
            self.card.__enter__()
       
        finally:
            self.card.reset()
            # self.data_voltage=self.channel.convert_data(self.raw_data)

        return(self.raw_data ) #,self.data_voltage)
