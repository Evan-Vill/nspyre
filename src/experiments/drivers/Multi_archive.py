"""
Emmeline Riendeau 

Adapted from https://github.com/SpectrumInstrumentation/spcm/blob/master/src/examples/1_acquisition/3_acq_multi.py

2024-08-29 

"""

import spcm
from spcm import units 

import numpy as np
import matplotlib.pyplot as plt

class Multi(): 
    def __init__(self,ip_address):
       
        # initialize the default configuration 
        self._init_config = dict(channels = None,
                                data_block = None,
                                ip_address = ip_address,
                                minimum = None,
                                maximum = None,
                                num_samples = None,
                                num_samples_in_segment = 2/32 * units.KiB,
                                num_segments = None,
                                num_timestamps = None,
                                notify_samples = None,
                                sample_rate = None,
                                segment = None,
                                segment_cnt = None,
                                ts = None,
                                ts_data_range = None,
                                timestampVal2 = None,
                                unit_data_block = None,
                                sampling_frequency = 0.4 * units.GHz,
                                trigger_level = 1.5 * units.V,
                                amplitude = 1 * units.V,
                                readout_ch = None,
                                card_timeout = 5 *units.s,
                                run_time = None,
                                reps = None,
                                mode = None,
                                data_raw = None,
                                data_volts = None)
       
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
            print("Card disconection unsuccessful")

    def stop_card(self):
                self.card.stop(spcm.M2CMD_DATA_STOPDMA)
                print("Card stopped")
            
    def assign_param(self,settings_dict):
        # assign the configuration to self
        for key, value in settings_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.num_samples==None and self.reps is not None:
            self.num_samples=self.reps*self.num_samples_in_segment
        
    def config(self):
            # setup card mode
            self.card.card_mode(spcm.SPC_REC_STD_MULTI) # multiple recording mode
            self.card.timeout(5 * units.s)

            # setup trigger engine
            trigger = spcm.Trigger(self.card)
            trigger.or_mask(spcm.SPC_TMASK_NONE)

            # setup clock engine
            clock = spcm.Clock(self.card)
            clock.mode(spcm.SPC_CM_INTPLL)
            clock.sample_rate(self.sampling_frequency)

            # setup channel 
            channel_mapping = {
            0: spcm.CHANNEL0,
            1: spcm.CHANNEL1,
            2: spcm.CHANNEL2,
            3: spcm.CHANNEL3
            }

            enabled_channel= channel_mapping.get(self.readout_ch)

            channels = spcm.Channels(self.card, card_enable=enabled_channel) # readout channel 
            channels.amp(self.amplitude)

            # Channel triggering
            trigger.ch_or_mask0(channels[0].ch_mask())
            trigger.ch_mode(channels[0], spcm.SPC_TM_POS)
            trigger.ch_level0(channels[0], 0 * units.mV, return_unit=units.mV)

            # setup data transfer
            num_samples = self.num_samples #4 * units.KiS
            samples_per_segment = self.num_samples_in_segment # 1 * units.KiS
            self.multiple_recording = spcm.Multi(self.card)
            self.multiple_recording.memory_size(num_samples)
            self.multiple_recording.allocate_buffer(samples_per_segment)
            self.multiple_recording.post_trigger(samples_per_segment//2)

 
        
    def acquire(self):
        
            self.multiple_recording.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)

            # wait until the transfer has finished
            try:
                self.card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)

                self.data = self.multiple_recording.buffer
                self.time_data = self.multiple_recording.time_data()

                # this is the point to do anything with the data
                # e.g. calculate minimum and maximum of the acquired data
                # fig, ax = plt.subplots(data.shape[0], 1, sharex=True, layout='constrained')
                # for segment in range(data.shape[0]):
                #     print("Segment {}".format(segment))
                #     for channel in channels:
                #         chan_data = channel.convert_data(data[segment, :, channel]) # index definition: [segment, sample, channel]
                #         minimum = np.min(chan_data)
                #         maximum = np.max(chan_data)
                #         print(f"\t{channel}")
                #         print(f"\t\tMinimum: {minimum}")
                #         print(f"\t\tMaximum: {maximum}")

                #         ax[segment].plot(time_data, chan_data, '.', label="{}, Seg {}".format(channel.index, segment))
                #     ax[segment].set_title(f"Segment {segment}")
                #     ax[segment].yaxis.set_units(units.V)
                #     ax[segment].xaxis.set_units(units.us)
                #     ax[segment].axvline(0, color='k', linestyle='--', label='Trigger')
                # # ax.legend()
                # plt.show()
            except spcm.SpcmTimeout as timeout:
                print("Timeout...")

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"'DictWrapper' object has no attribute '{name}'")

    def getkeys(self):
         return(self._init_config.keys())