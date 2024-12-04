'''
Pulse sequence class with all sequences used for Nanoscale NMR experiments for new nspyre

Edited & rewritten by Evan Villafranca - September 10, 2022
'''

import time
import numpy as np
import pandas as pd
from math import sin, cos, radians
import typing as t

from rpyc.utils.classic import obtain

from pulsestreamer.grpc.pulse_streamer_grpc import PulseStreamer
# from c:\NSpyre\miniconda3\envs\ev_nspy\Lib\site-packages\pulsestreamer
from pulsestreamer.sequence import Sequence
from pulsestreamer.enums import TriggerStart

class Pulses():

    '''
    ALL UNITS: [ns]
    '''
    def __init__(self, laser_time = 15e3, initial_delay = 100, singlet_decay = 500, readout_time = 1000, 
                 MW_buffer_time = 100, probe_time = 50e3, clock_time = 11, trig_spot = 70, 
                 awg_trig_time = 10, awg_pulse_delay = 0, ip="10.135.70.193"):
        '''
        :param channel_dict: Dictionary of which channels correspond to which instr controls
        :param readout_time: Laser+gate readout time in ns
        :param laser_time: Laser time to reinit post readout
        :param initial_delay: Delay in laser turning on
        :param MW_buffer_time: Buffer after MW turns off
        :param IQ: IQ modulation/analog channels
        '''
        self.channel_dict = {"clock": 0, "VSG": 1, "int": 4, "laser": 7}
        self.laser_time = laser_time
        self.initial_delay = initial_delay
        self.singlet_decay = singlet_decay
        self.readout_time = readout_time
        self.MW_buffer_time = MW_buffer_time
        self.probe_time = probe_time
        self.clock_time = clock_time
        self.trig_spot = trig_spot
        self.awg_trig_time = awg_trig_time
        self.awg_pulse_delay = awg_pulse_delay

        self.Pulser = PulseStreamer(ip)
        self.sequence = Sequence()

        self.latest_streamed = pd.DataFrame({})
        self.total_time = 0 #update when a pulse sequence is streamed

    def has_sequence(self):
        """
        Has Sequence
        """
        return self.Pulser.hasSequence()
    
    def has_finished(self):
        """
        Has Finished
        """
        return self.Pulser.hasFinished()
    
    def laser_on(self):
        return self.Pulser.constant(([7], 0.0, 0.0))

    def stream(self,seq,n_runs):
        seq = obtain(seq)
        # print(type(seq))
        # print(seq)
        self.Pulser.stream(seq,n_runs)

    def clocksource(self,clk_src):
        self.Pulser.selectClock(clk_src)

    def _normalize_IQ(self, IQ):
        self.IQ = IQ/(2.5*np.linalg.norm(IQ))

    _T = t.TypeVar('_T')

    def convert_type(self, arg: t.Any, converter: _T) -> _T:
        return converter(arg)

    def set_soft_trigger(self):
        self.Pulser.setTrigger(TriggerStart.SOFTWARE)

    def start_now(self):
        self.Pulser.startNow()

    '''
    PULSE SEQUENCES FOR NanoNMR EXPERIMENTS
    '''   
    def SigvsTime(self, sampling_interval):
        seq = self.Pulser.createSequence()
        
        trig_off = sampling_interval - self.clock_time
        daq_clock_seq = [(trig_off, 0), (self.clock_time, 1)]
        print(daq_clock_seq)

        seq.setDigital(0, daq_clock_seq) # integrator trigger

        return seq
    
    def CW_ODMR(self, num_freqs):
        
        '''
        CW ODMR Sequence
        Laser on for entire sequence. 
        MW on for probe_time.
        MW off for probe_time.
        User sets how many voltage samples (num_clocks) to take during each MW on/off window.
        '''
    
        def SingleCW_ODMR():
            
            # create sequence object
            seq_on = self.Pulser.createSequence()
            seq_off = self.Pulser.createSequence()

            # digitizer trigger timing
            clock_off1 = self.probe_time - 2*self.readout_time - self.clock_time
            clock_off2 = 2*self.readout_time

            iq_off = self.probe_time - self.awg_trig_time

            # define sequence structure for clock and MW I/Q channels
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
            
            mw_iq_seq_on = [(self.awg_trig_time, 1), (iq_off, 0)]
            mw_iq_seq_off = [(self.probe_time, 0)]

            # assign sequences to respective channels
            seq_on.setDigital(1, dig_clock_seq) # digitizer trigger
            seq_on.setDigital(2, mw_iq_seq_on) # MW IQ

            seq_off.setDigital(1, dig_clock_seq) # digitizer trigger
            seq_off.setDigital(2, mw_iq_seq_off) # MW IQ

            return seq_on + seq_off

        seqs = self.Pulser.createSequence()

        for i in range(num_freqs):
            seqs += SingleCW_ODMR()

        return seqs
    
    def CW_FM_ODMR(self, params):
        
        '''
        CW ODMR Sequence
        Laser on for entire sequence. 
        MW on for probe_time.
        MW off for probe_time.
        User sets how many voltage samples (num_clocks) to take during each MW on/off window.
        '''
        off_res_voltage = self.convert_type(round(params[0]), float)

        def SingleCW_ODMR(volt):
            
            volt = float(volt)

            # create sequence object
            seq = self.Pulser.createSequence()

            # digitizer trigger timing
            clock_off1 = self.probe_time - self.readout_time - self.clock_time
            clock_off2 = self.probe_time - self.clock_time
            clock_off3 = self.readout_time

            # define sequence structure for clock and MW I/Q channels
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0), (self.clock_time, 1), (clock_off3, 0)]
            
            mw_seq = [(self.probe_time, volt), (self.probe_time, off_res_voltage)]
            # print(mw_seq)
            # assign sequences to respective channels
            seq.setDigital(1, dig_clock_seq) # digitizer trigger
            seq.setAnalog(0, mw_seq) # MW IQ

            return seq

        seqs = self.Pulser.createSequence()

        for v in params:
            seqs += SingleCW_ODMR(v)
    
        return seqs
    
    def Pulsed_ODMR(self, params, pi_xy, pi_time):
        '''
        Pulsed ODMR sequence
        '''
        ## Run a MW pulse of varying frequency, then measure the signal
        ## and reference counts from NV.
        
        if pi_xy == 'x':
            self.IQ_ON = self.IQpx
        else:
            self.IQ_ON = self.IQpy

        def SinglePulsedODMR():
            '''
            CREATE SINGLE PULSED ODMR SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''

            '''
            DEFINE SPECIAL TIME INTERVALS FOR EXPERIMENT
            '''
            # padding time to equalize duration of every run (for different vsg_on durations)
            # pad_time = 50000 - self.initial_delay - self.laser_time - self.singlet_decay - iq_on - self.MW_buffer_time - self.readout_time 

            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''

            laser_off1 = self.initial_delay 
            laser_off2 = self.singlet_decay + pi_time + self.MW_buffer_time
            laser_off3 = 200 # constant wait 200 ns between sweeps

            # mw I & Q off windows
            iq_off1 = laser_off1 + self.laser_time + self.singlet_decay
            iq_off2 = self.MW_buffer_time + 1*self.readout_time + laser_off3 # + self.laser_time # + laser_off4 + laser_off5

            # DAQ trigger windows
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.readout_time - self.trig_spot - self.clock_time
            clock_off2 = self.trig_spot + laser_off3
                   
            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            seq_on = self.Pulser.createSequence()
            seq_off = self.Pulser.createSequence()

            # define sequence structure for laser            
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
                        #  (laser_off3, 0), (self.laser_time, 1), (laser_off4, 0), (self.readout_time, 1), (laser_off5, 0)]
        
            # define sequence structure for DAQ trigger
            daq_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]

            # define sequence structure for MW I and Q when MW = ON
            mw_I_on_seq = [(iq_off1, self.IQ0[0]), (pi_time, self.IQ_ON[0]), (iq_off2, self.IQ0[0])]
            mw_Q_on_seq = [(iq_off1, self.IQ0[1]), (pi_time, self.IQ_ON[1]), (iq_off2, self.IQ0[1])]
            
            # when MW = OFF
            mw_I_off_seq = [(iq_off1, self.IQ0[0]), (pi_time, self.IQ0[0]), (iq_off2, self.IQ0[0])]
            mw_Q_off_seq = [(iq_off1, self.IQ0[1]), (pi_time, self.IQ0[1]), (iq_off2, self.IQ0[1])]

            # switch_on_seq = [(iq_off1 - 20, 0), (iq_on + 40, 1), (iq_off2 - 20, 0)]
            # switch_off_seq = [(iq_off1 - 20, 0), (iq_on + 40, 0), (iq_off2 - 20, 0)]

            # assign sequences to respective channels for seq_on
            seq_on.setDigital(3, laser_seq) # laser 
            seq_on.setDigital(0, daq_clock_seq) # integrator trigger
            # seq_on.setDigital(1, switch_on_seq) # RF control switch
            seq_on.setAnalog(0, mw_I_on_seq) # mw_I
            seq_on.setAnalog(1, mw_Q_on_seq) # mw_Q
            
            # assign sequences to respective channels for seq_off
            seq_off.setDigital(3, laser_seq) # laser
            seq_off.setDigital(0, daq_clock_seq) # integrator trigger
            # seq_off.setDigital(1, switch_off_seq) # RF control switch
            seq_off.setAnalog(0, mw_I_off_seq) # mw_I
            seq_off.setAnalog(1, mw_Q_off_seq) # mw_Q

            return seq_on + seq_off

        seqs = self.Pulser.createSequence()

        for i in range(params):
            seqs += SinglePulsedODMR()

        return seqs
    
    def Rabi(self, params):
        '''
        Rabi sequence
        '''
        ## Run a MW pulse of varying duration, then measure the signal
        ## and reference counts from NV.
        # self.total_time = 0
        longest_time = self.convert_type(round(max(params)), float)
        ## we can measure the pi time on x and on y.
        ## they should be the same, but they technically
        ## have different offsets on our pulse streamer.

        def SingleRabi(iq_on):
            '''
            CREATE SINGLE RABI SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''

            iq_on = float(round(iq_on)) # convert to proper data type to avoid undesired rpyc netref data type

            '''
            DEFINE SPECIAL TIME INTERVALS FOR EXPERIMENT
            '''
            # padding time to equalize duration of every run (for different vsg_on durations)
            # pad_time = 50000 - self.initial_delay - self.laser_time - self.singlet_decay - iq_on - self.MW_buffer_time - self.readout_time 
            pad_time = longest_time - iq_on

            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''

            laser_off1 = self.initial_delay 
            laser_off2 = self.singlet_decay + iq_on + self.MW_buffer_time
            laser_off3 = 100 + pad_time
            # laser_off3 = pad_time + self.rest_time_btw_seqs
            # laser_off4 = laser_off2
            # laser_off5 = self.rest_time_btw_seqs

            # mw I & Q off windows
            iq_off1 = laser_off1 + self.laser_time + self.singlet_decay
            iq_off2 = (iq_on - self.awg_trig_time) + self.MW_buffer_time + self.readout_time + laser_off3 # + self.laser_time # + laser_off4 + laser_off5

            # Digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3
                   
            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            seq_on = self.Pulser.createSequence()
            seq_off = self.Pulser.createSequence()

            # define sequence structure for laser            
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
                        #  (laser_off3, 0), (self.laser_time, 1), (laser_off4, 0), (self.readout_time, 1), (laser_off5, 0)]
        
            # define sequence structure for DAQ trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]

            # define sequence structure for MW I and Q when MW = ON
            mw_iq_on_seq = [(iq_off1, 0), (self.awg_trig_time, 1), (iq_off2, 0)]
            mw_iq_off_seq = [(iq_off1, 0), (self.awg_trig_time, 0), (iq_off2, 0)]

            # assign sequences to respective channels for seq_on
            seq_on.setDigital(3, laser_seq) # laser 
            seq_on.setDigital(1, dig_clock_seq) # digitizer trigger
            seq_on.setDigital(2, mw_iq_on_seq) # RF control switch

            # assign sequences to respective channels for seq_off
            seq_off.setDigital(3, laser_seq) # laser
            seq_off.setDigital(1, dig_clock_seq) # digitizer trigger
            seq_off.setDigital(2, mw_iq_off_seq) # RF control switch

            return seq_on + seq_off

        seqs = self.Pulser.createSequence()

        for mw_time in params:
            seqs += SingleRabi(mw_time)

        return seqs
 
    def Optical_T1(self, params):
        '''
        Optical T1 sequence with integrator
        '''
        ## Run a pi pulse, then measure the signal
        ## and reference counts from NV.
        longest_time = self.convert_type(round(max(params)), float)
        print("LONGEST T1 time to plot: ", longest_time)
        ## we can measure the pi time on x and on y.
        ## they should be the same, but they technically
        ## have different offsets on our pulse streamer.

        def SingleOptical_T1(tau_time):
            '''
            CREATE SINGLE T1 SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''

            tau_time = int(round(tau_time)) # convert to proper data type to avoid undesired rpyc netref data type

            '''
            DEFINE SPECIAL TIME INTERVALS FOR EXPERIMENT
            '''
            # padding time to equalize duration of every run
            pad_time = longest_time - tau_time 

            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''

            # laser trigger windows
            laser_off1 = self.initial_delay
            laser_off2 = tau_time
            laser_off3 = self.initial_delay + pad_time
       
            # digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3

            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            seq = self.Pulser.createSequence()
            seq_ref = self.Pulser.createSequence()

            # define sequence structure for laser
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]

            # define sequence structure for digitizer trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]

            # print("LASER SEQ: ", laser_seq)

            # assign sequences to respective channels for seq_on
            seq.setDigital(3, laser_seq) # laser
            seq.setDigital(1, dig_clock_seq) # digitizer trigger            
            seq_ref.setDigital(3, laser_seq) # laser
            seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
                       
            return seq + seq_ref

        seqs = self.Pulser.createSequence()

        for tau in params:
            seqs += SingleOptical_T1(tau)

        return seqs

    def Diff_T1(self, params, pi_xy, pi_time):
        '''
        MW (differential) T1 sequence 
        '''
        ## Run a pi pulse, then measure the signal
        ## and reference counts from NV.
        longest_time = self.convert_type(round(max(params)), float)
        pi_time = self.convert_type(round(pi_time), float)

        ## we can measure the pi time on x and on y.
        ## they should be the same, but they technically
        ## have different offsets on our pulse streamer.

        def SingleDiff_T1(tau_time):
            '''
            CREATE SINGLE T1 SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''

            tau_time = self.convert_type(round(tau_time), float) # convert to proper data type to avoid undesired rpyc netref data type

            '''
            DEFINE SPECIAL TIME INTERVALS FOR EXPERIMENT
            '''
            # padding time to equalize duration of every run
            pad_time = longest_time - tau_time 

            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''
            laser_off1 = self.initial_delay
            laser_off2 = self.singlet_decay + pi_time + tau_time
            # laser_off3 = pad_time + self.rest_time_btw_seqs
            laser_off3 = self.initial_delay + pad_time

            # digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3

            # mw I & Q off windows
            iq_off1 = laser_off1 + self.laser_time + self.singlet_decay
            iq_off2 = (pi_time - self.awg_trig_time) + tau_time + self.readout_time + laser_off3

            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            seq_on = self.Pulser.createSequence()
            seq_off = self.Pulser.createSequence()

            # define sequence structure for laser
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]

            # define sequence structure for digitizer trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]

            # define sequence structure for MW I and Q when MW = ON
            mw_iq_on_seq = [(iq_off1, 0), (self.awg_trig_time, 1), (iq_off2,0)]
            mw_iq_off_seq = [(iq_off1, 0), (self.awg_trig_time, 0), (iq_off2,0)]

            # assign sequences to respective channels for seq_on
            seq_on.setDigital(3, laser_seq) # laser
            seq_on.setDigital(1, dig_clock_seq) # digitizer trigger
            seq_on.setDigital(2, mw_iq_on_seq) # mw_IQ

            # assign sequences to respective channels for seq_off
            seq_off.setDigital(3, laser_seq) # laser
            seq_off.setDigital(1, dig_clock_seq) # digitizer trigger
            seq_off.setDigital(2, mw_iq_off_seq) # mw_IQ

            return seq_on + seq_off

        seqs = self.Pulser.createSequence()

        for tau in params:
            seqs += SingleDiff_T1(tau)

        return seqs

    def DQ(self, params, pi_xy, pi_time_minus, pi_time_plus):
        '''
        Double Quantum Relaxation sequence
        '''
        ## Run a pi pulse, then measure the signal
        ## and reference counts from NV.
        longest_time = self.convert_type(round(max(params)), float)
        pi_time_minus = self.convert_type(round(pi_time_minus), float)
        pi_time_plus = self.convert_type(round(pi_time_plus), float)

        print("LASER INIT TIME = ", self.laser_time)
        ## we can measure the pi time on x and on y.
        ## they should be the same, but they technically
        ## have different offsets on our pulse streamer.

        def SingleDQ(tau_time):
            '''
            CREATE SINGLE T1 SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''

            tau_time = self.convert_type(round(tau_time), float) # convert to proper data type to avoid undesired rpyc netref data type

            '''
            DEFINE SPECIAL TIME INTERVALS FOR EXPERIMENT
            '''
            # padding time to equalize duration of every run
            pad_time = longest_time - tau_time 
            # print("padding time = ", pad_time)
            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''
            laser_off1 = self.initial_delay
            laser_off2 = self.singlet_decay + pi_time_minus + tau_time + pi_time_minus
            laser_off2_plus = self.singlet_decay + pi_time_minus + tau_time + pi_time_plus
            laser_off3 = self.initial_delay + pad_time

            # digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off1_plus = laser_off1 + self.laser_time + laser_off2_plus - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3

            # mw I & Q off windows
            iq_off1 = laser_off1 + self.laser_time + self.singlet_decay 
            iq_off2 = (pi_time_minus - self.awg_trig_time) + tau_time
            iq_off3 = (pi_time_minus - self.awg_trig_time) + self.readout_time + laser_off3
            iq_off3_plus = (pi_time_plus - self.awg_trig_time) + self.readout_time + laser_off3
            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            seq1 = self.Pulser.createSequence()
            seq2 = self.Pulser.createSequence()
            seq3 = self.Pulser.createSequence()
            seq4 = self.Pulser.createSequence() 
            
            # define sequence structure for laser
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
            laser_seq_plus = [(laser_off1, 0), (self.laser_time, 1), (laser_off2_plus, 0), (self.readout_time, 1), (laser_off3, 0)]

            # define sequence structure for digitizer trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
            dig_clock_seq_plus = [(clock_off1_plus, 0), (self.clock_time, 1), (clock_off2, 0)]

            # define sequence structure for MW I and Q when MW = ON
            mw_iq_on_seq1 = [(iq_off1, 0), (self.awg_trig_time, 0), (iq_off2, 0), (self.awg_trig_time, 0), (iq_off3, 0)]

            mw_iq_on_seq2 = [(iq_off1, 0), (self.awg_trig_time, 0), (iq_off2, 0), (self.awg_trig_time, 1), (iq_off3, 0)]

            mw_iq_on_seq3 = [(iq_off1, 0), (self.awg_trig_time, 1), (iq_off2, 0), (self.awg_trig_time, 1), (iq_off3, 0)]

            mw_iq_on_seq4 = [(iq_off1, 0), (self.awg_trig_time, 1), (iq_off2, 0), (self.awg_trig_time, 1), (iq_off3_plus, 0)]

            # assign sequences to respective channels for seq_on
            seq1.setDigital(3, laser_seq) # laser
            seq1.setDigital(1, dig_clock_seq) # digitizer trigger
            seq1.setDigital(2, mw_iq_on_seq1) # MW IQ
            
            # assign sequences to respective channels for seq_off
            seq2.setDigital(3, laser_seq) # laser
            seq2.setDigital(1, dig_clock_seq) # digitizer trigger
            seq2.setDigital(2, mw_iq_on_seq2) # MW IQ

            # assign sequences to respective channels for seq_off
            seq3.setDigital(3, laser_seq) # laser
            seq3.setDigital(1, dig_clock_seq) # digitizer trigger
            seq3.setDigital(2, mw_iq_on_seq3) # MW IQ

            # assign sequences to respective channels for seq_off
            seq4.setDigital(3, laser_seq_plus) # laser
            seq4.setDigital(1, dig_clock_seq_plus) # digitizer trigger
            seq4.setDigital(2, mw_iq_on_seq4) # MW IQ

            return seq1 + seq2 + seq3 + seq4

        seqs = self.Pulser.createSequence()

        for tau in params:
            seqs += SingleDQ(tau)

        return seqs 

    def Ramsey(self, params, pihalf_x, pihalf_y):
        
        '''
        Ramsey pulse sequence.
        MW sequence: pi/2(x) - tau - pi/2(x)

        '''
        longest_time = self.convert_type(round(max(params)), int)
        pihalf_x = self.convert_type(round(pihalf_x), int)
        pihalf_y = self.convert_type(round(pihalf_y), int)

        def SingleRamsey(tau_time):
            '''
            CREATE SINGLE RAMSEY SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''

            tau_time = self.convert_type(round(tau_time), int) # convert to proper data type to avoid undesired rpyc netref data type

            '''
            DEFINE SPECIAL TIME INTERVALS FOR EXPERIMENT
            '''
            # padding time to equalize duration of every run (for different tau durations)
            pad_time = longest_time - tau_time 

            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''

            laser_off1 = self.initial_delay
            laser_off2 = self.singlet_decay + pihalf_x + tau_time + pihalf_x + self.MW_buffer_time
            laser_off3 = 100 + pad_time 
            
            # digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3         

            # mw I & Q off windows (on slightly longer than VSG to ensure it's set)
            iq_off1 = laser_off1 + self.laser_time + self.singlet_decay
            iq_off2 = (pihalf_x - self.awg_trig_time) + tau_time
            iq_off3 = (pihalf_x - self.awg_trig_time) + self.MW_buffer_time + self.readout_time + laser_off3

            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            seq = self.Pulser.createSequence()
            seq_ref = self.Pulser.createSequence()

            # define sequence structure for laser
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
            
            # define sequence structure for digitizer trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
            
            # sequence structure for I & Q MW channels 
            mw_iq_seq = [(iq_off1, 0), (self.awg_trig_time, 1), (iq_off2, 0), (self.awg_trig_time, 1), (iq_off3, 0)]

            # assign sequences to respective channels for seq_on
            seq.setDigital(3, laser_seq) # laser
            seq.setDigital(1, dig_clock_seq) # digitizer trigger
            seq.setDigital(2, mw_iq_seq) # mw_I
            
            # assign sequences to respective channels for seq_off
            seq_ref.setDigital(3, laser_seq) # laser
            seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
            seq_ref.setDigital(2, mw_iq_seq) # mw_I

            return seq + seq_ref

        # concatenate single ODMR sequence "runs" number of times
        seqs = self.Pulser.createSequence()
        
        for tau in params:
            seqs += SingleRamsey(tau)
        
        return seqs

    def Echo(self, params, pihalf_x, pihalf_y, pi_x, pi_y):
        '''
        Spin Echo pulse sequence.
        MW sequence: pi/2(x) - tau - pi(y) - tau - pi/2(x)
        
        '''
        longest_time = self.convert_type(round(max(params)), float)
        pihalf_x = self.convert_type(round(pihalf_x), float)
        pihalf_y = self.convert_type(round(pihalf_y), float)
        pi_x = self.convert_type(round(pi_x), float)
        pi_y = self.convert_type(round(pi_y), float)

        def SingleEcho(tau_time):
            '''
            CREATE SINGLE HAHN-ECHO SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''

            tau_time = self.convert_type(round(tau_time), float) # convert to proper data type to avoid undesired rpyc netref data type
            
            '''
            DEFINE SPECIAL TIME INTERVALS FOR EXPERIMENT
            '''
            # padding time to equalize duration of every run (for different tau durations)
            pad_time = longest_time - tau_time

            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''            
            laser_off1 = self.initial_delay
            laser_off2 = self.singlet_decay + pihalf_x + tau_time + pi_y + tau_time + pihalf_x + self.MW_buffer_time
            # laser_off2 = self.singlet_decay + pihalf_x + tau_time + pi_y + tau_time + pihalf_x + self.MW_buffer_time
            laser_off3 = 100 + pad_time 

            # digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3

            # mw I & Q off windows (on slightly longer than VSG to ensure it's set)
            iq_off1 = laser_off1 + self.laser_time + self.singlet_decay
            iq_off2 = (pihalf_x - self.awg_trig_time) + tau_time
            iq_off3 = (pi_y - self.awg_trig_time) + tau_time
            iq_off4 = (pihalf_x - self.awg_trig_time) + self.MW_buffer_time + self.readout_time + laser_off3

            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            seq = self.Pulser.createSequence()
            seq_ref = self.Pulser.createSequence()

            # define sequence structure for laser
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
            
            # define sequence structure for digitizer trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
            
            # define sequence structure for MW I and Q when MW = ON
            mw_iq_seq = [(iq_off1, 0), (self.awg_trig_time, 1), (iq_off2, 0), (self.awg_trig_time, 1), (iq_off3, 0), (self.awg_trig_time, 1), (iq_off4, 0)]

            # assign sequences to respective channels for seq_on
            seq.setDigital(3, laser_seq) # laser
            seq.setDigital(1, dig_clock_seq) # digitizer trigger
            seq.setDigital(2, mw_iq_seq) # MW IQ
            
            # assign sequences to respective channels for seq_off
            seq_ref.setDigital(3, laser_seq) # laser
            seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
            seq_ref.setDigital(2, mw_iq_seq) # MW IQ

            return seq + seq_ref
        
        # concatenate single ODMR sequence "runs" number of times
        seqs = self.Pulser.createSequence()
        
        for tau in params:
            seqs += SingleEcho(tau)

        return seqs

    def XY4_N(self, params, pulse_axes, pihalf_x, pihalf_y, pi_x, pi_y, n):
        '''
        XY4-N pulse sequence.
        MW sequence: pi/2(x) - tau/2 - (pi(x) - tau - pi(y) - tau - pi(x) - tau - pi(y))^N - tau/2 - pi/2(x, -x)
        '''
        longest_time = self.convert_type(round(max(params)), float)
        pihalf_x = self.convert_type(round(pihalf_x), float)
        pihalf_y = self.convert_type(round(pihalf_y), float)
        pi_x = self.convert_type(round(pi_x), float)
        pi_y = self.convert_type(round(pi_y), float)
        n = self.convert_type(round(n), int)

        def PiPulsesN(axes, tau, N):            
            if axes == 'xy':
                xy4_iq_seq = [(tau/2, 0), (self.awg_trig_time, 1), 
                              ((pi_x - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_x - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau/2, 0)]
                mw_IQ = (xy4_iq_seq)*N
                
            elif axes == 'yy':
                yy4_iq_seq = [(tau/2, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau/2, 0)]
                mw_IQ = (yy4_iq_seq)*N

            return mw_IQ

        def SingleXY4(tau):
            '''
            CREATE SINGLE HAHN-ECHO SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''

            tau = self.convert_type(round(tau), float) # convert to proper data type to avoid undesired rpyc netref data type

            '''
            DEFINE SPECIAL TIME INTERVALS FOR EXPERIMENT

            pad_time = padding time to equalize duration of every run (for different tau durations)
            '''
            pad_time = longest_time - tau 
            # NOTICE: change if using PiHalf['y'] to pihalf_y
            xy4_time = 2*pihalf_x + (2*(tau/2) + 2*pi_x + 2*pi_y + 3*tau)*n
            # xy4_time = 2*pihalf_x + (2*tau + 2*pi_x + 2*pi_y + 3*(2*tau))*n

            yy4_time = 2*pihalf_x + (2*(tau/2) + 4*pi_y + 3*tau)*n
            # yy4_time = 2*pihalf_x + (2*tau + 4*pi_y + 3*(2*tau))*n

            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''            
            laser_off1 = self.initial_delay

            if pulse_axes == 'xy':
                laser_off2 = self.singlet_decay + xy4_time + self.MW_buffer_time
            else:
                laser_off2 = self.singlet_decay + yy4_time + self.MW_buffer_time

            laser_off3 = 100 + pad_time 
            
            # digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3         

            # mw I & Q off windows (on slightly longer than VSG to ensure it's set)
            iq_off_start = laser_off1 + self.laser_time + self.singlet_decay
            iq_off_end =  (pihalf_x - self.awg_trig_time) + self.MW_buffer_time + self.readout_time + laser_off3
            
            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            seq = self.Pulser.createSequence()
            seq_ref = self.Pulser.createSequence()

            # define sequence structure for laser
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
            
            # define sequence structure for digitizer trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
            
            # sequence structure for I & Q MW channels 
            mw_iq_seq = [(iq_off_start, 0), (self.awg_trig_time, 1), (pihalf_x - self.awg_trig_time, 0)] + PiPulsesN(pulse_axes, tau, n)[0] + [(self.awg_trig_time, 1), (iq_off_end, 0)]
            
            # assign sequences to respective channels for seq_on
            seq.setDigital(3, laser_seq) # laser
            seq.setDigital(1, dig_clock_seq) # digitizer trigger
            seq.setDigital(2, mw_iq_seq) # MW IQ
            
            # assign sequences to respective channels for seq_off
            seq_ref.setDigital(3, laser_seq) # laser
            seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
            seq_ref.setDigital(2, mw_iq_seq) # MW IQ
            
            return seq + seq_ref

        # concatenate single ODMR sequence "runs" number of times
        seqs = self.Pulser.createSequence()
        
        for tau in params:
            seqs += SingleXY4(tau)
        
        return seqs

    def XY8_N(self, params, pulse_axes, pihalf_x, pihalf_y, pi_x, pi_y, n):
        '''
        XY8-N pulse sequence.
        MW sequence: pi/2(x) - (tau/2 - pi(x) - tau - pi(y) - tau - pi(x) - tau... - pi(y) - tau/2)^N - pi/2(x, -x)

        '''
        longest_time = self.convert_type(round(max(params)), float)
        pihalf_x = self.convert_type(round(pihalf_x), float)
        pihalf_y = self.convert_type(round(pihalf_y), float)
        pi_x = self.convert_type(round(pi_x), float)
        pi_y = self.convert_type(round(pi_y), float)
        n = self.convert_type(round(n), int)
        
        def PiPulsesN(axes, tau, N):
            if axes == 'xy':
                # xy4_I_seq = [self.Pi('x', pi_x)[0], (tau, self.IQ0[0]), self.Pi('y', pi_y)[0], (tau, self.IQ0[0]), self.Pi('x', pi_x)[0], (tau, self.IQ0[0]), self.Pi('y', pi_y)[0]]
                xy8_iq_seq = [(tau/2, 0), (self.awg_trig_time, 1), 
                              ((pi_x - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_x - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_x - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_x - self.awg_trig_time) + tau/2, 0)]

                mw_IQ = (xy8_iq_seq)*N
                
            elif axes == 'yy':
                yy8_iq_seq = [(tau/2, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau/2, 0)]

                mw_IQ = (yy8_iq_seq)*N
            
            return mw_IQ
        
        def SingleXY8(tau):
            '''
            CREATE SINGLE HAHN-ECHO SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''

            tau = self.convert_type(round(tau), float) # convert to proper data type to avoid undesired rpyc netref data type

            '''
            DEFINE SPECIAL TIME INTERVALS FOR EXPERIMENT

            pad_time = padding time to equalize duration of every run (for different tau durations)
            '''
            pad_time = longest_time - tau 

            # NOTICE: change if using PiHalf['y'] to pihalf_y
            xy8_time = 2*pihalf_x + ((tau/2) + 4*pi_x + 4*pi_y + 7*tau + (tau/2))*n
            yy8_time = 2*pihalf_x + ((tau/2) + 8*pi_y + 7*tau + (tau/2))*n
            # xy8_time = 2*pihalf_x + (tau + 4*pi_x + 4*pi_y + 7*(2*tau) + tau)*n
            # yy8_time = 2*pihalf_x + (tau + 8*pi_y + 7*(2*tau) + tau)*n

            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''            
            laser_off1 = self.initial_delay
            if pulse_axes == 'xy':
                laser_off2 = self.singlet_decay + xy8_time + self.MW_buffer_time
            else:
                laser_off2 = self.singlet_decay + yy8_time + self.MW_buffer_time
            laser_off3 = 100 + pad_time

            # digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3           

            # mw I & Q off windows
            iq_off_start = laser_off1 + self.laser_time + self.singlet_decay
            iq_off_end = (pihalf_x - self.awg_trig_time) + self.MW_buffer_time + self.readout_time + laser_off3
            
            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            seq = self.Pulser.createSequence()
            seq_ref = self.Pulser.createSequence()

            # define sequence structure for laser
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
            
            # define sequence structure for digitizer trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
            
            # sequence structure for I & Q MW channels 
            mw_iq_seq = [(iq_off_start, 0), (self.awg_trig_time, 1), (pihalf_x - self.awg_trig_time, 0)] + PiPulsesN(pulse_axes, tau, n) + [(self.awg_trig_time, 1), (iq_off_end, 0)]
            
            # assign sequences to respective channels for seq_on
            seq.setDigital(3, laser_seq) # laser
            seq.setDigital(1, dig_clock_seq) # digitizer trigger
            seq.setDigital(2, mw_iq_seq) # MW IQ
            
            # assign sequences to respective channels for seq_off
            seq_ref.setDigital(3, laser_seq) # laser
            seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
            seq_ref.setDigital(2, mw_iq_seq) # MW IQ

            return seq + seq_ref

        # concatenate single XY8 sequence "runs" number of times
        seqs = self.Pulser.createSequence()
        
        for tau in params:
            seqs += SingleXY8(tau)

        return seqs
    
    def CPMG_N(self, params, pulse_axis, pihalf_x, pihalf_y, pi_x, pi_y, n):
        '''
        CPMG-N pulse sequence.
        MW sequence: pi/2(x) - tau/2N - (pi(y) - tau/N - ...)^N - tau/2N - pi/2(x, -x)

        '''
        longest_time = self.convert_type(round(max(params)), float)
        pihalf_x = self.convert_type(round(pihalf_x), float)
        pihalf_y = self.convert_type(round(pihalf_y), float)
        pi_x = self.convert_type(round(pi_x), float)
        pi_y = self.convert_type(round(pi_y), float)
        n = self.convert_type(round(n), int)

        def PiPulsesN(axis, tau, N):
            if axis in ["X","x"]:
                CPMG_seq = [(self.awg_trig_time, 1), ((pi_x - self.awg_trig_time) + tau, 0)]
                mw_IQ = CPMG_seq*(N-1) + [(self.awg_trig_time, 1), ((pi_x - self.awg_trig_time) + tau/2, 0)]

            elif axis in ["Y","y"]:
                CPMG_seq = [(self.awg_trig_time, 1), ((pi_y - self.awg_trig_time) + tau, 0)]
                mw_IQ = CPMG_seq*(N-1) + [(self.awg_trig_time, 1), ((pi_y - self.awg_trig_time) + tau/2, 0)]

            return mw_IQ

        def SingleCPMG(tau):
            '''
            CREATE SINGLE HAHN-ECHO SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''
            tau = self.convert_type(round(tau), float) # convert to proper data type to avoid undesired rpyc netref data type
            '''
            DEFINE SPECIAL TIME INTERVALS FOR EXPERIMENT

            pad_time = padding time to equalize duration of every run (for different tau durations)
            '''
            pad_time = longest_time - tau 
            
            if pulse_axis in ["X","x"]:
                cpmg_time = pihalf_x + tau/2 + (pi_x + tau)*(n-1) + pi_x + tau/2 + pihalf_x
            elif pulse_axis in ["Y","y"]:
                cpmg_time = pihalf_x + tau/2 + (pi_y + tau)*(n-1) + pi_y + tau/2 + pihalf_x

            # if pulse_axis == 'X':
            #     cpmg_time = pihalf_x + tau + (pi_x + 2*tau)*(n-1) + pi_x + tau + pihalf_x
            # elif pulse_axis == 'Y':
            #     cpmg_time = pihalf_x + tau + (pi_y + 2*tau)*(n-1) + pi_y + tau + pihalf_x
            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''       
            laser_off1 = self.initial_delay     
            laser_off2 = self.singlet_decay + cpmg_time + self.MW_buffer_time
            laser_off3 = 100 + pad_time #+ self.rest_time_btw_seqs
            
            # digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3

            # mw I & Q off windows
            iq_off1 = self.laser_time + self.singlet_decay
            iq_off2 = (pihalf_x - self.awg_trig_time) + self.MW_buffer_time + self.readout_time + laser_off3
            
            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            seq = self.Pulser.createSequence()
            seq_ref = self.Pulser.createSequence()

            # define sequence structure for laser
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
            
            # define sequence structure for digitizer trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
            
            # sequence structure for I & Q MW channels 
            mw_iq_seq = [(iq_off1, 0), (self.awg_trig_time, 1), ((pihalf_x - self.awg_trig_time) + tau/2, 0)] + PiPulsesN(pulse_axis, tau, n) + [(self.awg_trig_time, 1), (iq_off2, 0)]
            
            # assign sequences to respective channels for seq_on
            seq.setDigital(3, laser_seq) # laser
            seq.setDigital(1, dig_clock_seq) # digitizer trigger
            seq.setDigital(2, mw_iq_seq) # MW IQ
            
            # assign sequences to respective channels for seq_off
            seq_ref.setDigital(3, laser_seq) # laser
            seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
            seq_ref.setDigital(2, mw_iq_seq) # MW IQ

            return seq + seq_ref

        # concatenate single CPMG sequence "runs" number of times
        seqs = self.Pulser.createSequence()
        
        for tau in params:
            seqs += SingleCPMG(tau)
        
        return seqs

    def DEER(self, pihalf_x, pihalf_y, pi_x, pi_y, tau, num_freqs):
        '''
        DEER pulse sequence.
        MW sequence: pi/2(x) - tau - pi(y) - tau - pi/2(x)

        '''
        pihalf_x = self.convert_type(round(pihalf_x), float)
        pihalf_y = self.convert_type(round(pihalf_y), float)
        pi_x = self.convert_type(round(pi_x), float)
        pi_y = self.convert_type(round(pi_y), float)
        tau = self.convert_type(round(tau), float)

        def SingleDEER():
            '''
            CREATE SINGLE DEER SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''

            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''            
            # laser_off = self.initial_delay + pihalf_x + tau + pi_y + tau + pihalf_x + self.MW_buffer_time
            # laser_on = self.laser_time
            laser_off1 = self.initial_delay
            laser_off2 = self.singlet_decay + pihalf_x + tau + pi_y + tau + pihalf_x + self.MW_buffer_time
            laser_off3 = self.initial_delay + 1000 # constant 1 us wait period between seqs

            # digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3

            # mw I & Q off windows 
            iq_off1 = laser_off1 + self.laser_time + self.singlet_decay
            iq_off2 = (pihalf_x - self.awg_trig_time) + tau
            iq_off3 = (pi_y - self.awg_trig_time) + tau
            iq_off4 = (pihalf_x - self.awg_trig_time) + self.MW_buffer_time + self.readout_time + laser_off3

            awg_off1 = 15 + iq_off1 + pihalf_x + self.awg_pulse_delay # additional 10 ns delay at beginning to offset entire AWG pulse seq --> electron spin pulses right after NV pulses
            awg_off2 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pi_y + self.awg_pulse_delay
            awg_off3 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pihalf_x + iq_off4 - 15

            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            dark_seq = self.Pulser.createSequence()
            dark_seq_ref = self.Pulser.createSequence()
            echo_seq = self.Pulser.createSequence()
            echo_seq_ref = self.Pulser.createSequence()

            # define sequence structure for laser
            # laser_seq = [(laser_off, 0), (laser_on, 1)]
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
            
            # define sequence structure for digitizer trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
            
            # sequence structure for I & Q MW channels on SRS SG396
            mw_iq_seq = [(iq_off1, 0), (self.awg_trig_time, 1), (iq_off2, 0), (self.awg_trig_time, 1), (iq_off3, 0), (self.awg_trig_time, 1), (iq_off4, 0)]

            # sequence structure for I & Q MW channels (MW off)
            awg_seq = [(awg_off1, 0), (self.awg_trig_time, 1), (awg_off2, 0), (self.awg_trig_time, 1), (awg_off3, 0)]
            awg_ref_seq = [(laser_off1 + self.laser_time + laser_off2 + self.readout_time + laser_off3, 0)] # off the entire time

            # assign sequences to respective channels for seq_on
            dark_seq.setDigital(3, laser_seq) # laser
            dark_seq.setDigital(1, dig_clock_seq) # digitizer trigger
            dark_seq.setDigital(4, awg_seq)
            dark_seq.setDigital(2, mw_iq_seq) # mw_IQ
            
            # assign sequences to respective channels for seq_off
            dark_seq_ref.setDigital(3, laser_seq) # laser
            dark_seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
            dark_seq_ref.setDigital(4, awg_seq)
            dark_seq_ref.setDigital(2, mw_iq_seq) # mw_IQ

            # assign sequences to respective channels for seq_on
            echo_seq.setDigital(3, laser_seq) # laser
            echo_seq.setDigital(1, dig_clock_seq) # digitizer trigger
            echo_seq.setDigital(4, awg_ref_seq)
            echo_seq.setDigital(2, mw_iq_seq) # mw_IQ

            # assign sequences to respective channels for seq_off
            echo_seq_ref.setDigital(3, laser_seq) # laser
            echo_seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
            echo_seq_ref.setDigital(4, awg_ref_seq)
            echo_seq_ref.setDigital(2, mw_iq_seq) # mw_IQ

            return dark_seq + dark_seq_ref + echo_seq + echo_seq_ref
        
        seqs = self.Pulser.createSequence()

        for i in range(num_freqs):
            seqs += SingleDEER()

        # return SingleDEER()
        return seqs 
    
    def DEER_CD(self, pihalf_x, pihalf_y, pi_x, pi_y, tau, num_freqs):
        '''
        DEER constant drive pulse sequence.
        MW sequence: pi/2(x) - tau - pi(y) - tau - pi/2(x)

        '''
        pihalf_x = self.convert_type(round(pihalf_x), float)
        pihalf_y = self.convert_type(round(pihalf_y), float)
        pi_x = self.convert_type(round(pi_x), float)
        pi_y = self.convert_type(round(pi_y), float)
        tau = self.convert_type(round(tau), float)

        def SingleDEERCD():
            '''
            CREATE SINGLE DEER SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''

            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''            
            # laser_off = self.initial_delay + pihalf_x + tau + pi_y + tau + pihalf_x + self.MW_buffer_time
            # laser_on = self.laser_time
            laser_off1 = self.initial_delay
            laser_off2 = self.singlet_decay + pihalf_x + tau + pi_y + tau + pihalf_x + self.MW_buffer_time
            laser_off3 = self.initial_delay + 1000 # constant 1 us wait period between seqs

            # digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3

            # mw I & Q off windows 
            iq_off1 = laser_off1 + self.laser_time + self.singlet_decay
            iq_off2 = (pihalf_x - self.awg_trig_time) + tau
            iq_off3 = (pi_y - self.awg_trig_time) + tau
            iq_off4 = (pihalf_x - self.awg_trig_time) + self.MW_buffer_time + self.readout_time + laser_off3

            # awg_off1 = - self.initial_delay + iq_off1 # additional initial delay at beginning to offset entire AWG pulse seq
            # awg_off2 = - self.awg_trig_time + pihalf_x + iq_off2 + pi_y + iq_off3 + pihalf_x + iq_off4 + self.initial_delay

            awg_off1 = 15 + iq_off1 + pihalf_x + self.awg_pulse_delay # additional initial delay at beginning to offset entire AWG pulse seq
            awg_off2 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pi_y + self.awg_pulse_delay
            awg_off3 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pihalf_x + iq_off4 - 15

            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            dark_seq = self.Pulser.createSequence()
            dark_seq_ref = self.Pulser.createSequence()
            echo_seq = self.Pulser.createSequence()
            echo_seq_ref = self.Pulser.createSequence()
            cd_seq = self.Pulser.createSequence()
            cd_seq_ref = self.Pulser.createSequence()

            # define sequence structure for laser
            # laser_seq = [(laser_off, 0), (laser_on, 1)]
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
            
            # define sequence structure for digitizer trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
            
            # sequence structure for I & Q MW channels on SRS SG396
            mw_iq_seq = [(iq_off1, 0), (self.awg_trig_time, 1), (iq_off2, 0), (self.awg_trig_time, 1), (iq_off3, 0), (self.awg_trig_time, 1), (iq_off4, 0)]
            
            # sequence structure for I & Q MW channels (MW off)
            awg_seq = [(awg_off1, 0), (self.awg_trig_time, 1), (awg_off2, 0), (self.awg_trig_time, 1), (awg_off3, 0)]
            awg_ref_seq = [(laser_off1 + self.laser_time + laser_off2 + self.readout_time + laser_off3, 0)] # off the entire time

            # sequence structure for I & Q MW channels (MW off)
            awg_cd_seq = [(awg_off1, 0), (self.awg_trig_time, 1), (awg_off2, 0)]
            awg_cd_ref_seq = [(laser_off1 + self.laser_time + laser_off2 + self.readout_time + laser_off3, 0)] # off the entire time

            # assign sequences to respective channels for seq_on
            dark_seq.setDigital(3, laser_seq) # laser
            dark_seq.setDigital(1, dig_clock_seq) # digitizer trigger
            dark_seq.setDigital(4, awg_seq)
            dark_seq.setDigital(2, mw_iq_seq) # mw_IQ
            
            # assign sequences to respective channels for seq_off
            dark_seq_ref.setDigital(3, laser_seq) # laser
            dark_seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
            dark_seq_ref.setDigital(4, awg_seq)
            dark_seq_ref.setDigital(2, mw_iq_seq) # mw_IQ

            # assign sequences to respective channels for seq_on
            echo_seq.setDigital(3, laser_seq) # laser
            echo_seq.setDigital(1, dig_clock_seq) # digitizer trigger
            echo_seq.setDigital(4, awg_ref_seq)
            echo_seq.setDigital(2, mw_iq_seq) # mw_IQ

            # assign sequences to respective channels for seq_off
            echo_seq_ref.setDigital(3, laser_seq) # laser
            echo_seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
            echo_seq_ref.setDigital(4, awg_ref_seq)
            echo_seq_ref.setDigital(2, mw_iq_seq) # mw_IQ
            
            # assign sequences to respective channels for seq_on
            cd_seq.setDigital(3, laser_seq) # laser
            cd_seq.setDigital(1, dig_clock_seq) # digitizer trigger
            cd_seq.setDigital(4, awg_cd_seq)
            cd_seq.setDigital(2, mw_iq_seq) # mw_IQ
            
            # assign sequences to respective channels for seq_off
            cd_seq_ref.setDigital(3, laser_seq) # laser
            cd_seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
            cd_seq_ref.setDigital(4, awg_cd_ref_seq)
            cd_seq_ref.setDigital(2, mw_iq_seq) # mw_IQ

            return echo_seq + echo_seq_ref + dark_seq + dark_seq_ref + cd_seq + cd_seq_ref
        
        seqs = self.Pulser.createSequence()

        for i in range(num_freqs):
            seqs += SingleDEERCD()

        # return SingleDEER()
        return seqs 

    def DEER_Rabi(self, pihalf_x, pihalf_y, pi_x, pi_y, tau, num_pts):
        '''
        DEER pulse sequence.
        MW sequence: pi/2(x) - tau - pi(y) - tau - pi/2(x)

        '''
        pihalf_x = self.convert_type(round(pihalf_x), float)
        pihalf_y = self.convert_type(round(pihalf_y), float)
        pi_x = self.convert_type(round(pi_x), float)
        pi_y = self.convert_type(round(pi_y), float)
        tau = self.convert_type(round(tau), float)

        def SingleDEERRabi():
            '''
            CREATE SINGLE DEER SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''

            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''            
            # laser_off = self.initial_delay + pihalf_x + tau + pi_y + tau + pihalf_x + self.MW_buffer_time
            # laser_on = self.laser_time
            laser_off1 = self.initial_delay
            laser_off2 = self.singlet_decay + pihalf_x + tau + pi_y + tau + pihalf_x + self.MW_buffer_time
            laser_off3 = self.initial_delay + 1000

            # digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3

            # mw I & Q off windows 
            iq_off1 = laser_off1 + self.laser_time + self.singlet_decay
            iq_off2 = (pihalf_x - self.awg_trig_time) + tau
            iq_off3 = (pi_y - self.awg_trig_time) + tau
            iq_off4 = (pihalf_x - self.awg_trig_time) + self.MW_buffer_time + self.readout_time + laser_off3

            awg_off1 = 15 + iq_off1 + pihalf_x + self.awg_pulse_delay # additional initial delay at beginning to offset entire AWG pulse seq
            awg_off2 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pi_y + self.awg_pulse_delay
            awg_off3 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pihalf_x + iq_off4 - 15

            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            dark_seq = self.Pulser.createSequence()
            dark_seq_ref = self.Pulser.createSequence()
            echo_seq = self.Pulser.createSequence()
            echo_seq_ref = self.Pulser.createSequence()

            # define sequence structure for laser
            # laser_seq = [(laser_off, 0), (laser_on, 1)]
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
            
            # define sequence structure for digitizer trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
            
            # sequence structure for I & Q MW channels on SRS SG396
            mw_iq_seq = [(iq_off1, 0), (self.awg_trig_time, 1), (iq_off2, 0), (self.awg_trig_time, 1), (iq_off3, 0), (self.awg_trig_time, 1), (iq_off4, 0)]

            # sequence structure for I & Q MW channels (MW off)
            awg_seq = [(awg_off1, 0), (self.awg_trig_time, 1), (awg_off2, 0), (self.awg_trig_time, 1), (awg_off3, 0)]
            awg_ref_seq = [(laser_off1 + self.laser_time + laser_off2 + self.readout_time + laser_off3, 0)] # off the entire time

            # assign sequences to respective channels for seq_on
            dark_seq.setDigital(3, laser_seq) # laser
            dark_seq.setDigital(1, dig_clock_seq) # digitizer trigger
            dark_seq.setDigital(4, awg_seq)
            dark_seq.setDigital(2, mw_iq_seq) # MW IQ
            
            # assign sequences to respective channels for seq_off
            dark_seq_ref.setDigital(3, laser_seq) # laser
            dark_seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
            dark_seq_ref.setDigital(4, awg_seq)
            dark_seq_ref.setDigital(2, mw_iq_seq) # MW IQ

            # assign sequences to respective channels for seq_on
            echo_seq.setDigital(3, laser_seq) # laser
            echo_seq.setDigital(1, dig_clock_seq) # digitizer trigger
            echo_seq.setDigital(4, awg_ref_seq)
            echo_seq.setDigital(2, mw_iq_seq) # MW IQ
            
            # assign sequences to respective channels for seq_off
            echo_seq_ref.setDigital(3, laser_seq) # laser
            echo_seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
            echo_seq_ref.setDigital(4, awg_ref_seq)
            echo_seq_ref.setDigital(2, mw_iq_seq) # MW IQ

            return dark_seq + dark_seq_ref + echo_seq + echo_seq_ref
        
        seqs = self.Pulser.createSequence()

        for i in range(num_pts):
            seqs += SingleDEERRabi()

        # return SingleDEER()
        return seqs 

    def DEER_FID(self, params, pihalf_x, pihalf_y, pi_x, pi_y, n):
            '''
            DEER pulse sequence.
            MW sequence: pi/2(x) - tau - pi(y) - tau - pi/2(x)

            '''
            longest_time = self.convert_type(round(max(params)), float)
            pihalf_x = self.convert_type(round(pihalf_x), float)
            pihalf_y = self.convert_type(round(pihalf_y), float)
            pi_x = self.convert_type(round(pi_x), float)
            pi_y = self.convert_type(round(pi_y), float)
            n = self.convert_type(round(n), int)

            def PiPulsesN(tau, N):
                CPMG_I_seq = [self.Pi('y', pi_y)[0], (2*tau, self.IQ0[0])]
                CPMG_Q_seq = [self.Pi('y', pi_y)[1], (2*tau, self.IQ0[1])]
                mw_I = CPMG_I_seq*(N-1) + [self.Pi('y', pi_y)[0]]
                mw_Q = CPMG_Q_seq*(N-1) + [self.Pi('y', pi_y)[1]]
                
                return mw_I, mw_Q
            
            def AWGPulsesN(tau, N):
                '''
                Function to return all the AWG trigger pulses corresponding to the pi pulses in the SRS CPMG sequence.
                An initial pulse after the first pi/2 pulse is left off with the option to set it on.
                '''
                return [(self.awg_trig_time, 1), (tau - self.awg_pulse_delay - self.awg_trig_time + pi_y + self.awg_pulse_delay, 0)] + \
                [(self.awg_trig_time, 1), (2*tau - self.awg_pulse_delay - self.awg_trig_time + pi_y + self.awg_pulse_delay, 0)]*(N-1) + [(self.awg_trig_time, 1)]
                # awg_off2 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pi_y + self.awg_pulse_delay

            def SingleDEERFID(tau):
                '''
                CREATE SINGLE DEER SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
                '''
                tau = self.convert_type(round(tau), float)
                pad_time = longest_time - tau
                '''
                DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
                '''            
                
                cpmg_time = pihalf_x + tau + (pi_y + 2*tau)*(n-1) + pi_y + tau + pihalf_x
            
                laser_off1 = self.initial_delay
                laser_off2 = self.singlet_decay + cpmg_time + self.MW_buffer_time
                # laser_off2 = self.singlet_decay + pihalf_x + tau + pi_y + tau + pihalf_x + self.MW_buffer_time
                laser_off3 = self.initial_delay + pad_time
                # laser_off3 = 6000

                # digitizer trigger timing
                clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
                clock_off2 = - self.trig_spot + self.readout_time + laser_off3

                # mw I & Q off windows 
                iq_off1 = laser_off1 + self.laser_time + self.singlet_decay
                iq_off2 = (pihalf_x - self.awg_trig_time) + tau
                iq_off3 = (pi_y - self.awg_trig_time) + tau
                iq_off4 = (pihalf_x - self.awg_trig_time) + self.MW_buffer_time + self.readout_time + laser_off3

                # awg_off1 = - self.initial_delay + iq_off1 + pihalf_x + self.awg_pulse_delay # additional initial delay at beginning to offset entire AWG pulse seq
                # # awg_off2 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pi_y + self.awg_pulse_delay
                # # awg_off3 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pihalf_x + iq_off4 + self.initial_delay
                # awg_off2 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pihalf_x + iq_off4 + self.initial_delay

                awg_off1 = 15 + iq_off1 + pihalf_x + self.awg_pulse_delay # additional initial delay at beginning to offset entire AWG pulse seq
                # awg_off2 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pi_y + self.awg_pulse_delay
                # awg_off3 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pihalf_x + iq_off4 + self.initial_delay
                awg_off2 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pihalf_x + iq_off4 - 15

                '''
                CONSTRUCT PULSE SEQUENCE
                '''
                # create sequence objects for MW on and off blocks
                dark_seq = self.Pulser.createSequence()
                dark_seq_ref = self.Pulser.createSequence()
                echo_seq = self.Pulser.createSequence()
                echo_seq_ref = self.Pulser.createSequence()

                # define sequence structure for laser
                # laser_seq = [(laser_off, 0), (laser_on, 1)]
                laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
                
                # define sequence structure for digitizer trigger
                dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
                
                # sequence structure for I & Q MW channels on SRS SG396
                mw_iq_seq = [(iq_off1, 0), (self.awg_trig_time, 1), (iq_off2, 0), (self.awg_trig_time, 1), (iq_off3, 0), (self.awg_trig_time, 1), (iq_off4, 0)]

                # sequence structure for I & Q MW channels (MW off)
                awg_seq = [(awg_off1, 0)] + AWGPulsesN(tau, n) + [(awg_off2, 0)]
                awg_ref_seq = [(laser_off1 + self.laser_time + laser_off2 + self.readout_time + laser_off3, 0)] # off the entire time

                # assign sequences to respective channels for seq_on
                dark_seq.setDigital(3, laser_seq) # laser
                dark_seq.setDigital(1, dig_clock_seq) # digitizer trigger
                dark_seq.setDigital(4, awg_seq)
                dark_seq.setDigital(2, mw_iq_seq) # mw_IQ

                # assign sequences to respective channels for seq_off
                dark_seq_ref.setDigital(3, laser_seq) # laser
                dark_seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
                dark_seq_ref.setDigital(4, awg_seq)
                dark_seq_ref.setDigital(2, mw_iq_seq) # mw_IQ

                # assign sequences to respective channels for seq_on
                echo_seq.setDigital(3, laser_seq) # laser
                echo_seq.setDigital(1, dig_clock_seq) # digitizer trigger
                echo_seq.setDigital(4, awg_ref_seq)
                echo_seq.setDigital(2, mw_iq_seq) # mw_IQ
                
                # assign sequences to respective channels for seq_off
                echo_seq_ref.setDigital(3, laser_seq) # laser
                echo_seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
                echo_seq_ref.setDigital(4, awg_ref_seq)
                echo_seq_ref.setDigital(2, mw_iq_seq) # mw_IQ

                return dark_seq + dark_seq_ref + echo_seq + echo_seq_ref
            
            seqs = self.Pulser.createSequence()

            for t in params:
                seqs += SingleDEERFID(t)

            # return SingleDEER()
            return seqs 
    
    def DEER_FID_CD(self, params, pihalf_x, pihalf_y, pi_x, pi_y, n):
            '''
            DEER pulse sequence.
            MW sequence: pi/2(x) - tau - pi(y) - tau - pi/2(x)
            '''
            longest_time = self.convert_type(round(max(params)), float)
            pihalf_x = self.convert_type(round(pihalf_x), float)
            pihalf_y = self.convert_type(round(pihalf_y), float)
            pi_x = self.convert_type(round(pi_x), float)
            pi_y = self.convert_type(round(pi_y), float)
            n = self.convert_type(round(n), int)

            # print("TAU TIMES = ", params)

            def PiPulsesN(tau, N):
                CPMG_I_seq = [self.Pi('y', pi_y)[0], (2*tau, self.IQ0[0])]
                CPMG_Q_seq = [self.Pi('y', pi_y)[1], (2*tau, self.IQ0[1])]
                mw_I = CPMG_I_seq*(N-1) + [self.Pi('y', pi_y)[0]]
                mw_Q = CPMG_Q_seq*(N-1) + [self.Pi('y', pi_y)[1]]
                
                return mw_I, mw_Q
            
            def AWGPulsesN(tau, N):
                '''
                Function to return all the AWG trigger pulses corresponding to the pi pulses in the SRS CPMG sequence.
                An initial pulse after the first pi/2 pulse is left off with the option to set it on.
                '''
                return [(self.awg_trig_time, 1), (tau - self.awg_pulse_delay - self.awg_trig_time + pi_y + self.awg_pulse_delay, 0)] + \
                [(self.awg_trig_time, 1), (2*tau - self.awg_pulse_delay - self.awg_trig_time + pi_y + self.awg_pulse_delay, 0)]*(N-1) + [(self.awg_trig_time, 1)]
                # awg_off2 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pi_y + self.awg_pulse_delay

            def SingleDEERFIDCD(tau):
                '''
                CREATE SINGLE DEER SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
                '''
                tau = self.convert_type(round(tau), float)
                pad_time = longest_time - tau
                '''
                DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
                '''            
                
                cpmg_time = pihalf_x + tau + (pi_y + 2*tau)*(n-1) + pi_y + tau + pihalf_x
            
                laser_off1 = self.initial_delay
                laser_off2 = self.singlet_decay + cpmg_time + self.MW_buffer_time
                # laser_off2 = self.singlet_decay + pihalf_x + tau + pi_y + tau + pihalf_x + self.MW_buffer_time
                laser_off3 = self.initial_delay + pad_time
                # laser_off3 = 6000

                # digitizer trigger timing
                clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
                clock_off2 = - self.trig_spot + self.readout_time + laser_off3

                # mw I & Q off windows 
                iq_off1 = laser_off1 + self.laser_time + self.singlet_decay
                iq_off2 = (pihalf_x - self.awg_trig_time) + tau
                iq_off3 = (pi_y - self.awg_trig_time) + tau
                iq_off4 = (pihalf_x - self.awg_trig_time) + self.MW_buffer_time + self.readout_time + laser_off3

                awg_off1 = 15 + iq_off1 + pihalf_x + self.awg_pulse_delay # additional initial delay at beginning to offset entire AWG pulse seq
                # awg_off2 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pi_y + self.awg_pulse_delay
                # awg_off3 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pihalf_x + iq_off4 + self.initial_delay
                awg_off2 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pihalf_x + iq_off4 - 15

                '''
                CONSTRUCT PULSE SEQUENCE
                '''
                # create sequence objects for MW on and off blocks
                dark_seq = self.Pulser.createSequence()
                dark_seq_ref = self.Pulser.createSequence()
                echo_seq = self.Pulser.createSequence()
                echo_seq_ref = self.Pulser.createSequence()
                cd_seq = self.Pulser.createSequence()
                cd_seq_ref = self.Pulser.createSequence()

                # define sequence structure for laser
                # laser_seq = [(laser_off, 0), (laser_on, 1)]
                laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
                
                # define sequence structure for digitizer trigger
                dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
                
                # sequence structure for I & Q MW channels on SRS SG396
                mw_iq_seq = [(iq_off1, 0), (self.awg_trig_time, 1), (iq_off2, 0), (self.awg_trig_time, 1), (iq_off3, 0), (self.awg_trig_time, 1), (iq_off4, 0)]

                # sequence structure for I & Q MW channels (MW off)
                awg_seq = [(awg_off1, 0)] + AWGPulsesN(tau, n) + [(awg_off2, 0)]
                awg_ref_seq = [(laser_off1 + self.laser_time + laser_off2 + self.readout_time + laser_off3, 0)] # off the entire time

                # assign sequences to respective channels for seq_on
                dark_seq.setDigital(3, laser_seq) # laser
                dark_seq.setDigital(1, dig_clock_seq) # digitizer trigger
                dark_seq.setDigital(4, awg_seq)
                dark_seq.setDigital(2, mw_iq_seq) # MW IQ 
                
                # assign sequences to respective channels for seq_off
                dark_seq_ref.setDigital(3, laser_seq) # laser
                dark_seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
                dark_seq_ref.setDigital(4, awg_seq)
                dark_seq_ref.setDigital(2, mw_iq_seq) # MW IQ 

                # assign sequences to respective channels for seq_on
                echo_seq.setDigital(3, laser_seq) # laser
                echo_seq.setDigital(1, dig_clock_seq) # digitizer trigger
                echo_seq.setDigital(4, awg_ref_seq)
                echo_seq.setDigital(2, mw_iq_seq) # MW IQ 
                
                # assign sequences to respective channels for seq_off
                echo_seq_ref.setDigital(3, laser_seq) # laser
                echo_seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
                echo_seq_ref.setDigital(4, awg_ref_seq)
                echo_seq_ref.setDigital(2, mw_iq_seq) # MW IQ 

                # assign sequences to respective channels for seq_on
                cd_seq.setDigital(3, laser_seq) # laser
                cd_seq.setDigital(1, dig_clock_seq) # digitizer trigger
                cd_seq.setDigital(4, awg_seq)
                cd_seq.setDigital(2, mw_iq_seq) # MW IQ 

                # assign sequences to respective channels for seq_off
                cd_seq_ref.setDigital(3, laser_seq) # laser
                cd_seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
                cd_seq_ref.setDigital(4, awg_seq)
                cd_seq_ref.setDigital(2, mw_iq_seq) # MW IQ 

                return dark_seq + dark_seq_ref + echo_seq + echo_seq_ref + cd_seq + cd_seq_ref
            
            seqs = self.Pulser.createSequence()

            for t in params:
                seqs += SingleDEERFIDCD(t)

            # return SingleDEER()
            return seqs
    
    def DEER_Corr_T1(self, params, tau, pihalf_x, pihalf_y, pi_x, pi_y, dark_pi):
            '''
            Surface electron T1 pulse sequence.
            MW sequence: pi/2(x) - tau - pi(y) - tau - pi/2(x)

            '''
            longest_time = self.convert_type(round(max(params)), float)
            tau = self.convert_type(round(tau), float)
            pihalf_x = self.convert_type(round(pihalf_x), float)
            pihalf_y = self.convert_type(round(pihalf_y), float)
            pi_x = self.convert_type(round(pi_x), float)
            pi_y = self.convert_type(round(pi_y), float)
            dark_pi = self.convert_type(round(dark_pi), float)

            def SingleDEERCorrT1(t_corr):
                '''
                CREATE SINGLE DEER SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
                '''
                t_corr = self.convert_type(round(t_corr), float)
                pad_time = longest_time - t_corr
                '''
                DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
                '''            
                t1_corr_time = pihalf_x + tau + pi_y + tau + pihalf_y + 100 + dark_pi + t_corr + pihalf_x + tau + pi_y + tau + pihalf_y
            
                laser_off1 = self.initial_delay
                laser_off2 = self.singlet_decay + t1_corr_time + self.MW_buffer_time
                laser_off3 = self.initial_delay + 1000 + pad_time

                # digitizer trigger timing
                clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
                clock_off2 = - self.trig_spot + self.readout_time + laser_off3

                # mw I & Q off windows 
                iq_off1 = laser_off1 + self.laser_time + self.singlet_decay
                iq_off2 = (pihalf_x - self.awg_trig_time) + tau
                iq_off3 = (pi_y - self.awg_trig_time) + tau
                iq_off4 = (pihalf_y - self.awg_trig_time) + 100
                iq_off5 = t_corr
                iq_off6 = iq_off2
                iq_off7 = iq_off3
                iq_off8 = (pihalf_y - self.awg_trig_time) + self.MW_buffer_time + self.readout_time + laser_off3

                awg_off1 = 15 + iq_off1 + pihalf_x + self.awg_pulse_delay # additional initial delay at beginning to offset entire AWG pulse seq
                awg_off2 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pi_y + self.awg_pulse_delay
                awg_off3 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pihalf_y + iq_off4 
                awg_off4 = t_corr + pihalf_x
                awg_off5 = awg_off2
                awg_off6 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pihalf_y + iq_off4 - 15

                '''
                CONSTRUCT PULSE SEQUENCE
                '''
                # create sequence objects for MW on and off blocks
                seq = self.Pulser.createSequence() # with/without surface pi pulse
                seq_ref = self.Pulser.createSequence()
                seq2 = self.Pulser.createSequence() # y, -y normalization
                seq2_ref = self.Pulser.createSequence()

                # define sequence structure for laser
                # laser_seq = [(laser_off, 0), (laser_on, 1)]
                laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
                
                # define sequence structure for digitizer trigger
                dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
                
                # sequence structure for I & Q MW channels on SRS SG396
                mw_iq_seq = [(iq_off1, 0), (self.awg_trig_time, 1), (iq_off2, 0), (self.awg_trig_time, 1), (iq_off3, 0), (self.awg_trig_time, 1), (iq_off4, 0),  
                             (iq_off5, 0), (self.awg_trig_time, 1), (iq_off6, 0), (self.awg_trig_time, 1), (iq_off7, 0), (self.awg_trig_time, 1), (iq_off8, 0)]
                
                # sequence structure for I & Q MW channels (MW off)
                awg_seq =  [(awg_off1, 0), (self.awg_trig_time, 1), (awg_off2, 0), (self.awg_trig_time, 1), (awg_off3, 0),
                            (self.awg_trig_time, 1), (awg_off4, 0), (self.awg_trig_time, 1), (awg_off5, 0),
                            (self.awg_trig_time, 1), (awg_off6, 0)]
                awg_ref_seq = [(awg_off1, 0), (self.awg_trig_time, 1), (awg_off2, 0), (self.awg_trig_time, 1), (awg_off3, 0),
                               (self.awg_trig_time, 0), (awg_off4, 0), (self.awg_trig_time, 1), (awg_off5, 0),
                               (self.awg_trig_time, 1), (awg_off6, 0)]

                # assign sequences to respective channels for seq_on
                seq.setDigital(3, laser_seq) # laser
                seq.setDigital(1, dig_clock_seq) # digitizer trigger
                seq.setDigital(4, awg_seq)
                seq.setDigital(2, mw_iq_seq) # MW IQ

                # assign sequences to respective channels for seq_on
                seq_ref.setDigital(3, laser_seq) # laser
                seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
                seq_ref.setDigital(4, awg_ref_seq)
                seq_ref.setDigital(2, mw_iq_seq) # MW IQ

                # assign sequences to respective channels for seq_on
                seq2.setDigital(3, laser_seq) # laser
                seq2.setDigital(1, dig_clock_seq) # digitizer trigger
                seq2.setDigital(4, awg_seq)
                seq2.setDigital(2, mw_iq_seq) # MW IQ

                # assign sequences to respective channels for seq_on
                seq2_ref.setDigital(3, laser_seq) # laser
                seq2_ref.setDigital(1, dig_clock_seq) # digitizer trigger
                seq2_ref.setDigital(4, awg_ref_seq)
                seq2_ref.setDigital(2, mw_iq_seq) # MW IQ

                return seq + seq_ref + seq2 + seq2_ref
            
            seqs = self.Pulser.createSequence()

            for t in params:
                seqs += SingleDEERCorrT1(t)

            # return SingleDEER()
            return seqs
    
    def DEER_Corr_Rabi(self, params, tau, t_corr, pihalf_x, pihalf_y, pi_x, pi_y):
        '''
        Surface electron correlation Rabi pulse sequence.
        MW sequence: pi/2(x) - tau - pi(y) - tau - pi/2(x)
        '''
        longest_time = self.convert_type(round(max(params)), float)
        tau = self.convert_type(round(tau), float)
        t_corr = self.convert_type(round(t_corr), float)
        pihalf_x = self.convert_type(round(pihalf_x), float)
        pihalf_y = self.convert_type(round(pihalf_y), float)
        pi_x = self.convert_type(round(pi_x), float)
        pi_y = self.convert_type(round(pi_y), float)

        dark_pi_buffer_1 = 100
        dark_pi_buffer_2 = 100

        def SingleDEERCorrRabi(dark_pi):
            '''
            CREATE SINGLE DEER SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''
            dark_pi = self.convert_type(round(dark_pi), float)
            pad_time = longest_time - dark_pi

            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''            
            rabi_corr_time = pihalf_x + tau + pi_y + tau + pihalf_y + (dark_pi_buffer_1 + dark_pi + dark_pi_buffer_2) + t_corr + pihalf_x + tau + pi_y + tau + pihalf_y

            laser_off1 = self.initial_delay
            laser_off2 = self.singlet_decay + rabi_corr_time + self.MW_buffer_time
            laser_off3 = self.initial_delay + 1000 + pad_time

            # digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3

            # mw I & Q off windows 
            iq_off1 = laser_off1 + self.laser_time + self.singlet_decay
            iq_off2 = (pihalf_x - self.awg_trig_time) + tau
            iq_off3 = (pi_y - self.awg_trig_time) + tau
            iq_off4 = (pihalf_y - self.awg_trig_time) + dark_pi_buffer_1
            iq_off5 = dark_pi + dark_pi_buffer_2 + t_corr
            iq_off6 = iq_off2
            iq_off7 = iq_off3
            iq_off8 = (pihalf_y - self.awg_trig_time) + self.MW_buffer_time + self.readout_time + laser_off3

            # awg_pulse_delay = 0 is not used
            awg_off1 = 15 + iq_off1 + pihalf_x + self.awg_pulse_delay # additional initial delay at beginning to offset entire AWG pulse seq
            awg_off2 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pi_y + self.awg_pulse_delay
            awg_off3 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pihalf_y + iq_off4 
            awg_off4 = - self.awg_trig_time + iq_off5 + pihalf_x
            awg_off5 = awg_off2
            awg_off6 = (tau - self.awg_pulse_delay - self.awg_trig_time) + pihalf_y + iq_off8 - 15

            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            seq = self.Pulser.createSequence()
            seq_ref = self.Pulser.createSequence()

            # define sequence structure for laser
            # laser_seq = [(laser_off, 0), (laser_on, 1)]
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
            
            # define sequence structure for digitizer trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
            
            # sequence structure for I & Q MW channels on SRS SG396
            mw_iq_seq = [(iq_off1, 0), (self.awg_trig_time, 1), (iq_off2, 0), (self.awg_trig_time, 1), (iq_off3, 0), (self.awg_trig_time, 1), (iq_off4, 0),  
                            (iq_off5, 0), (self.awg_trig_time, 1), (iq_off6, 0), (self.awg_trig_time, 1), (iq_off7, 0), (self.awg_trig_time, 1), (iq_off8, 0)]
            
            # sequence structure for I & Q MW channels (MW off)
            awg_seq =  [(awg_off1, 0), (self.awg_trig_time, 1), (awg_off2, 0), (self.awg_trig_time, 1), (awg_off3, 0),
                        (self.awg_trig_time, 1), (awg_off4, 0), (self.awg_trig_time, 1), (awg_off5, 0),
                        (self.awg_trig_time, 1), (awg_off6, 0)]
            # awg_ref_seq = [(awg_off1, 0), (self.awg_trig_time, 1), (awg_off2, 0), (self.awg_trig_time, 1), (awg_off3, 0),
            #                (self.awg_trig_time, 1), (awg_off4, 0), (self.awg_trig_time, 1), (awg_off5, 0),
            #                (self.awg_trig_time, 1), (awg_off6, 0)]

            # assign sequences to respective channels for seq_on
            seq.setDigital(3, laser_seq) # laser
            seq.setDigital(1, dig_clock_seq) # digitizer trigger
            seq.setDigital(4, awg_seq)
            seq.setDigital(2, mw_iq_seq) # MW IQ

            # assign sequences to respective channels for seq_on
            seq_ref.setDigital(3, laser_seq) # laser
            seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
            seq_ref.setDigital(4, awg_seq)
            seq_ref.setDigital(2, mw_iq_seq) # MW IQ

            return seq + seq_ref
        
        seqs = self.Pulser.createSequence()

        for p in params:
            seqs += SingleDEERCorrRabi(p)

        # return SingleDEER()
        return seqs
    
    def Corr_Spectroscopy(self, params, tau, pihalf_x, pihalf_y, pi_x, pi_y, n):
        '''
        Correlation Spectroscopy sequence using YY8-N.
        MW sequence: pi/2(x) - tau - (pi(x) - 2*tau - pi(y) - 2*tau - pi(x) - 2*tau...)^N - tau - pi/2(x, -x)
        '''
        longest_time = self.convert_type(round(max(params)), float)
        tau = self.convert_type(round(tau), float)
        pihalf_x = self.convert_type(round(pihalf_x), float)
        pihalf_y = self.convert_type(round(pihalf_y), float)
        pi_x = self.convert_type(round(pi_x), float)
        pi_y = self.convert_type(round(pi_y), float)
        n = self.convert_type(round(n), int)
        
        def PiPulsesN(axes, tau, N):
            if axes == 'xy':
                # xy4_I_seq = [self.Pi('x', pi_x)[0], (tau, self.IQ0[0]), self.Pi('y', pi_y)[0], (tau, self.IQ0[0]), self.Pi('x', pi_x)[0], (tau, self.IQ0[0]), self.Pi('y', pi_y)[0]]
                xy8_iq_seq = [(tau/2, 0), (self.awg_trig_time, 1), 
                              ((pi_x - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_x - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_x - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_x - self.awg_trig_time) + tau/2, 0)]

                mw_IQ = (xy8_iq_seq)*N
                
            elif axes == 'yy':
                yy8_iq_seq = [(tau/2, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau/2, 0)]

                mw_IQ = (yy8_iq_seq)*N
            
            return mw_IQ

        def SingleCorrSpec(t_corr):
            '''
            CREATE SINGLE HAHN-ECHO SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''

            t_corr = self.convert_type(round(t_corr), float) # convert to proper data type to avoid undesired rpyc netref data type

            '''
            DEFINE SPECIAL TIME INTERVALS FOR EXPERIMENT

            pad_time = padding time to equalize duration of every run (for different tau durations)
            '''
            pad_time = longest_time - t_corr 

            # total time for correlation spectroscopy MW pulse sequence
            # corr_spec_time = pihalf_x + ((tau/2)/(8*n) + 4*pi_x + 4*pi_y + 7*tau/(8*n) + (tau/2)/(8*n))*n + pihalf_y + t_corr + \
            #                  pihalf_x + ((tau/2)/(8*n) + 4*pi_x + 4*pi_y + 7*tau/(8*n) + (tau/2)/(8*n))*n + pihalf_y

            # XY8
            corr_spec_time = pihalf_x + (tau/2 + 0*pi_x + 4*pi_x + 4*pi_y + 7*tau + tau/2)*n + pihalf_y + t_corr + \
                             pihalf_x + (tau/2 + 0*pi_x + 4*pi_x + 4*pi_y + 7*tau + tau/2)*n + pihalf_y
            
            
            # YY8
            # corr_spec_time = pihalf_x + (tau/2 + 0*pi_x + 0*pi_x + 8*pi_y + 7*tau + tau/2)*n + pihalf_y + t_corr + \
            #                  pihalf_x + (tau/2 + 0*pi_x + 0*pi_x + 8*pi_y + 7*tau + tau/2)*n + pihalf_y
            
            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''            
            laser_off1 = self.initial_delay
            laser_off2 = self.singlet_decay + corr_spec_time + self.MW_buffer_time
            laser_off3 = 100 + pad_time

            # digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3  

            # mw I & Q off windows 
            iq_off_start = laser_off1 + self.laser_time + self.singlet_decay
            iq_off_end = (pihalf_x - self.awg_trig_time) + self.MW_buffer_time + self.readout_time + laser_off3

            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            seq = self.Pulser.createSequence()
            seq_ref = self.Pulser.createSequence()

            # define sequence structure for laser
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
            
            # define sequence structure for digitizer trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
            
            # sequence structure for I & Q MW channels 
            mw_iq_seq = [(iq_off_start, 0), (self.awg_trig_time, 1), (pihalf_x - self.awg_trig_time, 0)] + PiPulsesN('xy', tau, n) + [(self.awg_trig_time, 1), (pihalf_y - self.awg_trig_time, 0), (t_corr, 0), (self.awg_trig_time, 1), (pihalf_x - self.awg_trig_time, 0)] + PiPulsesN('xy', tau, n) + [(self.awg_trig_time, 1), (pihalf_y - self.awg_trig_time, 0), (iq_off_end, 0)]
            
            # assign sequences to respective channels for seq_on
            seq.setDigital(3, laser_seq) # laser
            seq.setDigital(1, dig_clock_seq) # digitizer trigger
            seq.setDigital(2, mw_iq_seq) # MW IQ
        
            # assign sequences to respective channels for seq_off
            seq_ref.setDigital(3, laser_seq) # laser
            seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger    
            seq_ref.setDigital(2, mw_iq_seq) # MW IQ

            return seq + seq_ref

        # concatenate single correlation spectroscopy sequence "runs" number of times
        seqs = self.Pulser.createSequence()
        
        for t_corr in params:
            seqs += SingleCorrSpec(t_corr)

        return seqs
    
    def CASR(self, tau, pihalf_x, pihalf_y, pi_x, pi_y, n, n_sr):
        '''
        Coherent averaged synchronized readout (CASR).
        '''
        longest_time = self.convert_type(round(max(params)), float)
        tau = self.convert_type(round(tau), float)
        pihalf_x = self.convert_type(round(pihalf_x), float)
        pihalf_y = self.convert_type(round(pihalf_y), float)
        pi_x = self.convert_type(round(pi_x), float)
        pi_y = self.convert_type(round(pi_y), float)
        n = self.convert_type(round(n), int)
        n_sr = self.convert_type(round(n_sr), int) # number of subsequences (synchronized readouts) per "run"

        def PiPulsesN(axes, tau, N):
            if axes == 'xy':
                # xy4_I_seq = [self.Pi('x', pi_x)[0], (tau, self.IQ0[0]), self.Pi('y', pi_y)[0], (tau, self.IQ0[0]), self.Pi('x', pi_x)[0], (tau, self.IQ0[0]), self.Pi('y', pi_y)[0]]
                xy8_iq_seq = [(tau/2, 0), (self.awg_trig_time, 1), 
                              ((pi_x - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_x - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_x - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_x - self.awg_trig_time) + tau/2, 0)]

                mw_IQ = (xy8_iq_seq)*N
                
            elif axes == 'yy':
                yy8_iq_seq = [(tau/2, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1), 
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau, 0), (self.awg_trig_time, 1),
                              ((pi_y - self.awg_trig_time) + tau/2, 0)]

                mw_IQ = (yy8_iq_seq)*N
            
            return mw_IQ

        def SingleCASR():
            '''
            CREATE SINGLE HAHN-ECHO SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''

            '''
            DEFINE SPECIAL TIME INTERVALS FOR EXPERIMENT

            pad_time = padding time to equalize duration of every run (for different tau durations)
            '''
            pad_time = longest_time 

            # total time for correlation spectroscopy MW pulse sequence
            casr_time = pihalf_x + (tau/2 + 0*pi_x + 4*pi_x + 4*pi_y + 7*tau + tau/2)*n + pihalf_y + t_corr + \
                             pihalf_x + (tau/2 + 0*pi_x + 4*pi_x + 4*pi_y + 7*tau + tau/2)*n + pihalf_y
            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''     
            # total sequence time 100 ns + 6 us + 500 ns + corr spec time + 100 ns + 400 ns + 100 ns + pad time       
            laser_off1 = self.initial_delay
            laser_off2 = self.singlet_decay + casr_time + self.MW_buffer_time
            laser_off3 = pad_time

            # digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3     

            # mw I & Q off windows 
            iq_off_start = laser_off1 + self.laser_time + self.singlet_decay
            iq_off_end = (pihalf_y - self.awg_trig_time) + self.MW_buffer_time + self.readout_time + laser_off3

            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            seq_init = self.Pulser.createSequence()
            seq = self.Pulser.createSequence()

            # define sequence structure for laser
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
            
            # define sequence structure for digitizer trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
            
            # sequence structure for I & Q MW channels 
            mw_iq_seq = [(iq_off_start, 0), (self.awg_trig_time, 1), (pihalf_x - self.awg_trig_time, 0)] + PiPulsesN('xy', tau, n) + [(self.awg_trig_time, 1), (iq_off_end, 0)]

            # nuclear spin pi/2 initial pulse
            nuclear_spin_seq = [(self.awg_trig_time, 1)]

            # assign initial nuclear spin pi/2 pulse to seq_init
            seq_init.setDigital(5, nuclear_spin_seq)

            # assign sequences to respective channels for seq
            seq.setDigital(3, laser_seq) # laser
            seq.setDigital(1, dig_clock_seq) # digitizer trigger
            seq.setDigital(2, mw_iq_seq) # MW IQ

            return seq_init + seq*n_sr
        
        # concatenate single correlation spectroscopy sequence "runs" number of times
        seqs = SingleCASR()

        return seqs
    
    
    def AERIS(self, params, pulse_axes, pihalf_x, pihalf_y, pi_x, pi_y, n):

        '''
        Amplitude-Encoded Radio Induced Signal (AERIS) pulse sequence.
        '''
        longest_time = self.convert_type(round(params[-1]), float)
        pihalf_x = self.convert_type(round(pihalf_x), float)
        pihalf_y = self.convert_type(round(pihalf_y), float)
        pi_x = self.convert_type(round(pi_x), float)
        pi_y = self.convert_type(round(pi_y), float)
        n = self.convert_type(round(n), int)
        
        def PiPulsesN(axes, tau, N):
            if axes == 'xy':
                xy4_I_seq = [((tau/2)/(4*N), self.IQ0[0]), self.Pi('x', pi_x)[0], (tau/(4*N), self.IQ0[0]), self.Pi('y', pi_y)[0], (tau/(4*N), self.IQ0[0]), self.Pi('x', pi_x)[0], (tau/(4*N), self.IQ0[0]), self.Pi('y', pi_y)[0], ((tau/2)/(4*N), self.IQ0[0])]
                xy4_Q_seq = [((tau/2)/(4*N), self.IQ0[1]), self.Pi('x', pi_x)[1], (tau/(4*N), self.IQ0[1]), self.Pi('y', pi_y)[1], (tau/(4*N), self.IQ0[1]), self.Pi('x', pi_x)[1], (tau/(4*N), self.IQ0[1]), self.Pi('y', pi_y)[1], ((tau/2)/(4*N), self.IQ0[1])]
                mw_I = (xy4_I_seq)*N
                mw_Q = (xy4_Q_seq)*N
            elif axes == 'yy':
                yy4_I_seq = [((tau/2)/(4*N), self.IQ0[0]), self.Pi('y', pi_y)[0], (tau/(4*N), self.IQ0[0]), self.Pi('y', pi_y)[0], (tau/(4*N), self.IQ0[0]), self.Pi('y', pi_y)[0], (tau/(4*N), self.IQ0[0]), self.Pi('y', pi_y)[0], ((tau/2)/(4*N), self.IQ0[0])]
                yy4_Q_seq = [((tau/2)/(4*N), self.IQ0[1]), self.Pi('y', pi_y)[1], (tau/(4*N), self.IQ0[1]), self.Pi('y', pi_y)[1], (tau/(4*N), self.IQ0[1]), self.Pi('y', pi_y)[1], (tau/(4*N), self.IQ0[1]), self.Pi('y', pi_y)[1], ((tau/2)/(4*N), self.IQ0[1])]
                mw_I = (yy4_I_seq)*N
                mw_Q = (yy4_Q_seq)*N

            return mw_I, mw_Q

        def SingleAERIS(tau):
            '''
            CREATE SINGLE HAHN-ECHO SEQUENCE TO REPEAT THROUGHOUT EXPERIMENT
            '''

            tau = self.convert_type(round(tau), float) # convert to proper data type to avoid undesired rpyc netref data type

            '''
            DEFINE SPECIAL TIME INTERVALS FOR EXPERIMENT

            pad_time = padding time to equalize duration of every run (for different tau durations)
            '''
            pad_time = longest_time - tau 
            # NOTICE: change if using PiHalf['y'] to pihalf_y
            xy4_time = 2*pihalf_x + (2*(tau/2)/(4*n) + 2*pi_x + 2*pi_y + 3*tau/(4*n))*n
            
            '''
            DEFINE RELEVANT ON, OFF TIMES FOR DEVICES
            '''            
            laser_off1 = self.initial_delay
            laser_off2 = self.singlet_decay + xy4_time + self.MW_buffer_time
            laser_off3 = 100 + pad_time
            
            # digitizer trigger timing
            clock_off1 = laser_off1 + self.laser_time + laser_off2 + self.trig_spot - self.clock_time
            clock_off2 = - self.trig_spot + self.readout_time + laser_off3         

            # mw I & Q off windows (on slightly longer than VSG to ensure it's set)
            iq_off_start = laser_off1 + self.laser_time + self.singlet_decay
            iq_off_end = self.MW_buffer_time + self.readout_time + laser_off3
            
            '''
            CONSTRUCT PULSE SEQUENCE
            '''
            # create sequence objects for MW on and off blocks
            seq = self.Pulser.createSequence()
            seq_ref = self.Pulser.createSequence()
            
            # define sequence structure for laser
            laser_seq = [(laser_off1, 0), (self.laser_time, 1), (laser_off2, 0), (self.readout_time, 1), (laser_off3, 0)]
            
            # define sequence structure for digitizer trigger
            dig_clock_seq = [(clock_off1, 0), (self.clock_time, 1), (clock_off2, 0)]
            
            # sequence structure for I & Q MW channels 
            mw_I_seq = [(iq_off_start, self.IQ0[0]), self.PiHalf('x', pihalf_x)[0]] + PiPulsesN(pulse_axes, tau, n)[0] + [self.PiHalf('x', pihalf_x)[0], (iq_off_end, self.IQ0[0])]
            mw_Q_seq = [(iq_off_start, self.IQ0[1]), self.PiHalf('x', pihalf_x)[1]] + PiPulsesN(pulse_axes, tau, n)[1] + [self.PiHalf('x', pihalf_x)[1], (iq_off_end, self.IQ0[1])]
            
            # sequence structure for I & Q MW channels (MW off)
            mw_I_ref_seq = [(iq_off_start, self.IQ0[0]), self.PiHalf('x', pihalf_x)[0]] + PiPulsesN(pulse_axes, tau, n)[0] + [self.PiHalf('-x', pihalf_x)[0], (iq_off_end, self.IQ0[0])]
            mw_Q_ref_seq = [(iq_off_start, self.IQ0[1]), self.PiHalf('x', pihalf_x)[1]] + PiPulsesN(pulse_axes, tau, n)[1] + [self.PiHalf('-x', pihalf_x)[1], (iq_off_end, self.IQ0[1])]

            # sequence structure for nuclear spin RF generator 
            rf_seq = []
            rf_ref_seq = []

            # assign sequences to respective channels for seq_on
            seq.setDigital(3, laser_seq) # laser
            seq.setDigital(1, dig_clock_seq) # digitizer trigger
            # seq.setDigital(1, rf_seq) # VSG switch to enable MW
            seq.setAnalog(0, mw_I_seq) # mw_I
            seq.setAnalog(1, mw_Q_seq) # mw_Q
            
            # assign sequences to respective channels for seq_off
            seq_ref.setDigital(3, laser_seq) # laser
            seq_ref.setDigital(1, dig_clock_seq) # digitizer trigger
            # seq_ref.setDigital(1, rf_ref_seq) # VSG switch to enable MW
            seq_ref.setAnalog(0, mw_I_ref_seq) # mw_I
            seq_ref.setAnalog(1, mw_Q_ref_seq) # mw_Q

            return seq + seq_ref

        # concatenate single ODMR sequence "runs" number of times
        seqs = self.Pulser.createSequence()
        
        for tau in params:
            seqs += SingleAERIS(tau)
        
        return seqs
    
