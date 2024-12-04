'''
NI DAQ Sample Reading Function

Written by Evan Villafranca - 9/18/2023
'''

import numpy as np
from rpyc.utils.classic import obtain 
from nspyre import InstrumentManager

def read(buffer, seq, runs, timeout):
        '''
        Function that takes empty buffer as argument, reads samples to buffer and returns it.
        '''
        with InstrumentManager() as mgr:
            try:
                mgr.daq.start_ai_task()
            except:
                print("DAQ TASK ERROR!")
            
            runs = int(np.ceil(runs)) # integer no. of runs
            
            mgr.ps.stream(seq, runs) # execute chosen sequence on Pulse Streamer
            
            timeout = int(timeout)

            buffer = obtain(mgr.daq.read_samples(buffer, len(buffer), timeout)) # convert data back to numpy array from rpyc.netref data type 

            mgr.daq.stop_ai_task()
            mgr.ps.Pulser.reset()

            return buffer     

def read_9775(buffer, seq, runs, timeout):
        '''
        Function that takes empty buffer as argument, reads samples to buffer and returns it.
        '''
        with MyInstrumentManager() as mgr:
            try:
                mgr.daq.start_task()
            except:
                print("DAQ TASK ERROR!")
            
            runs = int(np.ceil(runs)) # integer no. of runs
            
            mgr.ps.stream(seq, runs) # execute chosen sequence on Pulse Streamer

            buffer = obtain(mgr.daq.read_samples(buffer, len(buffer), timeout)) # convert data back to numpy array from rpyc.netref data type 

            mgr.daq.stop_task()
            mgr.ps.Pulser.reset()

            return buffer     