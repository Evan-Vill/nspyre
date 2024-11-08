'''
Swabian Instruments Digitizer Sample Reading Function

Written by Emmeline Riendeau - 2024-08-30
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
                mgr.Digitizer.check_connection()
                # mgr.Digitizer.stop_card()
            except:
                print("DIGITIZER TASK ERROR!")
            
            runs = int(np.ceil(runs)) # integer no. of runs
            
            # mgr.SWABIAN_Pulse_Streamer.stream(seq, runs) # execute chosen sequence on Pulse Streamer
            
            # timeout = int(timeout) MAYBE PUT THIS ELSEWHERE

            buffer = obtain(mgr.Digitizer.acquire())# convert data back to numpy array from rpyc.netref data type 
        
            # maybe delete this command for the pulse streamer --> the card stops itself
            # mgr.Digitizer.stop_card() # this is redundant
            # mgr.SWABIAN_Pulse_Streamer.Pulser.reset()

            print("ACQUIRED DIGITIZER BUFFER: ", buffer)
            return buffer     