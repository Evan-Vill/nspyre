'''
Thorlabs Linear Stage Driver for Updated Nspyre

Author: Evan Villafranca
Date: 8/25/2022
'''

from ctypes import c_short, c_int, c_char_p

from thorlabs_kinesis import benchtop_stepper_motor as bsm

import logging
logger = logging.getLogger(__name__)

class BSC201:
    
    def __init__(self, serial_id):

        self.channel = c_short(1)
        self.millisecs = c_int(100)

        self.stage_serial_id = serial_id
        if self.stage_serial_id == '40179174':
            self.key = 'azi'
        elif self.stage_serial_id == '40251814':
            self.key = 'polar'
        else:
            print(f"Invalid Thorlabs serial no.")

        if bsm.TLI_BuildDeviceList() == 0:
            logger.debug(f"Found {bsm.TLI_GetDeviceListSize()} Thorlabs stages.")

            self.serial_no = c_char_p(bytes(self.stage_serial_id, 'utf-8'))
            
            if bsm.SBC_Open(self.serial_no) == 0:
                bsm.SBC_StartPolling(self.serial_no, self.channel, self.millisecs)
                logger.debug(f"Found Thorlabs stage {self.key} with serial no. {int(self.stage_serial_id)}.")
            else:
                print(f"Can't open Thorlabs stage {self.key} (serial no. {int(self.stage_serial_id)}). Check if Kinesis software is already open with devices connected. If so, close the Kinesis software w/o disconnecting devices.")
        else:
            print("Can't build Thorlabs device list.")

   
    def __getitem__(self, key):
        '''
        Return the Thorlabs stage object associated with the given key.
        '''
        if key == self.key:
            return key
        else:
            raise KeyError(f"Thorlabs stage key {key} not found.")

    def close(self):
        self.serial_no = c_char_p(bytes(self.stage_serial_id, 'utf-8'))
        bsm.SBC_StopPolling(self.serial_no, self.channel)
        bsm.SBC_Close(self.serial_no)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

if __name__ == '__main__':
    import pdb; pdb.set_trace()
    # enable logging to console
    logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s.%(msecs)03d [%(levelname)8s] %(message)s', datefmt = '%m-%d-%Y %H:%M:%S')

    with BSC201('40179174') as thor:
        pass
        

