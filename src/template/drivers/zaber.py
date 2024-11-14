'''
Zaber Linear Stage Driver for Updated Nspyre

Author: Evan Villafranca
Date: 8/25/2022
'''

from collections import OrderedDict
from zaber_motion import Units, Library
from zaber_motion.ascii import Connection

import logging

logger = logging.getLogger(__name__)

class X_MCC2:

    # hhDistance = 100

    def __init__(self, axis_mapping, serial_port = 'COM3'):
        '''
        Connect to Zaber stages with correct serial_port (always "COM3").
        axis_mapping: dictionary mapping an axis peripheral ID (int) to the axis name (str)
        '''   
        self.connection = Connection.open_serial_port(serial_port)      

        if self.connection is not None:
            self.device_list = self.connection.detect_devices()
            logger.debug(f"Found [{len(self.device_list)}] Zaber device(s).")
        else:
            raise IOError("No Zaber devices found.")
    
        self.axes = OrderedDict()
        for device in self.device_list:
            for i in range(1, device.axis_count + 1):
                axis = device.get_axis(i)
                logger.debug(f"Found Zaber stage [{axis}] with axis number [{axis.axis_number}] & peripheral ID [{axis.peripheral_id}].")
                if axis.axis_number in axis_mapping:
                    axis_key = axis_mapping[axis.axis_number]
                    logger.debug(f"Associated Zaber stage [{axis_key}] with axis number [{axis.axis_number}].")
                    self.axes[axis_key] = axis

    def __getitem__(self, key):
        '''
        Return the Zaber axis object associated with the given key.
        '''
        if key in self.axes:
            return self.axes[key]
        else:
            raise KeyError(f"Zaber axis key [{key}] not found.")

    def close(self):
        self.connection.close()

    def __iter__(self):
        return self.axes.__iter__()

    def __next__(self):
        return self.axes.__next__()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()    

if __name__ == '__main__':
    import pdb; pdb.set_trace()
    # enable logging to console
    logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s.%(msecs)03d [%(levelname)8s] %(message)s', datefmt = '%m-%d-%Y %H:%M:%S')

    with X_MCC2({'left': 1, 
                 'right': 2}) as zab:
        pass