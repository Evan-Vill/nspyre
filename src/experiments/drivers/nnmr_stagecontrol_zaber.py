'''
Python file for controlling Zaber equipment in NanoNMR experiments 
in UChicago Maurer Lab

Copyright (c) 2022, Evan Villafranca
All rights reserved.
'''

from zaber_motion import Units
import numpy as np

from zaber import X_MCC2

import logging
logger = logging.getLogger(__name__)

class NanoNMRZaber:
    '''
    NanoNMR Zaber stages motion control
    '''

    def __init__(self):
        
        # self.close()

        # define stages object to map Zaber stages to their respective axes
        self.Zstages = X_MCC2({1: 'left', 
                              2: 'right'})
        # print("ZABER 'LEFT' ID: ", self.Zstages['left'].identity)
        # print("ZABER 'RIGHT' ID: ", self.Zstages['right'].identity)
        self.Zstages.__enter__()

        self.current_positions = [None, None]
        self.update_positions_callback() # record positions on instrument server startup    

    def update_positions_callback(self):
        for axis in self.Zstages:
            pos = self.Zstages[axis].get_position(Units.LENGTH_MILLIMETRES)
            self.current_positions[self.Zstages[axis].axis_number - 1] = round(pos)
        logger.info(f"Zaber stage updated positions = {self.current_positions[0]} mm & {self.current_positions[1]} mm.")
        return self.current_positions

    def check_homed(self):
        all_homed = True
        for axis in self.Zstages:
            if not self.Zstages[axis].is_homed():
                all_homed = False
        return all_homed

    def check_standby(self):
        standby_condition_met = True
        for axis in self.Zstages:
            pos = self.Zstages[axis].get_position(unit = Units.LENGTH_MILLIMETRES)
            if not int(np.round(pos)) == 100:
                standby_condition_met = False
        return standby_condition_met

    def check_parked(self):
        parked_condition_met = True
        for axis in self.Zstages:
            parked = self.Zstages[axis].is_parked()
            if not parked:
                parked_condition_met = False
        return parked_condition_met
    
    def park(self):
        logger.info("Parking Zaber stages...")

        for axis in self.Zstages:
            self.Zstages[axis].park()

        logger.info("Zaber stages parked (inactive).")

    def unpark(self):
        logger.info("Unparking Zaber stages...")

        for axis in self.Zstages:
            self.Zstages[axis].unpark()

        logger.info("Zaber stages unparked (active).")

    def stop_motion(self):
        '''
        Stop motion of both Zaber magnet stages
        '''

        for axis in self.Zstages:
            self.Zstages[axis].stop(wait_until_idle = True)

    def standby(self):
        logger.info("Moving Zaber stages to standby mode...")

        for axis in self.Zstages:
            self.Zstages[axis].move_max(wait_until_idle = True)

        logger.info("Zaber stages in standby mode.")

    def home(self, callback):
        '''
        Home both Zaber magnet stages
        '''

        logger.info("Homing both Zaber axes...")

        for axis in self.Zstages:
            self.Zstages[axis].home(wait_until_idle = True)

        logger.info("Zaber stages successfully homed.")

        callback
        
    def move(self, pos, abs = False):
        if abs:
            if pos in range(101):
                for axis in self.Zstages:
                    self.Zstages[axis].move_absolute(pos, Units.LENGTH_MILLIMETRES)
            else:
                logger.warning("Absolute position requested out of range for Zaber stages.")
        
        else:
            if pos > 0:
                for axis in self.Zstages:
                    if (self.Zstages[axis].get_position(Units.LENGTH_MILLIMETRES) + pos <= 100):
                        self.Zstages[axis].move_relative(pos, Units.LENGTH_MILLIMETRES)
                    else:
                        logger.warning("Requested relative position move exceeds range for Zaber stages.")
            else:
                for axis in self.Zstages:
                    if (self.Zstages[axis].get_position(Units.LENGTH_MILLIMETRES) + pos >= 0):
                        self.Zstages[axis].move_relative(pos, Units.LENGTH_MILLIMETRES)
                    else:
                        logger.warning("Requested relative position move is below range for Zaber stages.")

    # TODO: implement Zaber move safety check function.
    # This will be based on current polar and azimuthal stage positions
    # and whether or not it is safe to bring magnets as close together as possible.
    def is_move_safe():
        pass

    def close(self):
        for axis in self.Zstages:
            self.Zstages.__exit__()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

if __name__ == '__main__':
    # enable logging to console
    logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s.%(msecs)03d [%(levelname)8s] %(message)s', datefmt = '%m-%d-%Y %H:%M:%S')

    with NanoNMRZaber() as zaber:
        pass