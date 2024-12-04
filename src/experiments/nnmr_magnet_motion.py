'''
Python file for controlling magnet mount equipment in NanoNMR experiments
in UChicago Maurer Lab

Copyright (c) 2022, Evan Villafranca
All rights reserved.
'''

import time

import numpy as np
from nspyre import InstrumentManager

from multiprocessing import Process
# from drivers.nnmr_stagecontrol_zaber import NanoNMRZaber
# from drivers.nnmr_stagecontrol_thorlabs import NanoNMRThorlabs

import logging
logger = logging.getLogger(__name__)
 
class NanoNMRMagnetMotion:
    '''
    Perform magnet mount movements & align magnetic fields using 
    functinality from nnmr_stagecontrol_.py files.
    '''
    def __init__(self):
        super().__init__()

    def get_all_positions(self):
        with InstrumentManager() as mgr:
            self.curr_zab_pos = mgr.zaber.current_positions
            self.curr_polar_pos = mgr.thor_polar.current_position
            self.curr_azi_pos = mgr.thor_azi.current_position

        return [self.curr_zab_pos, self.curr_polar_pos, self.curr_azi_pos]

    def get_parked_status(self):
        with InstrumentManager() as mgr:
            is_parked = mgr.zaber.check_parked()
            print("IS ZABER PARKED? ", is_parked)
            return is_parked
        
    def enable_all(self):
        with InstrumentManager() as mgr: 
            mgr.zaber.unpark()
            mgr.thor_polar.enable()
            mgr.thor_azi.enable()

    def disable_all(self):
        with InstrumentManager() as mgr:
            mgr.zaber.park()
            mgr.thor_polar.disable()
            mgr.thor_azi.disable()

    def move_to_home(self, stage):

        with InstrumentManager() as mgr:
            if stage == "zaber":
                mgr.zaber.home(mgr.zaber.update_positions_callback())
                # is_home = mgr.zaber.check_homed()
                # if is_home: 
                #     "Edit status bar here saying all devices already homed"
                #     logger.info("Zaber stages already homed.")
                # else:
                #     mgr.zaber.home(mgr.zaber.update_positions_callback())
            elif stage == "thor_polar":
                mgr.thor_polar.home(mgr.thor_polar.update_positions_callback())
                # is_home = mgr.thor_polar.check_homed()
                # if is_home:
                #     logger.info("Thorlabs polar stage already homed.")
                # else:
                #     mgr.thor_polar.home(mgr.thor_polar.update_positions_callback())
            elif stage == "thor_azi":
                mgr.thor_azi.home(mgr.thor_azi.update_positions_callback())
                # is_home = mgr.thor_azi.check_homed()
                # if is_home:
                #     logger.info("Thorlabs azimuthal stage already homed.")
                # else:
                #     mgr.thor_azi.home(mgr.thor_azi.update_positions_callback())

    def enable(self, stage):
        with InstrumentManager() as mgr:
            if stage == 'zaber':
                mgr.zaber.unpark()
            elif stage == 'thor_polar':
                mgr.thor_polar.enable()
            elif stage == 'thor_azi':
                mgr.thor_azi.enable()
            

    def disable(self, stage):
        with InstrumentManager() as mgr:
            if stage == 'zaber':
                mgr.zaber.park()
            elif stage == 'thor_polar':
                mgr.thor_polar.disable()
            elif stage == 'thor_azi':
                mgr.thor_azi.disable()
            

    def standby_all(self, **kwargs):
        with InstrumentManager() as mgr:
            try:
                kwargs['queue'].put_nowait(['start standby', mgr.zaber.current_positions, mgr.thor_polar.current_position, mgr.thor_azi.current_position])
                mgr.zaber.standby()
                mgr.thor_polar.standby()
                mgr.thor_azi.standby()

                # update positions of stages to display in GUI
                mgr.thor_azi.update_positions_callback()

                pos_azi = mgr.thor_azi.current_position 

                while not round(pos_azi,1) == round(115.0, 1):
                        time.sleep(0.8)
                        mgr.thor_azi.update_positions_callback() # update position
                        pos_azi = mgr.thor_azi.current_position

            except Exception as e:
                pass
            finally:
                time.sleep(0.1)
                # update positions of stages to display in GUI
                mgr.zaber.update_positions_callback()
                mgr.thor_polar.update_positions_callback()
                mgr.thor_azi.update_positions_callback()

                time.sleep(0.1)
                kwargs['queue'].put(['done', mgr.zaber.current_positions, mgr.thor_polar.current_position, mgr.thor_azi.current_position])

                time.sleep(0.1)


            
    def stop_motion(self, stage):
        with InstrumentManager() as mgr:
            if stage == "zaber":
                mgr.zaber.stop_motion()
            elif stage == "thor_polar":
                mgr.thor_polar.stop_motion()
            elif stage == "thor_azi":
                mgr.thor_azi.stop_motion()

    def stop_all_motion(self):
        with InstrumentManager() as mgr:
            mgr.zaber.stop_motion()
            mgr.thor_polar.stop_motion()
            mgr.thor_azi.stop_motion()

    def move_to_B_field(self):
        pass

    def move_to_orientation(self, **kwargs):
        with InstrumentManager() as mgr:        
            kwargs['queue'].put_nowait(['start', mgr.zaber.current_positions, mgr.thor_polar.current_position, mgr.thor_azi.current_position])

            try:
                if kwargs['stage'] == 'all': 
                    # reset zaber and polar positions to furthest apart and neutral respectively
                    mgr.zaber.move(100, kwargs['abs'])
                    mgr.thor_polar.move(0, kwargs['abs'])
                    time.sleep(0.1)
                    # after resetting stages to safe orientation, move to new positions
                    mgr.thor_azi.move(kwargs['new_pos'][0], kwargs['abs'])
                    mgr.thor_polar.move(kwargs['new_pos'][1], kwargs['abs'])
                    mgr.zaber.move(kwargs['new_pos'][2], kwargs['abs'])
    
                elif kwargs['stage'] == 'zaber':
                    # print(dir(mgr.zaber))
                    mgr.zaber.move(kwargs['new_pos'], kwargs['abs'])
                    print("NEW ZABER POSITION = ", kwargs['new_pos'])

                elif kwargs['stage'] == 'thor_polar':
                    mgr.thor_polar.move(kwargs['new_angle'], kwargs['abs'])
                    mgr.thor_polar.update_positions_callback() # update position
                    pos = mgr.thor_polar.current_position # set pos to be position as move is beginning
                    while not round(pos,1) == round(kwargs['new_angle'], 1):
                        time.sleep(0.8)
                        mgr.thor_polar.update_positions_callback() # update position
                        pos = mgr.thor_polar.current_position

                elif kwargs['stage'] == 'thor_azi':
                    mgr.thor_azi.move(kwargs['new_angle'], kwargs['abs'])
                    mgr.thor_azi.update_positions_callback() # update position
                    pos = mgr.thor_azi.current_position 
                    while not round(pos,1) == round(kwargs['new_angle'], 1):
                        time.sleep(0.8)
                        mgr.thor_azi.update_positions_callback() # update position
                        pos = mgr.thor_azi.current_position
                        
                else:
                    logger.info(f"Invalid stage name {kwargs['stage']}.")
                    pass
            except:
                pass
            finally:
                time.sleep(0.1)
                # update positions of stages to display in GUI
                mgr.zaber.update_positions_callback()
                mgr.thor_polar.update_positions_callback()
                mgr.thor_azi.update_positions_callback()

                time.sleep(0.1)
                kwargs['queue'].put(['done', mgr.zaber.current_positions, mgr.thor_polar.current_position, mgr.thor_azi.current_position])

                time.sleep(0.1)

    def field_calibration(self):
        pass


    