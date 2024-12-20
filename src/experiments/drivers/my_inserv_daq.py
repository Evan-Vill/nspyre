#!/usr/bin/env python
"""
Start up an instrument server to host drivers. For the purposes of this demo,
it's assumed that this is running on the same system that will run experimental
code.
"""
from pathlib import Path
import logging

from nspyre import InstrumentServer
from nspyre import InstrumentGateway
from nspyre import nspyre_init_logger
from nspyre import serve_instrument_server_cli

_HERE = Path(__file__).parent

# log to the console as well as a file inside the logs folder
nspyre_init_logger(
    logging.INFO,
    log_path=_HERE / '../logs',
    log_path_level=logging.DEBUG,
    prefix='local_inserv',
    file_size=10_000_000,
)

with InstrumentServer() as local_inserv:
    # local_inserv.add('subs', _HERE / 'subsystems_driver.py', 'SubsystemsDriver', args=[local_inserv, remote_gw], local_args=True)
    
    local_inserv.add(name = 'laser', 
                     class_path = _HERE / 'laser_driver.py', 
                     class_name = 'LaserControl',
                     args = ['LAS-08166'])

    local_inserv.add(name = 'laser_shutter', 
                     class_path = _HERE / 'thorlabs_laser_shutter_driver.py', 
                     class_name = 'LaserShutter',
                     args = ['68800950'])

    local_inserv.add(name = 'sg', 
                     class_path = _HERE / 'sg396_driver.py', 
                     class_name = 'SG396',
                     args = ['TCPIP::10.135.70.65::inst0::INSTR'])

    local_inserv.add(name = 'awg', 
                     class_path = _HERE / 'hdawg_driver_latest.py', 
                     class_name = 'HDAWG',
                     args = ['dev8181', '127.0.0.1', 8004])
    
    local_inserv.add(name = 'ps',
                     class_path = _HERE / 'ps_driver_new_sideband.py',
                     class_name = 'Pulses')
    
    # local_inserv.add(name = 'dig', 
    #                  class_path = _HERE  / 'FIFOv4.py', 
    #                  class_name = 'FIFO', 
    #                  args = ['/dev/spcm0'])

    local_inserv.add(name = 'daq',
                     class_path = _HERE / 'daq_driver_new.py',
                     class_name = 'NIDAQ')

    local_inserv.add(name = 'zaber',
                     class_path = _HERE / 'nnmr_stagecontrol_zaber.py',
                     class_name = 'NanoNMRZaber')

    local_inserv.add(name = 'thor_azi', 
                     class_path = _HERE / 'nnmr_stagecontrol_thorlabs.py', 
                     class_name = 'NanoNMRThorlabs',
                     args = ['40179174'])

    local_inserv.add(name = 'thor_polar', 
                     class_path = _HERE / 'nnmr_stagecontrol_thorlabs.py', 
                     class_name = 'NanoNMRThorlabs',
                     args = ['40251814'])    
    
    
    


    # run a CLI (command-line interface) that allows the user to enter
    # commands to control the server
    serve_instrument_server_cli(local_inserv)
