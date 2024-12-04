"""
This is example script demonstrates most of the basic functionality of nspyre.
"""
import time
import logging
# import spcm
from spcm import units
from pathlib import Path
from typing import List

import numpy as np
from scipy.signal import find_peaks
from nspyre import DataSource, DataSink
from nspyre import experiment_widget_process_queue
from nspyre import StreamingList
from nspyre import nspyre_init_logger

from saver import DataSaver
from nspyre.data.save import save_json
from tqdm import tqdm

from rpyc.utils.classic import obtain 

# from template.drivers.insmgr import MyInstrumentManager
from nspyre import InstrumentManager

from customUtils import flexSave
from daq_read_samples import read

_HERE = Path(__file__).parent
_logger = logging.getLogger(__name__)

class SpinMeasurements:
    """Spin measurement experiments."""

    def __init__(self, queue_to_exp=None, queue_from_exp=None):
        """
        Args:
            queue_to_exp: A multiprocessing Queue object used to send messages
                to the experiment from the GUI.
            queue_from_exp: A multiprocessing Queue object used to send messages
                to the GUI from the experiment.
        """
        self.queue_to_exp = queue_to_exp
        self.queue_from_exp = queue_from_exp

    # def __enter__(self):
    #     """Perform experiment setup."""
    #     # config logging messages
    #     # if running a method from the GUI, it will be run in a new process
    #     # this logging call is necessary in order to separate log messages
    #     # originating in the GUI from those in the new experiment subprocess
    #     nspyre_init_logger(
    #         log_level=logging.INFO,
    #         log_path=_HERE / '../logs',
    #         log_path_level=logging.DEBUG,
    #         prefix=Path(__file__).stem,
    #         file_size=10_000_000,
    #     )
    #     _logger.info('Created SpinMeasurements instance.')

    # def __exit__(self):
    #     """Perform experiment teardown."""
    #     _logger.info('Destroyed SpinMeasurements instance.')

    # def save(self, dataset, filename: str, timeout):
    #     dataset = dataset
    #     filename = filename

    #     try:
    #         try:
    #             # connect to the dataserver
    #             with DataSink(dataset) as sink:
    #                 # get the data from the dataserver
    #                 if DataSink(dataset) is not None:
    #                     print("ATTEMPTING TO SAVE: ", sink.params)
    #                     sink.pop()
    #                     save_json(filename, sink.data)
    #                 else:
    #                     pass 
    #                     # raise TypeError
                    
    #         except TimeoutError as err:
    #             raise TimeoutError(
    #                 f'Timed out retreiving the data set [{dataset}] from data server.'
    #             ) from err
    #         else:
    #             print("SUCCESSFULLY SAVED")
    #             _logger.info(f'Saved data set [{dataset}] to [{filename}].')
    #     except Exception as err:
    #         raise err
        
        # self.saver.save(filename = filename,
        #     dataset = dataset,
        #     save_fun = save_json,
        #     timeout = 10,
        #     # callback = self._save_callback
        #     )
        
    def generate_buffer(self, exp, runs, num_pts = 0):
        
        if exp == 'ODMR':
            buffer_size = 2*runs
            if buffer_size < 2:
                raise ValueError("Buffer size too small.")
        
        elif exp == 'DEER' or exp == 'DQ':
            buffer_size = 4*runs*num_pts
            if buffer_size < 4:
                raise ValueError("Buffer size too small.")

        elif exp == 'CD':
            buffer_size = 6*runs*num_pts
            if buffer_size < 6:
                raise ValueError("Buffer size too small.")
            
        else: 
            buffer_size = 2*runs*num_pts
            if buffer_size < 2:
                raise ValueError("Buffer size too small.")
            
        ni_sample_buffer = np.ascontiguousarray(np.zeros(buffer_size), dtype=np.float64)
        
        return [ni_sample_buffer]
    
    def analog_math(self, array, exp_type, pts = 0):        

        # split up buffers array into signal data (MW ON) and background data (MW OFF)

        ana_sig = array[::2] # single integrated data point for each MW on/off window
        ana_bg = array[1::2]

        if exp_type == 'ODMR':
            sum1 = np.sum(ana_sig) # MW ON - summed up dark data for plotting
            sum2 = np.sum(ana_bg) # MW OFF - summed up bright data for plotting
            
            return [sum1, sum2]
        
        elif exp_type == 'Cal' or exp_type == 'Optical_T1':
            sig_array = np.ones(pts)
            ana_sig = ana_sig + ana_bg
            
            for i in range(pts):
                sig_array[i] = np.sum(ana_sig[i::pts])

            return sig_array
        
        elif exp_type == 'DEER':
            ana_dark_sig = array[::4]
            ana_dark_bg = array[1::4]
            ana_echo_sig = array[2::4]
            ana_echo_bg = array[3::4]
            
            dark_ms1_array = np.ones(pts) # dark data array
            dark_ms0_array = np.ones(pts) # bright data array
            echo_ms1_array = np.ones(pts) # dark data array
            echo_ms0_array = np.ones(pts) # bright data array

            for i in range(pts):
                dark_ms1_array[i] = np.mean(ana_dark_sig[i::pts]) # pts = no. of runs
                dark_ms0_array[i] = np.mean(ana_dark_bg[i::pts])
                echo_ms1_array[i] = np.mean(ana_echo_sig[i::pts])
                echo_ms0_array[i] = np.mean(ana_echo_bg[i::pts])
            
            return [dark_ms1_array, dark_ms0_array, echo_ms1_array, echo_ms0_array]

        elif exp_type == 'DQ':
            s00 = array[::4]
            s0m = array[1::4]
            smm = array[2::4]
            smp = array[3::4]
            
            s00_array = np.ones(pts) # dark data array
            s0m_array = np.ones(pts) # bright data array
            smm_array = np.ones(pts) # dark data array
            smp_array = np.ones(pts) # bright data array

            for i in range(pts):
                s00_array[i] = np.mean(s00[i::pts]) # pts = no. of runs
                s0m_array[i] = np.mean(s0m[i::pts])
                smm_array[i] = np.mean(smm[i::pts])
                smp_array[i] = np.mean(smp[i::pts])
            
            return [s00_array, s0m_array, smm_array, smp_array]
        
        elif exp_type == 'CD':
            ana_dark_sig = array[::6]
            ana_dark_bg = array[1::6]
            ana_echo_sig = array[2::6]
            ana_echo_bg = array[3::6]
            ana_cd_sig = array[4::6]
            ana_cd_bg = array[5::6]
            
            dark_ms1_array = np.ones(pts) # dark data array
            dark_ms0_array = np.ones(pts) # bright data array
            echo_ms1_array = np.ones(pts) # dark data array
            echo_ms0_array = np.ones(pts) # bright data array
            cd_ms1_array = np.ones(pts) # dark data array
            cd_ms0_array = np.ones(pts) # bright data array

            for i in range(pts):
                dark_ms1_array[i] = np.mean(ana_dark_sig[i::pts]) # pts = no. of runs
                dark_ms0_array[i] = np.mean(ana_dark_bg[i::pts])
                echo_ms1_array[i] = np.mean(ana_echo_sig[i::pts])
                echo_ms0_array[i] = np.mean(ana_echo_bg[i::pts])
                cd_ms1_array[i] = np.mean(ana_cd_sig[i::pts])
                cd_ms0_array[i] = np.mean(ana_cd_bg[i::pts])

            return [dark_ms1_array, dark_ms0_array, echo_ms1_array, echo_ms0_array, cd_ms1_array, cd_ms0_array]
        
        else:
            ms1_array = np.ones(pts) # dark data array
            ms0_array = np.ones(pts) # bright data array

            for i in range(pts):
                ms1_array[i] = np.mean(ana_sig[i::pts])
                ms0_array[i] = np.mean(ana_bg[i::pts])
            
            return [ms1_array, ms0_array]
                
    def find_odmr_peaks(self):
        
        with DataSource('ODMR') as odmr_data:
            
            x_data = odmr_data.datasets['signal'][1][0]
            y_data = odmr_data.datasets['signal'][:][1] / odmr_data.dataset['background'][:][1]

            peaks, _ = find_peaks(y_data)

            res_freqs = x_data[peaks]

            return res_freqs

    def sort_taus(self, array):
        sorted_array = np.zeros(len(array)) # initialize sorted array

        array_copy = np.copy(array) # create copy of array to pop values from

        idx_iter = 0 # iteration index to determine whether an element is popped from beginning or end of array (alternating)

        for i in range(len(array)):
            if idx_iter%2 == 0: 
                popped, array_copy = array_copy[0], array_copy[1:] # even no. for idx_iter --> pop from beginning of array 
            else:
                popped, array_copy = array_copy[-1], array_copy[:-1] # odd no. for idx_iter --> pop from end of array 
            
            sorted_array[i] = popped # create sorted array
            idx_iter += 1

        return sorted_array

    def equipment_off(self):
        with InstrumentManager() as mgr:
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG
            dig = mgr.SPECTRUM_Digitizer

            laser.laser_off()
            sig_gen.set_rf_toggle(0)
            sig_gen.set_mod_toggle(0)

            ps.Pulser.reset()
            daq.close_do_task()
            daq.close_ao_task()
            daq.close_ai_task()
            dig.stop_card()
            # hdawg.set_disabled()

    def choose_sideband(self, opt, nv_freq, side_freq, pulse_axis='x'):
        match pulse_axis:
            case 'y':
                delta = 90
            case _: 
                delta = 0

        match opt:
            case 'Upper':
                frequencies = nv_freq - side_freq
                iq_phases = [delta+90, delta+0]
            case 'Both':
                frequencies = nv_freq - side_freq
                iq_phases = [delta+0, delta+90, delta+90, delta+0] # lower sideband phases + upper sideband phases
            case _:
                frequencies = nv_freq + side_freq
                iq_phases = [delta+0, delta+90]

        return frequencies, iq_phases
    
    def digitizer_configure(self, **kwargs):
        dig_config = {'num_pts_in_exp': 2*kwargs['num_pts_in_exp'], # MW ON + OFF subsequences
                      'segment_size' : kwargs['segment_size'] ,
                      'pretrig_size' : kwargs['pretrig_size'] ,
                      'readout_ch' : int(kwargs['read_channel']),
                      'runs' : kwargs['runs']}
        # print("config params: ", dig_config)

        # dig_config={'num_pts_in_exp': int(2*kwargs['num_pts_in_exp']),
        #                     # 'pretrig_size' : 32 * units.Sa,
        #                     # 'segment_size' : 512 * units.Sa,
        #                     'readout_ch' : 0,
        #                     'runs' : int(kwargs['runs'])}
        
        # for key, value in dig_config.items():
        #     print(f"{key}: {type(value)}")

        # print(dig_config.items)

        return dig_config
    
    ### Experiment logic

    def sigvstime_scan(self, **kwargs):
        
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as sigvstime_data:
            # run laser on continuously here from laser driver
            laser = mgr.OXXIUS_Laser
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ

            ps_seq = ps.SigvsTime(1/kwargs['sampling_rate'] * 1e9) # pulse streamer sequence for CW ODMR

            n_runs = 10
            signal_array = np.zeros(n_runs)

            ps.sampling_time = 1/kwargs['sampling_rate'] * 1e9 # period defining sig vs time sampling rate
            ps.clock_time = 11 #* 1e9 #width of our clock pulse.
            daq.open_ai_task(kwargs['sigvstime_detector'], n_runs) # one clock per each of the "n_runs" no. of sequences

            time_start = time.time()

            signal_sweeps = StreamingList()
            # PL_data = []
            # t = []

            for i in range(10000):
                print("Scan index ", i)
                sig_result = read(signal_array, ps_seq, n_runs, 1)

                sig_result = np.mean(sig_result)

                time_pt = time.time() - time_start

                # read the analog voltage levels received by the APD.
                # notify the streaminglist that this entry has updated so it will be pushed to the data server
                signal_sweeps.append(np.array([[time_pt], [sig_result]]))
                signal_sweeps.updated_item(-1) 
                
                # PL_data.append(sig_result) # line plot old version of plotting
                # t.append(time_pt)
                # print("Signal sweeps: ", signal_sweeps)
                # save the current data to the data server.
                sigvstime_data.push({'title': 'Signal Vs Time',
                                'xlabel': 'Time step',
                                'ylabel': 'APD Voltage (V)',
                                'datasets': {'signal': signal_sweeps}})

                # sigvstime_data.push({'idx': i,
                #                      't': t, 
                #                      'sig_data': PL_data})
                
                if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                    # the GUI has asked us nicely to exit. Save data if requested.
                    if kwargs['save'] == True:
                        flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                    
                    self.equipment_off()

                    return
            
    def odmr_scan(self, **kwargs):
        """
        Run a CW ODMR sweep over a set of microwave frequencies.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as cw_odmr_data:
            # load devices used in ODMR scan
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG
            dig = mgr.SPECTRUM_Digitizer

            # Generate ODMR data buffer
            odmr_buffer = self.generate_buffer('ODMR', kwargs['runs'])

            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'])


            # frequencies that will be swept over in the ODMR measurement
            frequencies = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])            
            
            frequencies_real, iq_phases = self.choose_sideband(kwargs['sideband'], frequencies, kwargs['sideband_freq'], 'y')

            sig_gen.set_rf_amplitude(kwargs['rf_power'])
            sig_gen.set_mod_type('QAM')
            sig_gen.set_mod_function('external')
            sig_gen.set_mod_toggle(1)
            
            ps.probe_time = kwargs['probe'] * 1e9
            ps_seq = ps.CW_ODMR() # pulse streamer sequence for CW ODMR
  
            try:
                hdawg.set_sequence(**{'seq': 'CW ODMR',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'probe_length': kwargs['probe'], 
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freq': kwargs['sideband_freq'], 
                                    'iq_phases': iq_phases,
                                    'num_pts': kwargs['num_pts'],
                                    'runs': kwargs['runs'], 
                                    'iters': kwargs['iters']})
                    
                time.sleep(0.5)
            except Exception as e:
                print(e)
            else:
                daq.open_ai_task(kwargs['detector'], len(odmr_buffer[0]))

                dig.assign_param(dig_config)

                # for storing the experiment data
                # list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure devices used in scan
                laser.set_modulation_state('cw')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):

                        sig_array = np.empty(kwargs['num_pts'])
                        sig_array[:] = np.nan
                        signal_sweeps.append(np.stack([frequencies/1e9, sig_array]))

                        bg_array = np.empty(kwargs['num_pts'])
                        bg_array[:] = np.nan
                        background_sweeps.append(np.stack([frequencies/1e9, bg_array]))

                        for f, freq in enumerate(frequencies_real):

                            sig_gen.set_frequency(freq) # set signal generator frequency
                            
                            dig.config()
                            dig.start_buffer()
                            ps.stream(ps_seq, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                            odmr_result_raw = obtain(dig.acquire())
                            
                            odmr_result = np.mean(odmr_result_raw, axis=1)
                                
                            # partition buffer into signal and background datasets
                            sig = self.analog_math(odmr_result, 'ODMR')[0]
                            bg = self.analog_math(odmr_result, 'ODMR')[1]
                            
                            # read the analog voltage levels received by the APD.
                            # notify the streaminglist that this entry has updated so it will be pushed to the data server
                            signal_sweeps[-1][1][f] = sig 
                            signal_sweeps.updated_item(-1) 
                            background_sweeps[-1][1][f] = bg
                            background_sweeps.updated_item(-1)

                            # save the current data to the data server.
                            cw_odmr_data.push({'params': {'kwargs': kwargs},
                                            'title': 'CW Optically Detected Magnetic Resonance',
                                            'xlabel': 'Frequency (GHz)',
                                            'ylabel': 'Signal',
                                            'datasets': {'signal' : signal_sweeps,
                                                        'background': background_sweeps}})

                            if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                                # the GUI has asked us nicely to exit. Save data if requested.
                                if kwargs['save'] == True:
                                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                                
                                self.equipment_off()

                                return

                        pbar.update(1)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])
                
                # Automatically save data if action requested.
                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                
                self.equipment_off()

    def rabi_scan(self, **kwargs):
        """
        Run a Rabi sweep over a set of microwave pulse durations.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as rabi_data:
            # load devices used in scan
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG
            dig = mgr.SPECTRUM_Digitizer

            # Generate Rabi data buffer
            rabi_buffer = self.generate_buffer('Rabi', kwargs['runs'], kwargs['num_pts'])
            
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'])
            
            # pi pulse durations that will be swept over in the Rabi measurement (converted to ns)
            mw_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
            np.random.shuffle(mw_times)

            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq'], kwargs['pulse_axis'])
        
            ps_seq = ps.Rabi(mw_times) # pulse streamer sequence

            sig_gen.set_frequency(sig_gen_freq) # set SG396 carrier frequency to drive NVs at selected sideband
            sig_gen.set_rf_amplitude(kwargs['rf_power'])
            sig_gen.set_mod_type('QAM')
            sig_gen.set_mod_function('external')
            sig_gen.set_mod_toggle(1)

            try:
                hdawg.set_sequence(**{'seq': 'Rabi',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freq': kwargs['sideband_freq'], 
                                    'iq_phases': iq_phases,
                                    'pi_pulses': mw_times/1e9, 
                                    'num_pts': kwargs['num_pts'],
                                    'runs': kwargs['runs'], 
                                    'iters': kwargs['iters']})  
                time.sleep(0.5)

            except Exception as e:
                print(e)
            else:
                daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))

                dig.assign_param(dig_config)

                # for storing the experiment data
                # list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                index_order = np.argsort(mw_times) 
                mw_times_sorted = np.sort(mw_times) # order mw_times for Rabi plotting

                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure devices used in scan
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        dig.config()
                        dig.start_buffer()
                        ps.stream(ps_seq, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                        rabi_result_raw = obtain(dig.acquire())
                        
                        rabi_result = np.mean(rabi_result_raw, axis=1)
                                        
                        # partition buffer into signal and background datasets
                        sig = self.analog_math(rabi_result, 'Rabi', kwargs['num_pts'])[0]
                        bg = self.analog_math(rabi_result, 'Rabi', kwargs['num_pts'])[1]
                        
                        sig = np.abs(sig)
                        bg = np.abs(bg)

                        # correct the y-axis data ordering for plots
                        sig = np.array([sig[i] for i in index_order])
                        bg = np.array([bg[i] for i in index_order])
                        
                        # read the analog voltage levels received by the APD.
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        
                        signal_sweeps.append(np.stack([mw_times_sorted, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([mw_times_sorted, bg]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server.
                        rabi_data.push({'params': {'kwargs': kwargs},
                                        'title': 'Rabi Oscillation',
                                        'xlabel': 'MW Pulse Duration (ns)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit
                            if kwargs['save'] == True:
                                flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            
                            self.equipment_off()

                            return
                        
                        pbar.update(1)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                self.equipment_off()

    def pulsed_odmr_scan(self, **kwargs):
        """
        Run a pulsed ODMR sweep over a set of microwave frequencies.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as odmr_data:
            # load devices used in ODMR scan
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG

            # Generate ODMR data buffer
            odmr_buffer = self.generate_buffer('ODMR', kwargs['runs'])
            
            # frequencies that will be swept over in the ODMR measurement
            frequencies = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])            

            # configure devices used in scan
            sig_gen.set_rf_amplitude(kwargs['rf_power'])
            sig_gen.set_mod_type('QAM')
            sig_gen.set_mod_function('external')
            sig_gen.set_mod_toggle(1)
            
            if kwargs['odmr_type'] == 'CW':
                # ps.probe_time = kwargs['probe'] * 1e9
                ps_seq = ps.CW_ODMR() # pulse streamer sequence for CW ODMR
            else:
                laser.set_modulation_state('Pulsed')
                # ps.laser_time = kwargs['init'] * 1e9 # initialization pulse duration
                # ps.readout_time = kwargs['read'] * 1e9 # readout window duration
                ps_seq = ps.Pulsed_ODMR(kwargs['runs'], kwargs['pulse_axis'], kwargs['pi']*1e9) # pulse streamer sequence for Pulsed ODMR
  
            daq.open_ai_task(kwargs['detector'], len(odmr_buffer[0]))

            # for storing the experiment data
            # list of numpy arrays of shape (2, num_points)
            signal_sweeps = StreamingList()
            background_sweeps = StreamingList()

            sig_gen.set_rf_toggle(1) # turn on NV signal generator

            laser.set_modulation_state('cw')
            laser.set_analog_control_mode('current')
            laser.set_diode_current_realtime(kwargs['laser_power'])
            laser.laser_on()
            
            with tqdm(total = kwargs['iters']) as pbar:

                for i in range(kwargs['iters']):

                    sig_array = np.empty(kwargs['num_pts'])
                    sig_array[:] = np.nan
                    signal_sweeps.append(np.stack([frequencies/1e9, sig_array]))

                    bg_array = np.empty(kwargs['num_pts'])
                    bg_array[:] = np.nan
                    background_sweeps.append(np.stack([frequencies/1e9, bg_array]))

                    for f, freq in enumerate(frequencies):

                        sig_gen.set_frequency(freq) # set signal generator frequency
                        
                        # read samples to buffer
                        odmr_result = read(odmr_buffer[0], ps_seq, kwargs['runs'], kwargs['timeout']) 
                            
                        # partition buffer into signal and background datasets
                        sig = self.analog_math(odmr_result, 'ODMR')[0]
                        bg = self.analog_math(odmr_result, 'ODMR')[1]
                        
                        # read the analog voltage levels received by the APD.
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps[-1][1][f] = sig 
                        signal_sweeps.updated_item(-1) 
                        background_sweeps[-1][1][f] = bg
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server.
                        odmr_data.push({'params': {'kwargs': kwargs},
                                        'title': 'Optically Detected Magnetic Resonance',
                                        'xlabel': 'Frequency (GHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}})

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            if kwargs['save'] == True:
                                flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            
                            self.equipment_off()
                            
                            return

                    pbar.update(1)
                    percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                    time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                    self.queue_from_exp.put_nowait([percent_completed, time_per_iter])
            
            # Automatically save data if action requested.
            if kwargs['save'] == True:
                flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
            
            self.equipment_off()

    def OPT_T1_scan(self, **kwargs):
        """
        Run a T1 sweep without MW over a set of precession time intervals.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as t1_data:
            # load devices used in scan
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG
            dig = mgr.SPECTRUM_Digitizer

            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
            
            t1_buffer = self.generate_buffer('Opt T1', kwargs['runs'], kwargs['num_pts'])
            
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'])
            
            ps_seq = ps.Optical_T1(tau_times)

            # configure devices used in scan
            laser.set_modulation_state('pulsed')
            laser.set_analog_control_mode('current')
            laser.set_diode_current_realtime(kwargs['laser_power'])
            laser.laser_on()
        
            # ps.laser_time = kwargs['init'] * 1e9 # initialization pulse duration
            # ps.readout_time = kwargs['read'] * 1e9 # readout window duration

            daq.open_ai_task(kwargs['detector'], len(t1_buffer[0]))

            dig.assign_param(dig_config)

            # index_order = np.argsort(sorted_taus) 

            # for storing the experiment data
            # list of numpy arrays of shape (2, num_points)
            signal_sweeps = StreamingList()
            background_sweeps = StreamingList()

            with tqdm(total = kwargs['iters']) as pbar:

                for i in range(kwargs['iters']):
                    
                    dig.config()
                    dig.start_buffer()
                    ps.stream(ps_seq, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                    t1_result_raw = obtain(dig.acquire())
                    
                    t1_result = np.mean(t1_result_raw, axis=1)
                    
                    # partition buffer into signal and background datasets
                    sig = self.analog_math(t1_result, 'MW_T1', kwargs['num_pts'])[0]
                    bg = self.analog_math(t1_result, 'MW_T1', kwargs['num_pts'])[1]
                    
                    sig = np.abs(sig)
                    bg = np.abs(bg)
                    
                    # notify the streaminglist that this entry has updated so it will be pushed to the data server                    
                    signal_sweeps.append(np.stack([tau_times/1e6, sig]))
                    signal_sweeps.updated_item(-1) 
                    background_sweeps.append(np.stack([tau_times/1e6, bg]))
                    background_sweeps.updated_item(-1)

                    # save the current data to the data server.
                    t1_data.push({'params': {'kwargs': kwargs},
                                    'title': 'Optical T1 Relaxation',
                                    'xlabel': 'Free Precession Interval (ms)',
                                    'ylabel': 'Signal',
                                    'datasets': {'signal' : signal_sweeps,
                                                'background': background_sweeps}
                    })

                    if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                        # the GUI has asked us nicely to exit
                        if kwargs['save'] == True:
                            flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                        
                        self.equipment_off()

                        return
                    
                    pbar.update(1)
                    percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                    time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                    self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

            if kwargs['save'] == True:
                flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

            self.equipment_off()

    def MW_T1_scan(self, **kwargs):
        """
        Run a T1 sweep with MW over a set of precession time intervals.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as t1_data:
            # load devices used in scan
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG
            dig = mgr.SPECTRUM_Digitizer

            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            np.random.shuffle(tau_times)

            # Generate Rabi data buffer
            t1_buffer = self.generate_buffer('MW T1', kwargs['runs'], kwargs['num_pts'])

            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'])
            
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq'], kwargs['pulse_axis'])

            pi_pulse = kwargs['pi']*1e9 # [ns] units for pulse streamer

            # set initial parameters for instrument server devices
            # configure devices used in scan
            sig_gen.set_frequency(sig_gen_freq)
            sig_gen.set_rf_amplitude(kwargs['rf_power'])
            sig_gen.set_mod_type('QAM')
            sig_gen.set_mod_function('external')
            sig_gen.set_mod_toggle(1)            
            ps_seq = ps.Diff_T1(tau_times, kwargs['pulse_axis'], pi_pulse)

            try:
                hdawg.set_sequence(**{'seq': 'T1',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freq': kwargs['sideband_freq'], 
                                    'iq_phases': iq_phases,
                                    'pi_pulse': pi_pulse/1e9, 
                                    'num_pts': kwargs['num_pts'],
                                    'runs': kwargs['runs'], 
                                    'iters': kwargs['iters']})
                time.sleep(0.5)
            except Exception as e:
                print(e)
            else:
                daq.open_ai_task(kwargs['detector'], len(t1_buffer[0]))

                dig.assign_param(dig_config)

                # for storing the experiment data
                # list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                index_order = np.argsort(tau_times) 
                tau_times_sorted = np.sort(tau_times) # order mw_times for T1 plotting
                
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        dig.config()
                        dig.start_buffer()
                        ps.stream(ps_seq, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                        t1_result_raw = obtain(dig.acquire())
                        
                        t1_result = np.mean(t1_result_raw, axis=1)

                        # partition buffer into signal and background datasets
                        sig = self.analog_math(t1_result, 'MW_T1', kwargs['num_pts'])[0]
                        bg = self.analog_math(t1_result, 'MW_T1', kwargs['num_pts'])[1]
                        
                        sig = np.abs(sig)
                        bg = np.abs(bg)

                        # correct the y-axis data ordering for plots
                        sig = np.array([sig[i] for i in index_order])
                        bg = np.array([bg[i] for i in index_order])

                        # notify the streaminglist that this entry has updated so it will be pushed to the data server                    
                        signal_sweeps.append(np.stack([tau_times_sorted[1:]/1e6, sig[1:]]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([tau_times_sorted[1:]/1e6, bg[1:]]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server.
                        t1_data.push({'params': {'kwargs': kwargs},
                                        'title': 'MW T1 Relaxation',
                                        'xlabel': 'Free Precession Interval (ms)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        # if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                        #     # the GUI has asked us nicely to exit
                        #     # if kwargs['save'] == True:
                        #     #     flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            
                        #     print("Turning off equipment...")
                        #     self.equipment_off()

                        #     return
                        
                        pbar.update(1)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                self.equipment_off()

    def T2_scan(self, **kwargs):
        """
        Run a T2 sweep over a set of precession time intervals.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as t2_data:
            # load devices used in scan
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG
            dig = mgr.SPECTRUM_Digitizer

            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            np.random.shuffle(tau_times)
            # sorted_taus = self.sort_taus(tau_times)

            # Generate Rabi data buffer
            t2_buffer = self.generate_buffer('T2', kwargs['runs'], kwargs['num_pts'])

            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'])
            
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # configure devices used in scan
            sig_gen.set_frequency(sig_gen_freq)
            sig_gen.set_rf_amplitude(kwargs['rf_power'])
            sig_gen.set_mod_type('QAM')
            sig_gen.set_mod_function('external')
            sig_gen.set_mod_toggle(1)

            # ps.laser_time = kwargs['init'] * 1e9 # initialization pulse duration
            # ps.readout_time = kwargs['read'] * 1e9 # readout window duration

            pi: List[float] = []
            pi_half: List[float] = []

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)

            match kwargs['t2_seq']:
                case 'Ramsey':
                    ps_seq = ps.Ramsey(tau_times, pi_half[0], pi_half[1])
                    # x_tau_times = 2*pi_half[0] + x_tau_times

                case 'Echo':
                    ps_seq = ps.Echo(tau_times, pi_half[0], pi_half[1], 
                                            pi[0], pi[1])
                    # x_tau_times = 2*pi_half[0] + x_tau_times + pi[1]

                case 'XY4':
                    ps_seq = ps.XY4_N(tau_times, 'xy', 
                                        pi_half[0], pi_half[1], 
                                        pi[0], pi[1], kwargs['n'])
                    # x_tau_times = 2*pi_half[0] + (2*(x_tau_times/2)/(4*kwargs['n']) + 2*pi[0] + \
                    #             2*pi[1] + 3*x_tau_times/(4*kwargs['n']))*kwargs['n']

                case 'YY4':
                    ps_seq = ps.XY4_N(tau_times, 'yy', 
                                        pi_half[0], pi_half[1], 
                                        pi[0], pi[1], kwargs['n'])
                    # x_tau_times = 2*pi_half[0] + (2*(x_tau_times/2)/(4*kwargs['n']) + 4*pi[1] + 3*x_tau_times/(4*kwargs['n']))*kwargs['n']

                case 'XY8':
                    ps_seq = ps.XY8_N(tau_times, 'xy', 
                                        pi_half[0], pi_half[1], 
                                        pi[0], pi[1], kwargs['n'])
                    # x_tau_times = 2*pi_half[0] + ((x_tau_times/2)/(8*kwargs['n']) + 4*pi[0] + \
                    #             4*pi[1] + 7*x_tau_times/(8*kwargs['n']) + (x_tau_times/2)/(8*kwargs['n']))*kwargs['n']

                case 'YY8':
                    ps_seq = ps.XY8_N(tau_times, 'yy', 
                                        pi_half[0], pi_half[1], 
                                        pi[0], pi[1], kwargs['n'])
                    # x_tau_times = 2*pi_half[0] + ((x_tau_times/2)/(8*kwargs['n']) + 8*pi[1] + \
                    #         7*x_tau_times/(8*kwargs['n']) + (x_tau_times/2)/(8*kwargs['n']))*kwargs['n']

                case 'CPMG':
                    ps_seq = ps.CPMG_N(tau_times, kwargs['pulse_axis'], 
                                        pi_half[0], pi_half[1], 
                                        pi[0], pi[1], kwargs['n'])
                    # x_tau_times = 2*pi_half[0] + x_tau_times/kwargs['n'] + (pi[0] + x_tau_times/kwargs['n'])*(kwargs['n']-1) + pi[0]

                case 'PulsePol':
                    ps_seq = ps.PulsePol(tau_times, 
                                        pi_half[0], pi_half[1], 
                                        pi[0], pi[1], kwargs['n'])
                    # x_tau_times = (x_tau_times + 2*pi_half[1] + pi[0] + 2*pi_half[0] + pi[1])*2*kwargs['n']

            try:
                hdawg.set_sequence(**{'seq': 'T2',
                                    'seq_dd': kwargs['t2_seq'],
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freq': kwargs['sideband_freq'], 
                                    'iq_phases': iq_phases,
                                    'pihalf_x': pi_half[0]/1e9,
                                    'pihalf_y': pi_half[1]/1e9,
                                    'pi_x': pi[0]/1e9, 
                                    'pi_y': pi[1]/1e9,
                                    'n': kwargs['n'],
                                    'num_pts': kwargs['num_pts'],
                                    'runs': kwargs['runs'], 
                                    'iters': kwargs['iters']})
                time.sleep(0.5)
            except Exception as e:
                print(e)
            else:        
                daq.open_ai_task(kwargs['detector'], len(t2_buffer[0]))

                dig.assign_param(dig_config)

                # for storing the experiment data
                # list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                index_order = np.argsort(tau_times) 
                tau_times_sorted = np.sort(tau_times) # order mw_times for T2 plotting

                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        dig.config()
                        dig.start_buffer()
                        ps.stream(ps_seq, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                        t2_result_raw = obtain(dig.acquire())
                        
                        t2_result = np.mean(t2_result_raw, axis=1)
                            
                        # partition buffer into signal and background datasets
                        sig = self.analog_math(t2_result, 'T2', kwargs['num_pts'])[0]
                        bg = self.analog_math(t2_result, 'T2', kwargs['num_pts'])[1]
                        
                        sig = np.abs(sig)
                        bg = np.abs(bg)

                        # correct the y-axis data ordering for plots
                        sig = np.array([sig[i] for i in index_order])
                        bg = np.array([bg[i] for i in index_order])
                        
                        # read the analog voltage levels received by the APD.
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        
                        signal_sweeps.append(np.stack([tau_times_sorted/1e3, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([tau_times_sorted/1e3, bg]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server.
                        t2_data.push({'params': {'kwargs': kwargs},
                                        'title': 'T2 Relaxation',
                                        'xlabel': 'Free Precession Interval (\u03BCs)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit
                            if kwargs['save'] == True:
                                flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            
                            self.equipment_off()

                            return
                        
                        pbar.update(1)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                # self.equipment_off()

    def DQ_scan(self, **kwargs):
        """
        Run a DQ sweep over a set of precession time intervals.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as dq_data:
            # load devices used in scan
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG
            dig = mgr.SPECTRUM_Digitizer

            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            np.random.shuffle(tau_times)

            # Generate DQ data buffer           
            dq_buffer = self.generate_buffer('DQ', kwargs['runs'], kwargs['num_pts'])

            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'])
                
            if kwargs['pulse_axis'] == 'y':
                delta = 90
            else:
                delta = 0
            
            iq_phases = [delta+0, delta+90, delta+90, delta+0] # set IQ phase relations for upper and lower sidebands
                
            sig_gen_freq = (kwargs['freq_minus'] + kwargs['freq_plus'])/2 # set mean value freq to sig gen 

            sideband_freq = sig_gen_freq - kwargs['freq_minus'] # set sideband freq to match the inputted values

            # print("SIG GEN FREQ: ", sig_gen_freq)
            # print("SIDEBAND FREQ: ", sideband_freq)

            pi_pulse_minus = kwargs['pi_minus']*1e9 # [ns] units for pulse streamer
            pi_pulse_plus = kwargs['pi_plus']*1e9 # [ns] units for pulse streamer

            # set initial parameters for instrument server devices
            # configure devices used in scan
            sig_gen.set_frequency(sig_gen_freq)
            sig_gen.set_rf_amplitude(kwargs['rf_power_minus'])
            sig_gen.set_mod_type('QAM')
            sig_gen.set_mod_function('external')
            sig_gen.set_mod_toggle(1)
            ps_seq = ps.DQ(tau_times, kwargs['pulse_axis'], pi_pulse_minus, pi_pulse_plus)

            # ps.laser_time = kwargs['init'] * 1e9 # initialization pulse duration
            # ps.readout_time = kwargs['read'] * 1e9 # readout window duration

            try:
                hdawg.set_sequence(**{'seq': 'DQ',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freq': sideband_freq, 
                                    'iq_phases': iq_phases,
                                    'pi_minus1': pi_pulse_minus/1e9, 
                                    'pi_plus1': pi_pulse_plus/1e9, 
                                    'num_pts': kwargs['num_pts'],
                                    'runs': kwargs['runs'], 
                                    'iters': kwargs['iters']})

                time.sleep(0.5)
            except Exception as e:
                print(e)
            else:
                daq.open_ai_task(kwargs['detector'], len(dq_buffer[0]))

                dig.assign_param(dig_config)

                # for storing the experiment data
                # list of numpy arrays of shape (2, num_points)
                s00_sweeps = StreamingList()
                s0m_sweeps = StreamingList()
                smm_sweeps = StreamingList()
                smp_sweeps = StreamingList()

                index_order = np.argsort(tau_times) 
                tau_times_sorted = np.sort(tau_times) # order mw_times for DQ plotting

                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        dig.config()
                        dig.start_buffer()
                        ps.stream(ps_seq, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                        dq_result_raw = obtain(dig.acquire())
                        
                        dq_result = np.mean(dq_result_raw, axis=1)

                        # partition buffer into signal and background datasets
                        s00 = self.analog_math(dq_result, 'DQ', kwargs['num_pts'])[0] # data for S0,0 sequence
                        s0m = self.analog_math(dq_result, 'DQ', kwargs['num_pts'])[1] # data for S0,-1 sequence
                        smm = self.analog_math(dq_result, 'DQ', kwargs['num_pts'])[2] # data for S-1,-1 sequence
                        smp = self.analog_math(dq_result, 'DQ', kwargs['num_pts'])[3] # data for S-1,+1 sequence
                        
                        s00 = np.abs(s00)
                        s0m = np.abs(s0m)
                        smm = np.abs(smm)
                        smp = np.abs(smp)

                        # correct the y-axis data ordering for plots
                        s00 = np.array([s00[i] for i in index_order])
                        s0m = np.array([s0m[i] for i in index_order])
                        smm = np.array([smm[i] for i in index_order])
                        smp = np.array([smp[i] for i in index_order])

                        s00_sweeps.append(np.stack([tau_times_sorted/1e3, s00]))
                        s00_sweeps.updated_item(-1) 
                        s0m_sweeps.append(np.stack([tau_times_sorted/1e3, s0m]))
                        s0m_sweeps.updated_item(-1)

                        smm_sweeps.append(np.stack([tau_times_sorted/1e3, smm]))
                        smm_sweeps.updated_item(-1) 
                        smp_sweeps.append(np.stack([tau_times_sorted/1e3, smp]))
                        smp_sweeps.updated_item(-1)

                        # save the current data to the data server.
                        dq_data.push({'params': {'kwargs': kwargs},
                                        'title': 'T1 Relaxation',
                                        'xlabel': 'Free Precession Interval (\u03BCs)',
                                        'ylabel': 'Signal',
                                        'datasets': {'S0,0' : s00_sweeps,
                                                    'S0,-1': s0m_sweeps,
                                                    'S-1,-1' : smm_sweeps,
                                                    'S-1,+1': smp_sweeps}
                        })

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit
                            if kwargs['save'] == True:
                                flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            
                            self.equipment_off()

                            return
                        
                        pbar.update(1)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                self.equipment_off()

    def DEER_scan(self, **kwargs):
        """
        Run a DEER sweep over a set of MW frequencies.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as deer_data:
            # load devices used in scan
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG
            dig = mgr.SPECTRUM_Digitizer

            frequencies = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])            
            
            np.random.shuffle(frequencies)

            # Generate Rabi data buffer
            deer_buffer = self.generate_buffer('DEER', kwargs['runs'], kwargs['num_pts'])

            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'])
            
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # configure devices used in scan
            sig_gen.set_frequency(sig_gen_freq)
            sig_gen.set_rf_amplitude(kwargs['rf_power'])
            sig_gen.set_mod_type('QAM')
            sig_gen.set_mod_function('external')
            sig_gen.set_mod_toggle(1)            
            
            # ps.laser_time = kwargs['init'] * 1e9 # initialization pulse duration
            # ps.readout_time = kwargs['read'] * 1e9 # readout window duration

            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)

            if kwargs['drive_type'] == 'Continuous':
                ps_seq = ps.DEER_CD(pi_half[0], pi_half[1], 
                                    pi[0], pi[1], 
                                    kwargs['tau']*1e9, kwargs['num_pts']) # send to PS in [ns] units
                dark_pulse = kwargs['pi']/2 + kwargs['tau'] + kwargs['pi'] + kwargs['tau'] + kwargs['pi']/2 # send to AWG in [s] units 
            else:
                ps_seq = ps.DEER(pi_half[0], pi_half[1], 
                                    pi[0], pi[1], 
                                    kwargs['tau']*1e9, kwargs['num_pts'])
                dark_pulse = dark_pi
                
            try:
                if kwargs['drive_type'] == 'Continuous':
                    hdawg.set_sequence(**{'seq': 'DEER CD',
                                        'i_offset': kwargs['i_offset'],
                                        'q_offset': kwargs['q_offset'],
                                        'sideband_power': kwargs['sideband_power'],
                                        'sideband_freq': kwargs['sideband_freq'], 
                                        'iq_phases': iq_phases,
                                        'pihalf_x': pi_half[0]/1e9,
                                        'pihalf_y': pi_half[1]/1e9,
                                        'pi_x': pi[0]/1e9, 
                                        'pi_y': pi[1]/1e9,
                                        'pi_pulse': dark_pulse, 
                                        'mw_power': kwargs['awg_power'], 
                                        'num_pts': kwargs['num_pts'],
                                        'runs': kwargs['runs'], 
                                        'iters': kwargs['iters'],
                                        'freqs': frequencies})
                else:
                    hdawg.set_sequence(**{'seq': 'DEER',
                                        'i_offset': kwargs['i_offset'],
                                        'q_offset': kwargs['q_offset'],
                                        'sideband_power': kwargs['sideband_power'],
                                        'sideband_freq': kwargs['sideband_freq'], 
                                        'iq_phases': iq_phases,
                                        'pihalf_x': pi_half[0]/1e9,
                                        'pihalf_y': pi_half[1]/1e9,
                                        'pi_x': pi[0]/1e9, 
                                        'pi_y': pi[1]/1e9,
                                        'pi_pulse': dark_pulse, 
                                        'mw_power': kwargs['awg_power'], 
                                        'num_pts': kwargs['num_pts'],
                                        'runs': kwargs['runs'], 
                                        'iters': kwargs['iters'],
                                        'freqs': frequencies})
                time.sleep(0.5)
            except Exception as e:
                print(e)
            else:
                daq.open_ai_task(kwargs['detector'], len(deer_buffer[0]))

                dig.assign_param(dig_config)

                # for storing the experiment data
                # list of numpy arrays of shape (2, num_points)
                dark_signal_sweeps = StreamingList()
                dark_background_sweeps = StreamingList()
                echo_signal_sweeps = StreamingList()
                echo_background_sweeps = StreamingList()

                index_order = np.argsort(frequencies) 
                frequencies_sorted = np.sort(frequencies) # order mw_times for DEER plotting

                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):

                        dig.config()
                        dig.start_buffer()
                        ps.stream(ps_seq, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                        deer_result_raw = obtain(dig.acquire())
                        
                        deer_result = np.mean(deer_result_raw, axis=1)
                            
                        # partition buffer into signal and background datasets
                        dark_sig = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[0]
                        dark_bg = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[1]
                        echo_sig = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[2]
                        echo_bg = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[3]
                        
                        dark_sig = np.abs(dark_sig)
                        dark_bg = np.abs(dark_bg)
                        echo_sig = np.abs(echo_sig)
                        echo_bg = np.abs(echo_bg)

                        # correct the y-axis data ordering for plots
                        dark_sig = np.array([dark_sig[i] for i in index_order])
                        dark_bg = np.array([dark_bg[i] for i in index_order])
                        echo_sig = np.array([echo_sig[i] for i in index_order])
                        echo_bg = np.array([echo_bg[i] for i in index_order])

                        # read the analog voltage levels received by the APD.
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        
                        dark_signal_sweeps.append(np.stack([frequencies_sorted/1e6, dark_sig]))
                        dark_signal_sweeps.updated_item(-1) 
                        dark_background_sweeps.append(np.stack([frequencies_sorted/1e6, dark_bg]))
                        dark_background_sweeps.updated_item(-1)
                        echo_signal_sweeps.append(np.stack([frequencies_sorted/1e6, echo_sig]))
                        echo_signal_sweeps.updated_item(-1) 
                        echo_background_sweeps.append(np.stack([frequencies_sorted/1e6, echo_bg]))
                        echo_background_sweeps.updated_item(-1)

                        # save the current data to the data server.
                        deer_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER',
                                        'xlabel': 'Surface Electron Resonance (MHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'dark_signal' : dark_signal_sweeps,
                                                    'dark_background': dark_background_sweeps,
                                                    'echo_signal' : echo_signal_sweeps,
                                                    'echo_background': echo_background_sweeps,}
                        })

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit
                            if kwargs['save'] == True:
                                flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            
                            self.equipment_off()

                            return
                        
                        pbar.update(1)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                # self.equipment_off()

    def DEER_rabi_scan(self, **kwargs):
        """
        Run a DEER Rabi sweep over a set of MW pulse durations.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as deer_data:
            # load devices used in scan
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG
            dig = mgr.SPECTRUM_Digitizer

            dark_taus = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9            

            np.random.shuffle(dark_taus)

            # Generate Rabi data buffer
            deer_buffer = self.generate_buffer('DEER', kwargs['runs'], kwargs['num_pts'])

            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'])
            
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # configure devices used in scan
            sig_gen.set_frequency(sig_gen_freq)
            sig_gen.set_rf_amplitude(kwargs['rf_power'])
            sig_gen.set_mod_type('QAM')
            sig_gen.set_mod_function('external')
            sig_gen.set_mod_toggle(1)

            # ps.laser_time = kwargs['init'] * 1e9 # initialization pulse duration
            # ps.readout_time = kwargs['read'] * 1e9 # readout window duration

            pi: List[float] = []
            pi_half: List[float] = []

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)

            ps_seq = ps.DEER_Rabi(pi_half[0], pi_half[1], 
                                pi[0], pi[1], 
                                kwargs['tau']*1e9, kwargs['num_pts'])
            
            try:
                hdawg.set_sequence(**{'seq': 'DEER Rabi',     
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freq': kwargs['sideband_freq'], 
                                    'iq_phases': iq_phases,
                                    'pihalf_x': pi_half[0]/1e9,
                                    'pihalf_y': pi_half[1]/1e9,
                                    'pi_x': pi[0]/1e9, 
                                    'pi_y': pi[1]/1e9,
                                    'dark_freq': kwargs['dark_freq'],
                                    'mw_power': kwargs['awg_power'], 
                                    'num_pts': kwargs['num_pts'],
                                    'runs': kwargs['runs'], 
                                    'iters': kwargs['iters'],
                                    'pi_pulses': dark_taus/1e9})
                time.sleep(0.5)
            except Exception as e:
                print(e)
            else:
                daq.open_ai_task(kwargs['detector'], len(deer_buffer[0]))

                dig.assign_param(dig_config)

                # for storing the experiment data
                # list of numpy arrays of shape (2, num_points)
                dark_signal_sweeps = StreamingList()
                dark_background_sweeps = StreamingList()
                echo_signal_sweeps = StreamingList()
                echo_background_sweeps = StreamingList()

                index_order = np.argsort(dark_taus) 
                dark_taus_sorted = np.sort(dark_taus) # order mw_times for Rabi plotting

                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):

                        dig.config()
                        dig.start_buffer()
                        ps.stream(ps_seq, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                        deer_result_raw = obtain(dig.acquire())
                        
                        deer_result = np.mean(deer_result_raw, axis=1)
                            
                        # partition buffer into signal and background datasets
                        dark_sig = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[0]
                        dark_bg = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[1]
                        echo_sig = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[2]
                        echo_bg = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[3]

                        dark_sig = np.abs(dark_sig)
                        dark_bg = np.abs(dark_bg)
                        echo_sig = np.abs(echo_sig)
                        echo_bg = np.abs(echo_bg)

                        # correct the y-axis data ordering for plots
                        dark_sig = np.array([dark_sig[i] for i in index_order])
                        dark_bg = np.array([dark_bg[i] for i in index_order])
                        echo_sig = np.array([echo_sig[i] for i in index_order])
                        echo_bg = np.array([echo_bg[i] for i in index_order])

                        # read the analog voltage levels received by the APD.
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        # print("dark taus: ", dark_taus/1e9)
                        dark_signal_sweeps.append(np.stack([dark_taus_sorted, dark_sig]))
                        dark_signal_sweeps.updated_item(-1) 
                        dark_background_sweeps.append(np.stack([dark_taus_sorted, dark_bg]))
                        dark_background_sweeps.updated_item(-1)
                        echo_signal_sweeps.append(np.stack([dark_taus_sorted, echo_sig]))
                        echo_signal_sweeps.updated_item(-1) 
                        echo_background_sweeps.append(np.stack([dark_taus_sorted, echo_bg]))
                        echo_background_sweeps.updated_item(-1)

                        # save the current data to the data server.
                        deer_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER Rabi',
                                        'xlabel': 'MW Pulse Duration (ns)',
                                        'ylabel': 'Signal',
                                        'datasets': {'dark_signal' : dark_signal_sweeps,
                                                    'dark_background': dark_background_sweeps,
                                                    'echo_signal' : echo_signal_sweeps,
                                                    'echo_background': echo_background_sweeps,}
                        })

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit
                            if kwargs['save'] == True:
                                flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            
                            self.equipment_off()

                            return
                        
                        pbar.update(1)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                self.equipment_off()

    def DEER_FID_scan(self, **kwargs):
        """
        Run a DEER FID sweep over a set of MW pulse durations.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as deer_data:
            # load devices used in scan
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG
            dig = mgr.SPECTRUM_Digitizer

            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # np.random.shuffle(tau_times)

            # Generate Rabi data buffer
            deer_buffer = self.generate_buffer('DEER', kwargs['runs'], kwargs['num_pts'])

            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'])
            
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # configure devices used in scan
            sig_gen.set_frequency(sig_gen_freq)
            sig_gen.set_rf_amplitude(kwargs['rf_power'])
            sig_gen.set_mod_type('QAM')
            sig_gen.set_mod_function('external')
            sig_gen.set_mod_toggle(1)

            # ps.laser_time = kwargs['init'] * 1e9 # initialization pulse duration
            # ps.readout_time = kwargs['read'] * 1e9 # readout window duration

            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            ps_seq = ps.DEER_FID(tau_times, pi_half[0], pi_half[1], 
                                pi[0], pi[1], kwargs['n'])

            try:
                hdawg.set_sequence(**{'seq': 'DEER FID',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freq': kwargs['sideband_freq'], 
                                    'iq_phases': iq_phases,
                                    'pihalf_x': pi_half[0]/1e9,
                                    'pihalf_y': pi_half[1]/1e9,
                                    'pi_x': pi[0]/1e9, 
                                    'pi_y': pi[1]/1e9,  
                                    'pi_pulse': dark_pi, 
                                    'dark_freq': kwargs['dark_freq'],
                                    'mw_power': kwargs['awg_power'], 
                                    'num_pts': kwargs['num_pts'],
                                    'n': kwargs['n'],
                                    'runs': kwargs['runs'], 
                                    'iters': kwargs['iters']})
                    
                time.sleep(0.5)
            except Exception as e:
                print(e)
            else:
                daq.open_ai_task(kwargs['detector'], len(deer_buffer[0]))
                
                dig.assign_param(dig_config)

                # for storing the experiment data
                # list of numpy arrays of shape (2, num_points)
                dark_signal_sweeps = StreamingList()
                dark_background_sweeps = StreamingList()
                echo_signal_sweeps = StreamingList()
                echo_background_sweeps = StreamingList()

                # index_order = np.argsort(tau_times) 
                # tau_times_sorted = np.sort(tau_times) # order mw_times for FID plotting

                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):

                        dig.config()
                        dig.start_buffer()
                        ps.stream(ps_seq, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                        deer_result_raw = obtain(dig.acquire())
                        
                        deer_result = np.mean(deer_result_raw, axis=1)
                            
                        # partition buffer into signal and background datasets
                        dark_sig = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[0]
                        dark_bg = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[1]
                        echo_sig = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[2]
                        echo_bg = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[3]

                        dark_sig = np.abs(dark_sig)
                        dark_bg = np.abs(dark_bg)
                        echo_sig = np.abs(echo_sig)
                        echo_bg = np.abs(echo_bg)

                        # correct the y-axis data ordering for plots
                        # dark_sig = np.array([dark_sig[i] for i in index_order])
                        # dark_bg = np.array([dark_bg[i] for i in index_order])
                        # echo_sig = np.array([echo_sig[i] for i in index_order])
                        # echo_bg = np.array([echo_bg[i] for i in index_order])
                        
                        # read the analog voltage levels received by the APD.
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        
                        dark_signal_sweeps.append(np.stack([tau_times, dark_sig]))
                        dark_signal_sweeps.updated_item(-1) 
                        dark_background_sweeps.append(np.stack([tau_times, dark_bg]))
                        dark_background_sweeps.updated_item(-1)
                        echo_signal_sweeps.append(np.stack([tau_times, echo_sig]))
                        echo_signal_sweeps.updated_item(-1) 
                        echo_background_sweeps.append(np.stack([tau_times, echo_bg]))
                        echo_background_sweeps.updated_item(-1)

                        # save the current data to the data server.
                        deer_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER FID',
                                        'xlabel': 'Free Precession Interval (ns)',
                                        'ylabel': 'Signal',
                                        'datasets': {'dark_signal' : dark_signal_sweeps,
                                                    'dark_background': dark_background_sweeps,
                                                    'echo_signal' : echo_signal_sweeps,
                                                    'echo_background': echo_background_sweeps,}
                        })

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit
                            if kwargs['save'] == True:
                                flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            
                            self.equipment_off()

                            return
                        
                        pbar.update(1)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                self.equipment_off()

    def DEER_FID_CD_scan(self, **kwargs):
        """
        Run a DEER FID CD sweep over a set of MW pulse durations.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as cd_data:
            # load devices used in scan
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG
            dig = mgr.SPECTRUM_Digitizer

            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            np.random.shuffle(tau_times)

            # Generate Rabi data buffer
            deer_buffer = self.generate_buffer('CD', kwargs['runs'], kwargs['num_pts'])

            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'])

            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # configure devices used in scan
            sig_gen.set_frequency(sig_gen_freq)
            sig_gen.set_rf_amplitude(kwargs['rf_power'])
            sig_gen.set_mod_type('QAM')
            sig_gen.set_mod_function('external')
            sig_gen.set_mod_toggle(1)

            # ps.laser_time = kwargs['init'] * 1e9 # initialization pulse duration
            # ps.readout_time = kwargs['read'] * 1e9 # readout window duration

            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)


            ps_seq = ps.DEER_FID_CD(tau_times, pi_half[0], pi_half[1], 
                                    pi[0], pi[1], kwargs['n'])
            # dark_pulses = kwargs['pi']/2 + tau_times/1e9 + (kwargs['pi'] + 2*tau_times/1e9)*(kwargs['n']-1) + kwargs['pi'] + tau_times/1e9 + kwargs['pi']/2 

            try:
                hdawg.set_sequence(**{'seq': 'DEER FID CD',                
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freq': kwargs['sideband_freq'], 
                                    'iq_phases': iq_phases,
                                    'pihalf_x': pi_half[0]/1e9,
                                    'pihalf_y': pi_half[1]/1e9,
                                    'pi_x': pi[0]/1e9, 
                                    'pi_y': pi[1]/1e9,
                                    'dark_freq': kwargs['dark_freq'],
                                    'pi_pulse': dark_pi,
                                    'taus': tau_times/1e9,
                                    'mw_power': kwargs['awg_power'],
                                    'cd_mw_power': kwargs['awg_cd_power'], 
                                    'num_pts': kwargs['num_pts'],
                                    'n': kwargs['n'],
                                    'runs': kwargs['runs'], 
                                    'iters': kwargs['iters']})
                    
                time.sleep(0.5)
            except Exception as e:
                print(e)
            else:
                daq.open_ai_task(kwargs['detector'], len(deer_buffer[0]))

                dig.assign_param(dig_config)

                # for storing the experiment data
                # list of numpy arrays of shape (2, num_points)
                dark_signal_sweeps = StreamingList()
                dark_background_sweeps = StreamingList()
                echo_signal_sweeps = StreamingList()
                echo_background_sweeps = StreamingList()
                cd_signal_sweeps = StreamingList()
                cd_background_sweeps = StreamingList()
                
                index_order = np.argsort(tau_times) 
                tau_times_sorted = np.sort(tau_times) # order mw_times for FID plotting

                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):

                        dig.config()
                        dig.start_buffer()
                        ps.stream(ps_seq, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                        cd_result_raw = obtain(dig.acquire())
                        
                        cd_result = np.mean(cd_result_raw, axis=1)
                            
                        # partition buffer into signal and background datasets
                        dark_sig = self.analog_math(cd_result, 'CD', kwargs['num_pts'])[0]
                        dark_bg = self.analog_math(cd_result, 'CD', kwargs['num_pts'])[1]
                        echo_sig = self.analog_math(cd_result, 'CD', kwargs['num_pts'])[2]
                        echo_bg = self.analog_math(cd_result, 'CD', kwargs['num_pts'])[3]
                        cd_sig = self.analog_math(cd_result, 'CD', kwargs['num_pts'])[4]
                        cd_bg = self.analog_math(cd_result, 'CD', kwargs['num_pts'])[5]

                        dark_sig = np.abs(dark_sig)
                        dark_bg = np.abs(dark_bg)
                        echo_sig = np.abs(echo_sig)
                        echo_bg = np.abs(echo_bg)
                        cd_sig = np.abs(cd_sig)
                        cd_bg = np.abs(cd_bg)

                        # correct the y-axis data ordering for plots
                        dark_sig = np.array([dark_sig[i] for i in index_order])
                        dark_bg = np.array([dark_bg[i] for i in index_order])
                        echo_sig = np.array([echo_sig[i] for i in index_order])
                        echo_bg = np.array([echo_bg[i] for i in index_order])
                        cd_sig = np.array([cd_sig[i] for i in index_order])
                        cd_bg = np.array([cd_bg[i] for i in index_order])

                        # read the analog voltage levels received by the APD.
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        
                        dark_signal_sweeps.append(np.stack([tau_times_sorted, dark_sig]))
                        dark_signal_sweeps.updated_item(-1) 
                        dark_background_sweeps.append(np.stack([tau_times_sorted, dark_bg]))
                        dark_background_sweeps.updated_item(-1)
                        echo_signal_sweeps.append(np.stack([tau_times_sorted, echo_sig]))
                        echo_signal_sweeps.updated_item(-1) 
                        echo_background_sweeps.append(np.stack([tau_times_sorted, echo_bg]))
                        echo_background_sweeps.updated_item(-1)
                        cd_signal_sweeps.append(np.stack([tau_times_sorted, cd_sig]))
                        cd_signal_sweeps.updated_item(-1) 
                        cd_background_sweeps.append(np.stack([tau_times_sorted, cd_bg]))
                        cd_background_sweeps.updated_item(-1)

                        # save the current data to the data server.
                        cd_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER FID Continuous Drive',
                                        'xlabel': 'Free Precession Interval (ns)',
                                        'ylabel': 'Signal',
                                        'datasets': {'dark_signal' : dark_signal_sweeps,
                                                    'dark_background': dark_background_sweeps,
                                                    'echo_signal' : echo_signal_sweeps,
                                                    'echo_background': echo_background_sweeps,
                                                    'cd_signal' : cd_signal_sweeps,
                                                    'cd_background': cd_background_sweeps,}
                        })

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit
                            if kwargs['save'] == True:
                                flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            
                            self.equipment_off()

                            return
                        
                        pbar.update(1)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                self.equipment_off()

    def DEER_corr_rabi_scan(self, **kwargs):
        """
        Run a DEER Correlation Rabi sweep over a set of MW pulses.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as corr_rabi_data:
            # load devices used in scan
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG
            dig = mgr.SPECTRUM_Digitizer

            dark_taus = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])          

            np.random.shuffle(dark_taus)

            # Generate Rabi data buffer
            corr_buffer = self.generate_buffer('Corr', kwargs['runs'], kwargs['num_pts'])

            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'])
            
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # configure devices used in scan
            sig_gen.set_frequency(sig_gen_freq)
            sig_gen.set_rf_amplitude(kwargs['rf_power'])
            sig_gen.set_mod_type('QAM')
            sig_gen.set_mod_function('external')
            sig_gen.set_mod_toggle(1)
              
            # ps.laser_time = kwargs['init'] * 1e9 # initialization pulse duration
            # ps.readout_time = kwargs['read'] * 1e9 # readout window duration

            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            ps_seq = ps.DEER_Corr_Rabi(dark_taus*1e9, kwargs['tau']*1e9, kwargs['t_corr']*1e9, pi_half[0], pi_half[1], 
                                pi[0], pi[1]) # send to PS in [ns] units
            
            try:
                hdawg.set_sequence(**{'seq': 'DEER Corr Rabi',                
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freq': kwargs['sideband_freq'], 
                                    'iq_phases': iq_phases,
                                    'pihalf_x': pi_half[0]/1e9,
                                    'pihalf_y': pi_half[1]/1e9,
                                    'pi_x': pi[0]/1e9, 
                                    'pi_y': pi[1]/1e9,
                                    'dark_freq': kwargs['dark_freq'],
                                    'dark_pulse': dark_pi,
                                    'mw_power': kwargs['awg_power'], 
                                    'num_pts': kwargs['num_pts'],
                                    'runs': kwargs['runs'], 
                                    'iters': kwargs['iters'],
                                    'pi_pulses': dark_taus})
                time.sleep(0.5)
            except Exception as e:
                print(e)
            else:
                daq.open_ai_task(kwargs['detector'], len(corr_buffer[0]))

                dig.assign_param(dig_config)

                # for storing the experiment data
                # list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                index_order = np.argsort(dark_taus) 
                dark_taus_sorted = np.sort(dark_taus) # order mw_times for Rabi plotting

                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):

                        dig.config()
                        dig.start_buffer()
                        ps.stream(ps_seq, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                        corr_result_raw = obtain(dig.acquire())
                        
                        corr_result = np.mean(corr_result_raw, axis=1)
                            
                        # partition buffer into signal and background datasets
                        sig = self.analog_math(corr_result, 'Corr', kwargs['num_pts'])[0]
                        bg = self.analog_math(corr_result, 'Corr', kwargs['num_pts'])[1]

                        sig = np.abs(sig)
                        bg = np.abs(bg)

                        # correct the y-axis data ordering for plots
                        sig = np.array([sig[i] for i in index_order])
                        bg = np.array([bg[i] for i in index_order])

                        # read the analog voltage levels received by the APD.
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        
                        signal_sweeps.append(np.stack([dark_taus_sorted*1e9, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([dark_taus_sorted*1e9, bg]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server.
                        corr_rabi_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER Correlation Rabi',
                                        'xlabel': 'MW Pulse Duration (ns)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit
                            if kwargs['save'] == True:
                                flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            
                            self.equipment_off()

                            return
                        
                        pbar.update(1)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                self.equipment_off()

    def DEER_corr_T1_scan(self, **kwargs):
        """
        Run a DEER Correlation T1 sweep over a set of correlation intervals.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as corr_t1_data:
            # load devices used in scan
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG
            dig = mgr.SPECTRUM_Digitizer

            match kwargs['array_type']:
                case 'geomspace':
                    t_corr_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])     
                case 'linspace':
                    t_corr_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])     

            np.random.shuffle(t_corr_times)

            # Generate Rabi data buffer
            corr_buffer = self.generate_buffer('DEER', kwargs['runs'], kwargs['num_pts'])

            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'])
            
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # configure devices used in scan
            sig_gen.set_frequency(sig_gen_freq)
            sig_gen.set_rf_amplitude(kwargs['rf_power'])
            sig_gen.set_mod_type('QAM')
            sig_gen.set_mod_function('external')
            sig_gen.set_mod_toggle(1)            
            
            # ps.laser_time = kwargs['init'] * 1e9 # initialization pulse duration
            # ps.readout_time = kwargs['read'] * 1e9 # readout window duration

            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # ps_seq = ps.DEER_Corr_T1(t_corr_times*1e9, kwargs['tau']*1e9, pi_half[0], pi_half[1], 
            #                     pi[0], pi[1], dark_pi*1e9) # send to PS in [ns] units
            ps_seq = ps.DEER_Corr_T1(t_corr_times*1e9, kwargs['tau']*1e9, pi_half[0], pi_half[1], 
                                pi[0], pi[1], dark_pi*1e9) # send to PS in [ns] units
            try:
                hdawg.set_sequence(**{'seq': 'DEER Corr T1',                  
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freq': kwargs['sideband_freq'], 
                                    'iq_phases': iq_phases,
                                    'pihalf_x': pi_half[0]/1e9,
                                    'pihalf_y': pi_half[1]/1e9,
                                    'pi_x': pi[0]/1e9, 
                                    'pi_y': pi[1]/1e9,
                                    'dark_freq': kwargs['dark_freq'],
                                    'dark_pulse': dark_pi,
                                    'mw_power': kwargs['awg_power'], 
                                    'num_pts': kwargs['num_pts'],
                                    'runs': kwargs['runs'], 
                                    'iters': kwargs['iters']})
                time.sleep(0.5)
            except Exception as e:
                print(e)
            else:
                daq.open_ai_task(kwargs['detector'], len(corr_buffer[0]))

                dig.assign_param(dig_config)

                # for storing the experiment data
                # list of numpy arrays of shape (2, num_points)
                with_pulse_py_sweeps = StreamingList()
                without_pulse_py_sweeps = StreamingList()
                with_pulse_ny_sweeps = StreamingList()
                without_pulse_ny_sweeps = StreamingList()

                index_order = np.argsort(t_corr_times) 
                t_corr_times_sorted = np.sort(t_corr_times) # order mw_times for T1 plotting

                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):

                        dig.config()
                        dig.start_buffer()
                        ps.stream(ps_seq, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                        corr_result_raw = obtain(dig.acquire())
                        
                        corr_result = np.mean(corr_result_raw, axis=1)
                            
                        # partition buffer into signal and background datasets
                        with_py = self.analog_math(corr_result, 'DEER', kwargs['num_pts'])[0]
                        without_py = self.analog_math(corr_result, 'DEER', kwargs['num_pts'])[1]
                        with_ny = self.analog_math(corr_result, 'DEER', kwargs['num_pts'])[2]
                        without_ny = self.analog_math(corr_result, 'DEER', kwargs['num_pts'])[3]

                        with_py = np.abs(with_py)
                        without_py = np.abs(without_py)
                        with_ny = np.abs(with_ny)
                        without_ny = np.abs(without_ny)

                        # correct the y-axis data ordering for plots
                        with_py = np.array([with_py[i] for i in index_order])
                        without_py = np.array([without_py[i] for i in index_order])
                        with_ny = np.array([with_ny[i] for i in index_order])
                        without_ny = np.array([without_ny[i] for i in index_order])

                        # read the analog voltage levels received by the APD.
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        
                        with_pulse_py_sweeps.append(np.stack([t_corr_times_sorted*1e6, with_py]))
                        with_pulse_py_sweeps.updated_item(-1) 
                        without_pulse_py_sweeps.append(np.stack([t_corr_times_sorted*1e6, without_py]))
                        without_pulse_py_sweeps.updated_item(-1)
                        with_pulse_ny_sweeps.append(np.stack([t_corr_times_sorted*1e6, with_ny]))
                        with_pulse_ny_sweeps.updated_item(-1) 
                        without_pulse_ny_sweeps.append(np.stack([t_corr_times_sorted*1e6, without_ny]))
                        without_pulse_ny_sweeps.updated_item(-1)

                        # save the current data to the data server.
                        corr_t1_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER Correlation T1',
                                        'xlabel': 'Free Precession Interval (\u03BCs) or Frequency (MHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'with_py': with_pulse_py_sweeps,
                                                    'without_py': without_pulse_py_sweeps,
                                                    'with_ny': with_pulse_ny_sweeps,
                                                    'without_ny': without_pulse_ny_sweeps}
                        })

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit
                            if kwargs['save'] == True:
                                flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            
                            self.equipment_off()

                            return
                        
                        pbar.update(1)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                self.equipment_off()

    def Corr_Spec_scan(self, **kwargs):
        """
        Run a Correlation Spectroscopy NMR sweep over a set of precession time intervals.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as nmr_data:
            # load devices used in scan
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG
            dig = mgr.SPECTRUM_Digitizer

            t_corr_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            np.random.shuffle(t_corr_times)

            # Generate Rabi data buffer
            nmr_buffer = self.generate_buffer('NMR', kwargs['runs'], kwargs['num_pts'])

            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'])
            
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for x pulse by default

            # configure devices used in scan
            sig_gen.set_frequency(sig_gen_freq)
            sig_gen.set_rf_amplitude(kwargs['rf_power'])
            sig_gen.set_mod_type('QAM')
            sig_gen.set_mod_function('external')
            sig_gen.set_mod_toggle(1)

            # ps.laser_time = kwargs['init'] * 1e9 # initialization pulse duration
            # ps.readout_time = kwargs['read'] * 1e9 # readout window duration

            pi: List[float] = []
            pi_half: List[float] = []

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)

            ps_seq = ps.Corr_Spectroscopy(t_corr_times, kwargs['tau']*1e9, 
                                     pi_half[0], pi_half[1], 
                                     pi[0], pi[1], kwargs['n'])

            try:
                hdawg.set_sequence(**{'seq': 'NMR',
                                    'seq_nmr': 'Correlation Spectroscopy',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freq': kwargs['sideband_freq'], 
                                    'iq_phases': iq_phases,
                                    'pihalf_x': pi_half[0]/1e9,
                                    'pihalf_y': pi_half[1]/1e9,
                                    'pi_x': pi[0]/1e9, 
                                    'pi_y': pi[1]/1e9,
                                    'n': kwargs['n'],
                                    'num_pts': kwargs['num_pts'],
                                    'runs': kwargs['runs'], 
                                    'iters': kwargs['iters']})
                time.sleep(0.5)
            except Exception as e:
                print(e)
            else:  
                daq.open_ai_task(kwargs['detector'], len(nmr_buffer[0]))

                dig.assign_param(dig_config)

                # for storing the experiment data
                # list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()
                
                index_order = np.argsort(t_corr_times) 
                t_corr_times_sorted = np.sort(t_corr_times) # order mw_times for T1 plotting

                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):

                        dig.config()
                        dig.start_buffer()
                        ps.stream(ps_seq, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                        nmr_result_raw = obtain(dig.acquire())
                        
                        nmr_result = np.mean(nmr_result_raw, axis=1)

                        # partition buffer into signal and background datasets
                        sig = self.analog_math(nmr_result, 'NMR', kwargs['num_pts'])[0]
                        bg = self.analog_math(nmr_result, 'NMR', kwargs['num_pts'])[1]

                        sig = np.abs(sig)
                        bg = np.abs(bg)

                        # correct the y-axis data ordering for plots
                        sig = np.array([sig[i] for i in index_order])
                        bg = np.array([bg[i] for i in index_order])

                        # read the analog voltage levels received by the APD.
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        
                        signal_sweeps.append(np.stack([t_corr_times_sorted/1e3, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([t_corr_times_sorted/1e3, bg]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server.
                        nmr_data.push({'params': {'kwargs': kwargs},
                                        'title': 'NMR Time Domain Data',
                                        'xlabel': 'Free Precession Interval (\u03BCs) or Frequency (MHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit
                            if kwargs['save'] == True:
                                flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            
                            self.equipment_off()

                            return
                        
                        pbar.update(1)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                # self.equipment_off()

    def CASR_scan(self, **kwargs):
        """
        Run a coherent averaged synchronized readout (CASR) NMR experiment.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as casr_data:
            # load devices used in scan
            laser = mgr.OXXIUS_Laser
            sig_gen = mgr.SRS_SG396
            ps = mgr.SWABIAN_Pulse_Streamer
            daq = mgr.NI_DAQ
            hdawg = mgr.ZI_HDAWG
            dig = mgr.SPECTRUM_Digitizer

            t_corr_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            np.random.shuffle(t_corr_times)

            # Generate Rabi data buffer
            casr_buffer = self.generate_buffer('NMR', kwargs['runs'], kwargs['num_pts'])

            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'])
            
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for x pulse by default

            # configure devices used in scan
            sig_gen.set_frequency(sig_gen_freq)
            sig_gen.set_rf_amplitude(kwargs['rf_power'])
            sig_gen.set_mod_type('QAM')
            sig_gen.set_mod_function('external')
            sig_gen.set_mod_toggle(1)

            # ps.laser_time = kwargs['init'] * 1e9 # initialization pulse duration
            # ps.readout_time = kwargs['read'] * 1e9 # readout window duration

            pi: List[float] = []
            pi_half: List[float] = []

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)

            ps_seq = ps.Corr_Spectroscopy(t_corr_times, kwargs['tau']*1e9, 
                                     pi_half[0], pi_half[1], 
                                     pi[0], pi[1], kwargs['n'])

            try:
                hdawg.set_sequence(**{'seq': 'NMR',
                                    'seq_nmr': 'Correlation Spectroscopy',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freq': kwargs['sideband_freq'], 
                                    'iq_phases': iq_phases,
                                    'pihalf_x': pi_half[0]/1e9,
                                    'pihalf_y': pi_half[1]/1e9,
                                    'pi_x': pi[0]/1e9, 
                                    'pi_y': pi[1]/1e9,
                                    'n': kwargs['n'],
                                    'num_pts': kwargs['num_pts'],
                                    'runs': kwargs['runs'], 
                                    'iters': kwargs['iters']})
                time.sleep(0.5)
            except Exception as e:
                print(e)
            else:  
                daq.open_ai_task(kwargs['detector'], len(casr_buffer[0]))

                dig.assign_param(dig_config)

                # for storing the experiment data
                # list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()
                
                index_order = np.argsort(t_corr_times) 
                t_corr_times_sorted = np.sort(t_corr_times) # order mw_times for T1 plotting

                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):

                        dig.config()
                        dig.start_buffer()
                        ps.stream(ps_seq, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                        nmr_result_raw = obtain(dig.acquire())
                        
                        nmr_result = np.mean(nmr_result_raw, axis=1)
                            
                        # partition buffer into signal and background datasets
                        sig = self.analog_math(nmr_result, 'NMR', kwargs['num_pts'])[0]
                        bg = self.analog_math(nmr_result, 'NMR', kwargs['num_pts'])[1]

                        sig = np.abs(sig)
                        bg = np.abs(bg)

                        # correct the y-axis data ordering for plots
                        sig = np.array([sig[i] for i in index_order])
                        bg = np.array([bg[i] for i in index_order])

                        # read the analog voltage levels received by the APD.
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        
                        signal_sweeps.append(np.stack([t_corr_times_sorted/1e3, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([t_corr_times_sorted/1e3, bg]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server.
                        casr_data.push({'params': {'kwargs': kwargs},
                                        'title': 'NMR Time Domain Data',
                                        'xlabel': 'Free Precession Interval (\u03BCs) or Frequency (MHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit
                            if kwargs['save'] == True:
                                flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            
                            self.equipment_off()

                            return
                        
                        pbar.update(1)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                # self.equipment_off()