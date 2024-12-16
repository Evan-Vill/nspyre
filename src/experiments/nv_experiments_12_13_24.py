"""
This is example script demonstrates most of the basic functionality of nspyre.
"""
import time
import logging
# import spcm
from spcm import units
from pathlib import Path
from typing import List

from pulsestreamer import PulseStreamer


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
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig

            laser.laser_off()
            laser_shutter.close_shutter()

            sig_gen.set_rf_toggle(0)
            sig_gen.set_mod_toggle(0)

            ps.Pulser.reset()
            daq.close_do_task()
            daq.close_ao_task()
            daq.close_ai_task()
            dig.stop_card()
            dig.reset()
            hdawg.set_disabled()

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
                      'num_iters': kwargs['iters'], # number of exp. iterations
                      'segment_size' : kwargs['segment_size'] ,
                      'pretrig_size' : kwargs['pretrig_size'] ,
                      'readout_ch' : int(kwargs['read_channel']),
                      'runs' : kwargs['runs'],
                      'ACCOUPLE': kwargs['coupling'],
                      'HF_INPUT_50OHM': kwargs['termination']}
    
        return dig_config
    
    ### Experiment logic

    def sigvstime_scan(self, **kwargs):
        
        with InstrumentManager() as mgr, DataSource('sigvstime') as sigvstime_data:
            # run laser on continuously here from laser driver
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            ps = mgr.ps
            daq = mgr.daq
            dig = mgr.dig

            ps_seq = ps.SigvsTime(1/kwargs['sampling_rate'] * 1e9) # pulse streamer sequence for CW ODMR

            # configure digitizer (need to set impedance = 1 Mohm)
            dig_config = self.digitizer_configure(num_pts_in_exp = 1, iters = 1, segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = 400, coupling = 0, termination = 1)
            
            # daq.open_ai_task(kwargs['sigvstime_detector'], n_runs) # one clock per each of the "n_runs" no. of sequences

            time_start = time.time()

            signal_sweeps = StreamingList()

            # open laser shutter
            laser_shutter.open_shutter()
            # daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))
            time.sleep(0.5)
            
            # upload digitizer parameters
            dig.assign_param(dig_config)

            # set pulsestreamer to start on software trigger & run infinitely
            ps.set_soft_trigger()
            ps.stream(ps_seq, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
            
            # start digitizer --> waits for trigger from pulse sequence
            dig.config()
            dig.start_buffer()
            
            # start pulse sequence
            ps.start_now()
            
            for i in range(10000):
                
                sig_result_raw = obtain(dig.acquire()) # acquire data from digitizer

                # average all data over each trigger/segment 
                sig_result=np.mean(sig_result_raw,axis=1)
                segments=(np.shape(sig_result))[0]
                
                sig_result = np.mean(sig_result)

                time_pt = time.time() - time_start

                # read the analog voltage levels received by the APD.
                # notify the streaminglist that this entry has updated so it will be pushed to the data server
                signal_sweeps.append(np.array([[time_pt], [sig_result]]))
                signal_sweeps.updated_item(-1) 
                
                # save the current data to the data server.
                sigvstime_data.push({'title': 'Signal Vs Time',
                                'xlabel': 'Time step',
                                'ylabel': 'APD Voltage (V)',
                                'datasets': {'signal': signal_sweeps}})

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
        with InstrumentManager() as mgr, DataSource('odmr') as cw_odmr_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig
            
            # check to make sure number of points is even 
            # assert kwargs['num_pts'] % 2 == 0 

            # define NV drive frequency & sideband           
            delta = 0
            iq_phases = [delta+0, delta+90] # set IQ phase relations for lower sideband [lower I, lower Q]
                
            sig_gen_freq = kwargs['center_freq'] + kwargs['half_span_sideband_freq'] # set freq to sig gen 

            max_sideband_freq = 2*kwargs['half_span_sideband_freq'] # set span of ODMR sweep as max sideband modulation frequency --> 100 MHz max. for SG396 IQ bandwidth

            # define parameter array that will be swept over in experiment & shuffle
            mod_freqs = np.linspace(0, max_sideband_freq, kwargs['num_pts'])            
            mod_freqs = np.flip(mod_freqs)
            
            np.random.shuffle(mod_freqs)
            
            real_freqs = sig_gen_freq - mod_freqs
            
            # define pulse sequence
            ps_seq = ps.CW_ODMR(kwargs['num_pts']) # pulse streamer sequence for CW ODMR
            ps.probe_time = kwargs['probe'] * 1e9

            # configure digitizer (need to set impedance = 1 Mohm)
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 0)
            
            # configure signal generator for NV drive
            sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
            sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode
            
            try:
                hdawg.set_sequence(**{'seq': 'CW ODMR',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'probe_length': kwargs['probe'], 
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freqs': mod_freqs, 
                                    'iq_phases': iq_phases,
                                    'num_pts': kwargs['num_pts']})
                    
                time.sleep(0.5)
            except Exception as e:
                print(e)
            
            # run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # sort sweep parameters back to proper plotting order
                index_order = np.argsort(real_freqs) 
                freqs_sorted = np.sort(real_freqs) 

                # open laser shutter
                laser_shutter.open_shutter()
                # daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))
                time.sleep(0.5)

                # upload digitizer parameters
                dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on TODO: change back to CW
                # laser.set_modulation_state('cw')
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(ps_seq, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                dig.config()
                dig.start_buffer()
                
                # start pulse sequence
                ps.start_now()

                total_segments=kwargs['iters']*kwargs['runs']*kwargs['num_pts']*2

                # start experiment loop
                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        odmr_result_raw = obtain(dig.acquire()) # acquire data from digitizer

                        # average all data over each trigger/segment 
                        odmr_result=np.mean(odmr_result_raw,axis=1)
                        segments=(np.shape(odmr_result))[0]
                        # odmr_result=np.ravel(odmr_result_raw)
                        # odmr_result=np.ravel(odmr_result_raw)[250::512]
                        
                        # print("shape of result array: ", np.shape(odmr_result))
                        # partition buffer into signal and background datasets
                        sig = self.analog_math(odmr_result, 'CW ODMR', kwargs['num_pts'])[0]
                        bg = self.analog_math(odmr_result, 'CW ODMR', kwargs['num_pts'])[1]

                        # take absolute value of voltages    
                        sig = np.abs(sig)
                        bg = np.abs(bg)

                        # correct the y-axis data ordering for plots
                        sig = np.array([sig[i] for i in index_order])
                        bg = np.array([bg[i] for i in index_order])
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([freqs_sorted/1e9, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([freqs_sorted/1e9, bg]))
                        background_sweeps.updated_item(-1)

                        # signal_sweeps.append(np.stack([np.arange(len(np.ravel(odmr_result_raw))), np.ravel(odmr_result_raw)]))
                        # signal_sweeps.updated_item(-1) 
                        # background_sweeps.append(np.stack([np.arange(len(np.ravel(odmr_result_raw))), np.ravel(odmr_result_raw)]))
                        # background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        cw_odmr_data.push({'params': {'kwargs': kwargs},
                                            'title': 'CW Optically Detected Magnetic Resonance',
                                            'xlabel': 'Frequency (GHz)',
                                            'ylabel': 'Signal',
                                            'datasets': {'signal' : signal_sweeps,
                                                        'background': background_sweeps}})

                        # update GUI progress bar                        
                        pbar.update(segments/total_segments)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                # save data if requested upon completion of experiment
                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
            
            finally:
                self.equipment_off() # turn off equipment regardless of if experiment started or failed

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
        with InstrumentManager() as mgr, DataSource('rabi') as rabi_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig
            
            # define parameter array that will be swept over in experiment & shuffle
            mw_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
            np.random.shuffle(mw_times)

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq'], kwargs['pulse_axis'])

            # define pulse sequence
            ps_seq = ps.Rabi(mw_times) # pulse streamer sequence

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 1)
            
            # configure signal generator for NV drive
            sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
            sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode

            # upload AWG sequence first
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
                time.sleep(0.1)

            except Exception as e:
                print(e)
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # sort sweep parameters back to proper plotting order
                index_order = np.argsort(mw_times) 
                mw_times_sorted = np.sort(mw_times) 

                # open laser shutter
                laser_shutter.open_shutter()
                # daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))
                time.sleep(0.5)

                # upload digitizer parameters
                dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(ps_seq, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                dig.config()
                dig.start_buffer()
                
                # start pulse sequence
                ps.start_now()

                total_segments=kwargs['iters']*kwargs['runs']*kwargs['num_pts']*2

                # start experiment loop
                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        rabi_result_raw = obtain(dig.acquire()) # acquire data from digitizer
                        
                        # print("shape of raw result: ", np.shape(rabi_result_raw))
                        # average all data over each trigger/segment 
                        rabi_result=np.mean(rabi_result_raw,axis=1)
                        segments=(np.shape(rabi_result))[0]
                        # print("shape of result array: ", np.shape(rabi_result))

                        # partition buffer into signal and background datasets
                        sig = self.analog_math(rabi_result, 'Rabi', kwargs['num_pts'])[0]
                        bg = self.analog_math(rabi_result, 'Rabi', kwargs['num_pts'])[1]

                        # take absolute value of voltages    
                        sig = np.abs(sig)
                        bg = np.abs(bg)

                        # correct the y-axis data ordering for plots
                        sig = np.array([sig[i] for i in index_order])
                        bg = np.array([bg[i] for i in index_order])
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([mw_times_sorted, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([mw_times_sorted, bg]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        rabi_data.push({'params': {'kwargs': kwargs},
                                        'title': 'Rabi Oscillation',
                                        'xlabel': 'MW Pulse Duration (ns)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        # update GUI progress bar                        
                        pbar.update(segments/total_segments)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                # save data if requested upon completion of experiment
                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

            finally:
                self.equipment_off() # turn off equipment regardless of if experiment started or failed

    # def pulsed_odmr_scan(self, **kwargs):
    #     """
    #     Run a Pulsed ODMR sweep over a set of microwave frequencies.

    #     Keyword args:
    #         dataset: name of the dataset to push data to
    #         start (float): start frequency
    #         stop (float): stop frequency
    #         num_pts (int): number of points between start-stop (inclusive)
    #         iterations: number of times to repeat the experiment
    #     """
    #     # connect to the instrument server & the data server.
    #     # create a data set, or connect to an existing one with the same name if it was created earlier.
    #     with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as pulsed_odmr_data:
    #         # load devices used in scan
    #         laser = mgr.laser
    #         laser_shutter = mgr.laser_shutter
    #         sig_gen = mgr.sg
    #         ps = mgr.ps
    #         daq = mgr.daq
    #         hdawg = mgr.awg
    #         dig = mgr.dig
            
    #         # define NV drive frequency & sideband           
    #         delta = 0
    #         iq_phases = [delta+0, delta+90] # set IQ phase relations for lower sideband [lower I, lower Q]
                
    #         sig_gen_freq = kwargs['freq'] # set freq to sig gen 

    #         max_sideband_freq = kwargs['max_sideband_freq'] # set span of ODMR sweep as max sideband modulation frequency --> 100 MHz max. for SG396 IQ bandwidth

    #         # define parameter array that will be swept over in experiment & shuffle
    #         mod_freqs = np.linspace(0, max_sideband_freq, kwargs['num_pts'])            
    #         mod_freqs = np.flip(mod_freqs)
            
    #         np.random.shuffle(mod_freqs)
      
    #         real_freqs = sig_gen_freq - mod_freqs

    #         pi_pulse = kwargs['pi']*1e9 # [ns] units for pulse streamer
    #         modified_pulse_time = 35000 + 2000

    #         # define pulse sequence
    #         ps_seq = ps.Pulsed_ODMR(kwargs['num_pts'], pi_pulse) # pulse streamer sequence

    #         # configure digitizer
    #         dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], segment_size = kwargs['segment_size'], 
    #                                               pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 1)
            
    #         # configure signal generator for NV drive
    #         sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
    #         sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
    #         sig_gen.set_mod_type(7) # quadrature amplitude modulation
    #         sig_gen.set_mod_subtype(1) # no constellation mapping
    #         sig_gen.set_mod_function('IQ', 5) # external modulation
    #         sig_gen.set_mod_toggle(1) # turn on modulation mode

    #         # upload AWG sequence first
    #         try:
    #             hdawg.set_sequence(**{'seq': 'Pulsed ODMR',
    #                                 'i_offset': kwargs['i_offset'],
    #                                 'q_offset': kwargs['q_offset'],
    #                                 'sideband_power': kwargs['sideband_power'],
    #                                 'sideband_freqs': mod_freqs, 
    #                                 'iq_phases': iq_phases,
    #                                 # 'pi_pulse': pi_pulse/1e9, 
    #                                 'pi_pulse': modified_pulse_time/1e9, 
    #                                 'num_pts': kwargs['num_pts'],
    #                                 'runs': kwargs['runs'], 
    #                                 'iters': kwargs['iters']})                  
    #             time.sleep(0.5)

    #         except Exception as e:
    #             print(e)
            
    #         # if successfully uploaded, run the experiment
    #         else:
    #             # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
    #             signal_sweeps = StreamingList()
    #             background_sweeps = StreamingList()

    #             # sort sweep parameters back to proper plotting order
    #             index_order = np.argsort(real_freqs) 
    #             freqs_sorted = np.sort(real_freqs) 

    #             # open laser shutter
    #             laser_shutter.open_shutter()
    #             # daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))
    #             time.sleep(0.5)

    #             # upload digitizer parameters
    #             dig.assign_param(dig_config)

    #             # emit MW for NV drive
    #             sig_gen.set_rf_toggle(1) # turn on NV signal generator

    #             # configure laser settings and turn on
    #             laser.set_modulation_state('pulsed')
    #             laser.set_analog_control_mode('current')
    #             laser.set_diode_current_realtime(kwargs['laser_power'])
    #             laser.laser_on()

    #             # set pulsestreamer to start on software trigger & run infinitely
    #             ps.set_soft_trigger()
    #             ps.stream(ps_seq, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
    #             # start digitizer --> waits for trigger from pulse sequence
    #             dig.config()
    #             dig.start_buffer()
                
    #             # start pulse sequence
    #             ps.start_now()

    #             total_segments=kwargs['iters']*kwargs['runs']*kwargs['num_pts']*2

    #             # start experiment loop
    #             with tqdm(total = kwargs['iters']) as pbar:

    #                 for i in range(kwargs['iters']):
                        
    #                     pulsed_odmr_result_raw = obtain(dig.acquire()) # acquire data from digitizer
                        
    #                     print("shape of raw result: ", np.shape(pulsed_odmr_result_raw))
    #                     # average all data over each trigger/segment 
    #                     pulsed_odmr_result=np.mean(pulsed_odmr_result_raw,axis=1)
    #                     segments=(np.shape(pulsed_odmr_result))[0]
    #                     print("shape of result array: ", np.shape(pulsed_odmr_result))

    #                     # partition buffer into signal and background datasets
    #                     sig = self.analog_math(pulsed_odmr_result, 'Pulsed ODMR', kwargs['num_pts'])[0]
    #                     bg = self.analog_math(pulsed_odmr_result, 'Pulsed ODMR', kwargs['num_pts'])[1]

    #                     # take absolute value of voltages    
    #                     sig = np.abs(sig)
    #                     bg = np.abs(bg)

    #                     # correct the y-axis data ordering for plots
    #                     sig = np.array([sig[i] for i in index_order])
    #                     bg = np.array([bg[i] for i in index_order])
                        
    #                     # notify the streaminglist that this entry has updated so it will be pushed to the data server
    #                     signal_sweeps.append(np.stack([freqs_sorted/1e9, sig]))
    #                     signal_sweeps.updated_item(-1) 
    #                     background_sweeps.append(np.stack([freqs_sorted/1e9, bg]))
    #                     background_sweeps.updated_item(-1)

    #                     # save the current data to the data server
    #                     pulsed_odmr_data.push({'params': {'kwargs': kwargs},
    #                                     'title': 'CW Optically Detected Magnetic Resonance',
    #                                     'xlabel': 'Frequency (GHz)',
    #                                     'ylabel': 'Signal',
    #                                     'datasets': {'signal' : signal_sweeps,
    #                                                 'background': background_sweeps}
    #                     })

    #                     # update GUI progress bar                        
    #                     pbar.update(segments/total_segments)
    #                     percent_completed = str(int(((i+1)/kwargs['iters'])*100))
    #                     time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
    #                     self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

    #             # save data if requested upon completion of experiment
    #             if kwargs['save'] == True:
    #                 flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

    #         finally:
    #             self.equipment_off() # turn off equipment regardless of if experiment started or failed

    def pulsed_odmr_scan(self, **kwargs):
        """
        Run a Pulsed ODMR sweep over a set of microwave frequencies.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource('odmr') as pulsed_odmr_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig
            
            # define NV drive frequency & sideband           
            delta = 0
            iq_phases = [delta+0, delta+90] # set IQ phase relations for lower sideband [lower I, lower Q]
                
            sig_gen_freq = kwargs['center_freq'] + kwargs['half_span_sideband_freq'] # set freq to sig gen 

            max_sideband_freq = 2*kwargs['half_span_sideband_freq'] # set span of ODMR sweep as max sideband modulation frequency --> 100 MHz max. for SG396 IQ bandwidth

            # define parameter array that will be swept over in experiment & shuffle
            mod_freqs = np.linspace(0, max_sideband_freq, kwargs['num_pts'])            
            mod_freqs = np.flip(mod_freqs)
            
            np.random.shuffle(mod_freqs)
      
            real_freqs = sig_gen_freq - mod_freqs

            pi_pulse = kwargs['pi']*1e9 # [ns] units for pulse streamer

            # define pulse sequence
            ps_seq = ps.Pulsed_ODMR(kwargs['num_pts'], pi_pulse) # pulse streamer sequence

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 1)
            
            # configure signal generator for NV drive
            sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
            sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'Pulsed ODMR',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freqs': mod_freqs, 
                                    'iq_phases': iq_phases,
                                    'pi_pulse': pi_pulse/1e9, 
                                    'num_pts': kwargs['num_pts'],
                                    'runs': kwargs['runs'], 
                                    'iters': kwargs['iters']})                  
                time.sleep(0.5)

            except Exception as e:
                print(e)
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # sort sweep parameters back to proper plotting order
                index_order = np.argsort(real_freqs) 
                freqs_sorted = np.sort(real_freqs) 

                # open laser shutter
                laser_shutter.open_shutter()
                # daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))
                time.sleep(0.5)

                # upload digitizer parameters
                dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(ps_seq, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                dig.config()
                dig.start_buffer()
                
                # start pulse sequence
                ps.start_now()

                total_segments=kwargs['iters']*kwargs['runs']*kwargs['num_pts']*2

                # start experiment loop
                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        pulsed_odmr_result_raw = obtain(dig.acquire()) # acquire data from digitizer
                        
                        print("shape of raw result: ", np.shape(pulsed_odmr_result_raw))
                        # average all data over each trigger/segment 
                        pulsed_odmr_result=np.mean(pulsed_odmr_result_raw,axis=1)
                        segments=(np.shape(pulsed_odmr_result))[0]
                        print("shape of result array: ", np.shape(pulsed_odmr_result))

                        # partition buffer into signal and background datasets
                        sig = self.analog_math(pulsed_odmr_result, 'Pulsed ODMR', kwargs['num_pts'])[0]
                        bg = self.analog_math(pulsed_odmr_result, 'Pulsed ODMR', kwargs['num_pts'])[1]

                        # take absolute value of voltages    
                        sig = np.abs(sig)
                        bg = np.abs(bg)

                        # correct the y-axis data ordering for plots
                        sig = np.array([sig[i] for i in index_order])
                        bg = np.array([bg[i] for i in index_order])
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([freqs_sorted/1e9, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([freqs_sorted/1e9, bg]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        pulsed_odmr_data.push({'params': {'kwargs': kwargs},
                                        'title': 'CW Optically Detected Magnetic Resonance',
                                        'xlabel': 'Frequency (GHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        # update GUI progress bar                        
                        pbar.update(segments/total_segments)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                # save data if requested upon completion of experiment
                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

            finally:
                self.equipment_off() # turn off equipment regardless of if experiment started or failed

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
        with InstrumentManager() as mgr, DataSource('t1') as t1_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig

            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
            
            t1_buffer = self.generate_buffer('Opt T1', kwargs['runs'], kwargs['num_pts'])
            
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 1)
            
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
                    
                    pbar.update(segments/total_segments)
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
        with InstrumentManager() as mgr, DataSource('t1') as t1_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig
            
            # define parameter array that will be swept over in experiment & shuffle
            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq'], kwargs['pulse_axis'])

            pi_pulse = kwargs['pi']*1e9 # [ns] units for pulse streamer

            # define pulse sequence
            ps_seq = ps.Diff_T1(tau_times, kwargs['pulse_axis'], pi_pulse)

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 1)
            
            # configure signal generator for NV drive
            sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
            sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode

            # upload AWG sequence first
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
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()
                # daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))
                time.sleep(0.5)

                # upload digitizer parameters
                dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(ps_seq, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                dig.config()
                dig.start_buffer()
                
                # start pulse sequence
                ps.start_now()

                total_segments=kwargs['iters']*kwargs['runs']*kwargs['num_pts']*2

                # start experiment loop
                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        t1_result_raw = obtain(dig.acquire()) # acquire data from digitizer

                        # define dummy array to contain experiment data --> size (runs*num_pts) --> (segment_size)
                        t1_result=np.mean(t1_result_raw,axis=1)
                        segments=(np.shape(t1_result))[0]

                        # partition buffer into signal and background datasets
                        sig = self.analog_math(t1_result, 'MW_T1', kwargs['num_pts'])[0]
                        bg = self.analog_math(t1_result, 'MW_T1', kwargs['num_pts'])[1]

                        # take absolute value of voltages    
                        sig = np.abs(sig)
                        bg = np.abs(bg)
                      
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([tau_times[1:]/1e6, sig[1:]]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([tau_times[1:]/1e6, bg[1:]]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        t1_data.push({'params': {'kwargs': kwargs},
                                        'title': 'MW T1 Relaxation',
                                        'xlabel': 'Free Precession Interval (ms)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        # update GUI progress bar                        
                        pbar.update(segments/total_segments)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                # save data if requested upon completion of experiment
                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

            finally:
                self.equipment_off() # turn off equipment regardless of if experiment started or failed

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
        with InstrumentManager() as mgr, DataSource('t2') as t2_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig
            
            # define parameter array that will be swept over in experiment & shuffle
            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq'])

            pi: List[float] = []
            pi_half: List[float] = []

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)

            # define pulse sequence
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

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 1)
            
            # configure signal generator for NV drive
            sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
            sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode

            # upload AWG sequence first
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
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()
                # daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))
                time.sleep(0.5)

                # upload digitizer parameters
                dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(ps_seq, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                dig.config()
                dig.start_buffer()
                
                # start pulse sequence
                ps.start_now()

                total_segments=kwargs['iters']*kwargs['runs']*kwargs['num_pts']*2

                # start experiment loop
                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        t2_result_raw = obtain(dig.acquire()) # acquire data from digitizer

                        # define dummy array to contain experiment data --> size (runs*num_pts) --> (segment_size)
                        t2_result=np.mean(t2_result_raw,axis=1)
                        segments=(np.shape(t2_result))[0]

                        # partition buffer into signal and background datasets
                        sig = self.analog_math(t2_result, 'T2', kwargs['num_pts'])[0]
                        bg = self.analog_math(t2_result, 'T2', kwargs['num_pts'])[1]

                        # take absolute value of voltages    
                        sig = np.abs(sig)
                        bg = np.abs(bg)
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([tau_times, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([tau_times, bg]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        t2_data.push({'params': {'kwargs': kwargs},
                                        'title': 'T2 Relaxation',
                                        'xlabel': 'Free Precession Interval (\u03BCs)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        # update GUI progress bar                        
                        pbar.update(segments/total_segments)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                # save data if requested upon completion of experiment
                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

            finally:
                self.equipment_off() # turn off equipment regardless of if experiment started or failed
    
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
        with InstrumentManager() as mgr, DataSource('dq') as dq_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig
            
            # define parameter array that will be swept over in experiment & shuffle
            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # define NV drive frequency & sideband
            if kwargs['pulse_axis'] == 'y':
                delta = 90
            else:
                delta = 0
            
            iq_phases = [delta+0, delta+90, delta+90, delta+0] # set IQ phase relations for upper and lower sidebands
                
            sig_gen_freq = (kwargs['freq_minus'] + kwargs['freq_plus'])/2 # set mean value freq to sig gen 

            sideband_freq = sig_gen_freq - kwargs['freq_minus'] # set sideband freq to match the inputted values

            pi_pulse_minus = kwargs['pi_minus']*1e9 # [ns] units for pulse streamer
            pi_pulse_plus = kwargs['pi_plus']*1e9 # [ns] units for pulse streamer

            # define pulse sequence
            ps_seq = ps.DQ(tau_times, kwargs['pulse_axis'], pi_pulse_minus, pi_pulse_plus)

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 1)
            
            # configure signal generator for NV drive
            sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
            sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode

            # upload AWG sequence first
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
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                s00_sweeps = StreamingList()
                s0m_sweeps = StreamingList()
                smm_sweeps = StreamingList()
                smp_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()
                # daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))
                time.sleep(0.5)

                # upload digitizer parameters
                dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(ps_seq, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                dig.config()
                dig.start_buffer()
                
                # start pulse sequence
                ps.start_now()

                total_segments=kwargs['iters']*kwargs['runs']*kwargs['num_pts']*2

                # start experiment loop
                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        dq_result_raw = obtain(dig.acquire()) # acquire data from digitizer

                        # define dummy array to contain experiment data --> size (runs*num_pts) --> (segment_size)
                        dq_result = np.zeros(len(dq_result_raw)//dig_num_to_avg)
                        for i in range(0,len(dq_result_raw)//dig_num_to_avg):
                            dq_result[i] = np.mean(dq_result_raw[i*dig_num_to_avg:(i+1)*dig_num_to_avg])

                        # partition buffer into signal and background datasets
                        s00 = self.analog_math(dq_result, 'DQ', kwargs['num_pts'])[0] # data for S0,0 sequence
                        s0m = self.analog_math(dq_result, 'DQ', kwargs['num_pts'])[1] # data for S0,-1 sequence
                        smm = self.analog_math(dq_result, 'DQ', kwargs['num_pts'])[2] # data for S-1,-1 sequence
                        smp = self.analog_math(dq_result, 'DQ', kwargs['num_pts'])[3] # data for S-1,+1 sequence
                        
                        s00 = np.abs(s00)
                        s0m = np.abs(s0m)
                        smm = np.abs(smm)
                        smp = np.abs(smp)
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        s00_sweeps.append(np.stack([tau_times/1e3, s00]))
                        s00_sweeps.updated_item(-1) 
                        s0m_sweeps.append(np.stack([tau_times/1e3, s0m]))
                        s0m_sweeps.updated_item(-1)

                        smm_sweeps.append(np.stack([tau_times/1e3, smm]))
                        smm_sweeps.updated_item(-1) 
                        smp_sweeps.append(np.stack([tau_times/1e3, smp]))
                        smp_sweeps.updated_item(-1)

                        # save the current data to the data server
                        dq_data.push({'params': {'kwargs': kwargs},
                                        'title': 'T1 Relaxation',
                                        'xlabel': 'Free Precession Interval (\u03BCs)',
                                        'ylabel': 'Signal',
                                        'datasets': {'S0,0' : s00_sweeps,
                                                    'S0,-1': s0m_sweeps,
                                                    'S-1,-1' : smm_sweeps,
                                                    'S-1,+1': smp_sweeps}
                        })

                        # update GUI progress bar                        
                        pbar.update(segments/total_segments)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                # save data if requested upon completion of experiment
                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

            finally:
                self.equipment_off() # turn off equipment regardless of if experiment started or failed
    
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
        with InstrumentManager() as mgr, DataSource('deer') as deer_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig
            
            # define parameter array that will be swept over in experiment & shuffle
            frequencies = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])            

            np.random.shuffle(frequencies)

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
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

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 1)
            
            # configure signal generator for NV drive
            sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
            sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode

            # upload AWG sequence first
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
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                dark_signal_sweeps = StreamingList()
                dark_background_sweeps = StreamingList()
                echo_signal_sweeps = StreamingList()
                echo_background_sweeps = StreamingList()

                # sort sweep parameters back to proper plotting order
                index_order = np.argsort(frequencies) 
                frequencies_sorted = np.sort(frequencies) # order mw_times for DEER plotting

                # open laser shutter
                laser_shutter.open_shutter()
                # daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))
                time.sleep(0.5)

                # upload digitizer parameters
                dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(ps_seq, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                dig.config()
                dig.start_buffer()
                
                # start pulse sequence
                ps.start_now()

                total_segments=kwargs['iters']*kwargs['runs']*kwargs['num_pts']*2

                # start experiment loop
                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        deer_result_raw = obtain(dig.acquire()) # acquire data from digitizer

                        # define dummy array to contain experiment data --> size (runs*num_pts) --> (segment_size)
                        deer_result = np.mean(deer_result_raw,axis=1)
                        segments = (np.shape(deer_result))[0]

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
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        dark_signal_sweeps.append(np.stack([frequencies_sorted/1e6, dark_sig]))
                        dark_signal_sweeps.updated_item(-1) 
                        dark_background_sweeps.append(np.stack([frequencies_sorted/1e6, dark_bg]))
                        dark_background_sweeps.updated_item(-1)
                        echo_signal_sweeps.append(np.stack([frequencies_sorted/1e6, echo_sig]))
                        echo_signal_sweeps.updated_item(-1) 
                        echo_background_sweeps.append(np.stack([frequencies_sorted/1e6, echo_bg]))
                        echo_background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        deer_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER',
                                        'xlabel': 'Surface Electron Resonance (MHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'dark_signal' : dark_signal_sweeps,
                                                    'dark_background': dark_background_sweeps,
                                                    'echo_signal' : echo_signal_sweeps,
                                                    'echo_background': echo_background_sweeps,}
                        })

                        # update GUI progress bar                        
                        pbar.update(segments/total_segments)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                # save data if requested upon completion of experiment
                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

            finally:
                self.equipment_off() # turn off equipment regardless of if experiment started or failed

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
        with InstrumentManager() as mgr, DataSource('deer rabi') as deer_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig
            
            # define parameter array that will be swept over in experiment & shuffle
            dark_taus = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9            

            np.random.shuffle(dark_taus)

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
            ps_seq = ps.DEER_Rabi(pi_half[0], pi_half[1], 
                                pi[0], pi[1], 
                                kwargs['tau']*1e9, kwargs['num_pts'])

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 1)
            
            # configure signal generator for NV drive
            sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
            sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode

            # upload AWG sequence first
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
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                dark_signal_sweeps = StreamingList()
                dark_background_sweeps = StreamingList()
                echo_signal_sweeps = StreamingList()
                echo_background_sweeps = StreamingList()

                # sort sweep parameters back to proper plotting order
                index_order = np.argsort(dark_taus) 
                dark_taus_sorted = np.sort(dark_taus) # order mw_times for Rabi plotting

                # open laser shutter
                laser_shutter.open_shutter()
                # daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))
                time.sleep(0.5)

                # upload digitizer parameters
                dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(ps_seq, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                dig.config()
                dig.start_buffer()
                
                # start pulse sequence
                ps.start_now()

                total_segments=kwargs['iters']*kwargs['runs']*kwargs['num_pts']*2

                # start experiment loop
                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        deer_result_raw = obtain(dig.acquire()) # acquire data from digitizer

                        # define dummy array to contain experiment data --> size (runs*num_pts) --> (segment_size)
                        deer_result = np.zeros(len(deer_result_raw)//dig_num_to_avg)
                        for i in range(0,len(deer_result_raw)//dig_num_to_avg):
                            deer_result[i] = np.mean(deer_result_raw[i*dig_num_to_avg:(i+1)*dig_num_to_avg])

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
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        dark_signal_sweeps.append(np.stack([dark_taus_sorted, dark_sig]))
                        dark_signal_sweeps.updated_item(-1) 
                        dark_background_sweeps.append(np.stack([dark_taus_sorted, dark_bg]))
                        dark_background_sweeps.updated_item(-1)
                        echo_signal_sweeps.append(np.stack([dark_taus_sorted, echo_sig]))
                        echo_signal_sweeps.updated_item(-1) 
                        echo_background_sweeps.append(np.stack([dark_taus_sorted, echo_bg]))
                        echo_background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        deer_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER Rabi',
                                        'xlabel': 'MW Pulse Duration (ns)',
                                        'ylabel': 'Signal',
                                        'datasets': {'dark_signal' : dark_signal_sweeps,
                                                    'dark_background': dark_background_sweeps,
                                                    'echo_signal' : echo_signal_sweeps,
                                                    'echo_background': echo_background_sweeps,}
                        })

                        # update GUI progress bar                        
                        pbar.update(segments/total_segments)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                # save data if requested upon completion of experiment
                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

            finally:
                self.equipment_off() # turn off equipment regardless of if experiment started or failed

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
        with InstrumentManager() as mgr, DataSource('fid') as deer_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig
            
            # define parameter array that will be swept over in experiment & shuffle
            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
            ps_seq = ps.DEER_FID(tau_times, pi_half[0], pi_half[1], 
                                pi[0], pi[1], kwargs['n'])
            
            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 1)
            
            # configure signal generator for NV drive
            sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
            sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode

            # upload AWG sequence first
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
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                dark_signal_sweeps = StreamingList()
                dark_background_sweeps = StreamingList()
                echo_signal_sweeps = StreamingList()
                echo_background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()
                # daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))
                time.sleep(0.5)

                # upload digitizer parameters
                dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(ps_seq, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                dig.config()
                dig.start_buffer()
                
                # start pulse sequence
                ps.start_now()

                total_segments=kwargs['iters']*kwargs['runs']*kwargs['num_pts']*2

                # start experiment loop
                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        deer_result_raw = obtain(dig.acquire()) # acquire data from digitizer

                        # define dummy array to contain experiment data --> size (runs*num_pts) --> (segment_size)
                        deer_result = np.zeros(len(deer_result_raw)//dig_num_to_avg)
                        for i in range(0,len(deer_result_raw)//dig_num_to_avg):
                            deer_result[i] = np.mean(deer_result_raw[i*dig_num_to_avg:(i+1)*dig_num_to_avg])

                        # partition buffer into signal and background datasets
                        dark_sig = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[0]
                        dark_bg = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[1]
                        echo_sig = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[2]
                        echo_bg = self.analog_math(deer_result, 'DEER', kwargs['num_pts'])[3]
                        
                        dark_sig = np.abs(dark_sig)
                        dark_bg = np.abs(dark_bg)
                        echo_sig = np.abs(echo_sig)
                        echo_bg = np.abs(echo_bg)
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        dark_signal_sweeps.append(np.stack([tau_times, dark_sig]))
                        dark_signal_sweeps.updated_item(-1) 
                        dark_background_sweeps.append(np.stack([tau_times, dark_bg]))
                        dark_background_sweeps.updated_item(-1)
                        echo_signal_sweeps.append(np.stack([tau_times, echo_sig]))
                        echo_signal_sweeps.updated_item(-1) 
                        echo_background_sweeps.append(np.stack([tau_times, echo_bg]))
                        echo_background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        deer_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER FID',
                                        'xlabel': 'Free Precession Interval (ns)',
                                        'ylabel': 'Signal',
                                        'datasets': {'dark_signal' : dark_signal_sweeps,
                                                    'dark_background': dark_background_sweeps,
                                                    'echo_signal' : echo_signal_sweeps,
                                                    'echo_background': echo_background_sweeps,}
                        })

                        # update GUI progress bar                        
                        pbar.update(segments/total_segments)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                # save data if requested upon completion of experiment
                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

            finally:
                self.equipment_off() # turn off equipment regardless of if experiment started or failed

    def DEER_FID_CD_scan(self, **kwargs):
        """
        Run a continuous drive DEER FID sweep over a set of free precession intervals.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource('fid cd') as cd_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig
            
            # define parameter array that will be swept over in experiment & shuffle
            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
            ps_seq = ps.DEER_FID_CD(tau_times, pi_half[0], pi_half[1], 
                                    pi[0], pi[1], kwargs['n'])
            # dark_pulses = kwargs['pi']/2 + tau_times/1e9 + (kwargs['pi'] + 2*tau_times/1e9)*(kwargs['n']-1) + kwargs['pi'] + tau_times/1e9 + kwargs['pi']/2 
            
            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 1)
            
            # configure signal generator for NV drive
            sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
            sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode

            # upload AWG sequence first
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
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                dark_signal_sweeps = StreamingList()
                dark_background_sweeps = StreamingList()
                echo_signal_sweeps = StreamingList()
                echo_background_sweeps = StreamingList()
                cd_signal_sweeps = StreamingList()
                cd_background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()
                # daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))
                time.sleep(0.5)

                # upload digitizer parameters
                dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(ps_seq, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                dig.config()
                dig.start_buffer()
                
                # start pulse sequence
                ps.start_now()

                total_segments=kwargs['iters']*kwargs['runs']*kwargs['num_pts']*2

                # start experiment loop
                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        cd_result_raw = obtain(dig.acquire()) # acquire data from digitizer

                        # define dummy array to contain experiment data --> size (runs*num_pts) --> (segment_size)
                        cd_result = np.zeros(len(cd_result_raw)//dig_num_to_avg)
                        for i in range(0,len(cd_result_raw)//dig_num_to_avg):
                            cd_result[i] = np.mean(cd_result_raw[i*dig_num_to_avg:(i+1)*dig_num_to_avg])

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

                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        dark_signal_sweeps.append(np.stack([tau_times, dark_sig]))
                        dark_signal_sweeps.updated_item(-1) 
                        dark_background_sweeps.append(np.stack([tau_times, dark_bg]))
                        dark_background_sweeps.updated_item(-1)
                        echo_signal_sweeps.append(np.stack([tau_times, echo_sig]))
                        echo_signal_sweeps.updated_item(-1) 
                        echo_background_sweeps.append(np.stack([tau_times, echo_bg]))
                        echo_background_sweeps.updated_item(-1)
                        cd_signal_sweeps.append(np.stack([tau_times, cd_sig]))
                        cd_signal_sweeps.updated_item(-1) 
                        cd_background_sweeps.append(np.stack([tau_times, cd_bg]))
                        cd_background_sweeps.updated_item(-1)

                        # save the current data to the data server
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

                        # update GUI progress bar                        
                        pbar.update(segments/total_segments)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                # save data if requested upon completion of experiment
                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

            finally:
                self.equipment_off() # turn off equipment regardless of if experiment started or failed

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
        with InstrumentManager() as mgr, DataSource('corr rabi') as corr_rabi_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig
            
            # define parameter array that will be swept over in experiment & shuffle
            dark_taus = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])          

            np.random.shuffle(dark_taus)

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
            ps_seq = ps.DEER_Corr_Rabi(dark_taus*1e9, kwargs['tau']*1e9, kwargs['t_corr']*1e9, pi_half[0], pi_half[1], 
                                pi[0], pi[1]) # send to PS in [ns] units
            
            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 1)
            
            # configure signal generator for NV drive
            sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
            sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode

            # upload AWG sequence first
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
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # sort sweep parameters back to proper plotting order
                index_order = np.argsort(dark_taus) 
                dark_taus_sorted = np.sort(dark_taus) # order mw_times for Rabi plotting

                # open laser shutter
                laser_shutter.open_shutter()
                # daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))
                time.sleep(0.5)

                # upload digitizer parameters
                dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(ps_seq, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                dig.config()
                dig.start_buffer()
                
                # start pulse sequence
                ps.start_now()

                total_segments=kwargs['iters']*kwargs['runs']*kwargs['num_pts']*2

                # start experiment loop
                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        corr_result_raw = obtain(dig.acquire()) # acquire data from digitizer

                        # define dummy array to contain experiment data --> size (runs*num_pts) --> (segment_size)
                        corr_result = np.zeros(len(corr_result_raw)//dig_num_to_avg)
                        for i in range(0,len(corr_result_raw)//dig_num_to_avg):
                            corr_result[i] = np.mean(corr_result_raw[i*dig_num_to_avg:(i+1)*dig_num_to_avg])

                        # partition buffer into signal and background datasets
                        sig = self.analog_math(corr_result, 'Corr', kwargs['num_pts'])[0]
                        bg = self.analog_math(corr_result, 'Corr', kwargs['num_pts'])[1]

                        sig = np.abs(sig)
                        bg = np.abs(bg)

                        # correct the y-axis data ordering for plots
                        sig = np.array([sig[i] for i in index_order])
                        bg = np.array([bg[i] for i in index_order])

                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([dark_taus_sorted*1e9, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([dark_taus_sorted*1e9, bg]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        corr_rabi_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER Correlation Rabi',
                                        'xlabel': 'MW Pulse Duration (ns)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        # update GUI progress bar                        
                        pbar.update(segments/total_segments)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                # save data if requested upon completion of experiment
                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

            finally:
                self.equipment_off() # turn off equipment regardless of if experiment started or failed

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
        with InstrumentManager() as mgr, DataSource('corr t1') as corr_t1_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig
            
            # define parameter array that will be swept over in experiment & shuffle
            match kwargs['array_type']:
                case 'geomspace':
                    t_corr_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])     
                case 'linspace':
                    t_corr_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])     

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
            ps_seq = ps.DEER_Corr_T1(t_corr_times*1e9, kwargs['tau']*1e9, pi_half[0], pi_half[1], 
                                pi[0], pi[1], dark_pi*1e9) # send to PS in [ns] units
            
            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 1)
            
            # configure signal generator for NV drive
            sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
            sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode

            # upload AWG sequence first
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
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                with_pulse_py_sweeps = StreamingList()
                without_pulse_py_sweeps = StreamingList()
                with_pulse_ny_sweeps = StreamingList()
                without_pulse_ny_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()
                # daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))
                time.sleep(0.5)

                # upload digitizer parameters
                dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(ps_seq, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                dig.config()
                dig.start_buffer()
                
                # start pulse sequence
                ps.start_now()

                total_segments=kwargs['iters']*kwargs['runs']*kwargs['num_pts']*2

                # start experiment loop
                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        corr_result_raw = obtain(dig.acquire()) # acquire data from digitizer

                        # define dummy array to contain experiment data --> size (runs*num_pts) --> (segment_size)
                        corr_result = np.zeros(len(corr_result_raw)//dig_num_to_avg)
                        for i in range(0,len(corr_result_raw)//dig_num_to_avg):
                            corr_result[i] = np.mean(corr_result_raw[i*dig_num_to_avg:(i+1)*dig_num_to_avg])

                        # partition buffer into signal and background datasets
                        with_py = self.analog_math(corr_result, 'DEER', kwargs['num_pts'])[0]
                        without_py = self.analog_math(corr_result, 'DEER', kwargs['num_pts'])[1]
                        with_ny = self.analog_math(corr_result, 'DEER', kwargs['num_pts'])[2]
                        without_ny = self.analog_math(corr_result, 'DEER', kwargs['num_pts'])[3]

                        with_py = np.abs(with_py)
                        without_py = np.abs(without_py)
                        with_ny = np.abs(with_ny)
                        without_ny = np.abs(without_ny)

                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        with_pulse_py_sweeps.append(np.stack([t_corr_times*1e6, with_py]))
                        with_pulse_py_sweeps.updated_item(-1) 
                        without_pulse_py_sweeps.append(np.stack([t_corr_times*1e6, without_py]))
                        without_pulse_py_sweeps.updated_item(-1)
                        with_pulse_ny_sweeps.append(np.stack([t_corr_times*1e6, with_ny]))
                        with_pulse_ny_sweeps.updated_item(-1) 
                        without_pulse_ny_sweeps.append(np.stack([t_corr_times*1e6, without_ny]))
                        without_pulse_ny_sweeps.updated_item(-1)

                        # save the current data to the data server
                        corr_t1_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER Correlation T1',
                                        'xlabel': 'Free Precession Interval (\u03BCs) or Frequency (MHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'with_py': with_pulse_py_sweeps,
                                                    'without_py': without_pulse_py_sweeps,
                                                    'with_ny': with_pulse_ny_sweeps,
                                                    'without_ny': without_pulse_ny_sweeps}
                        })

                        # update GUI progress bar                        
                        pbar.update(segments/total_segments)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                # save data if requested upon completion of experiment
                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

            finally:
                self.equipment_off() # turn off equipment regardless of if experiment started or failed

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
        with InstrumentManager() as mgr, DataSource('nmr') as nmr_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig
            
            # define parameter array that will be swept over in experiment & shuffle
            t_corr_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for x pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
            ps_seq = ps.Corr_Spectroscopy(t_corr_times, kwargs['tau']*1e9, 
                                     pi_half[0], pi_half[1], 
                                     pi[0], pi[1], kwargs['n'])
            
            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 1)
            
            # configure signal generator for NV drive
            sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
            sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode

            # upload AWG sequence first
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
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()
                # daq.open_ai_task(kwargs['detector'], len(rabi_buffer[0]))
                time.sleep(0.5)

                # upload digitizer parameters
                dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(ps_seq, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                dig.config()
                dig.start_buffer()
                
                # start pulse sequence
                ps.start_now()

                total_segments=kwargs['iters']*kwargs['runs']*kwargs['num_pts']*2

                # start experiment loop
                with tqdm(total = kwargs['iters']) as pbar:

                    for i in range(kwargs['iters']):
                        
                        nmr_result_raw = obtain(dig.acquire()) # acquire data from digitizer

                        # define dummy array to contain experiment data --> size (runs*num_pts) --> (segment_size)
                        nmr_result = np.zeros(len(nmr_result_raw)//dig_num_to_avg)
                        for i in range(0,len(nmr_result_raw)//dig_num_to_avg):
                            nmr_result[i] = np.mean(nmr_result_raw[i*dig_num_to_avg:(i+1)*dig_num_to_avg])

                        # partition buffer into signal and background datasets
                        sig = self.analog_math(nmr_result, 'NMR', kwargs['num_pts'])[0]
                        bg = self.analog_math(nmr_result, 'NMR', kwargs['num_pts'])[1]

                        sig = np.abs(sig)
                        bg = np.abs(bg)

                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([t_corr_times/1e3, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([t_corr_times/1e3, bg]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        nmr_data.push({'params': {'kwargs': kwargs},
                                        'title': 'NMR Time Domain Data',
                                        'xlabel': 'Free Precession Interval (\u03BCs) or Frequency (MHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        # update GUI progress bar                        
                        pbar.update(segments/total_segments)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                # save data if requested upon completion of experiment
                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

            finally:
                self.equipment_off() # turn off equipment regardless of if experiment started or failed

    # TODO: update the experimental logic for this sequence
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
        with InstrumentManager() as mgr, DataSource('nmr') as casr_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            daq = mgr.daq
            hdawg = mgr.awg
            dig = mgr.dig

            t_corr_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # Generate Rabi data buffer
            casr_buffer = self.generate_buffer('NMR', kwargs['runs'], kwargs['num_pts'])

            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], segment_size = kwargs['segment_size'], 
                                                  pretrig_size = kwargs['pretrig_size'], read_channel = kwargs['read_channel'], runs = kwargs['runs'], coupling = 1, termination = 1)
            
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

                        # read the analog voltage levels received by the APD.
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        
                        signal_sweeps.append(np.stack([t_corr_times/1e3, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([t_corr_times/1e3, bg]))
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
                        
                        pbar.update(segments/total_segments)
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        time_per_iter = (pbar.last_print_t - pbar.start_t)/pbar.n
                        self.queue_from_exp.put_nowait([percent_completed, time_per_iter])

                if kwargs['save'] == True:
                    flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                # self.equipment_off()