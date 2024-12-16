"""
Zurich Instruments HDAWG driver for integration into Nspyre

Author: Evan Villafranca 
Created: 2023-3-29
"""

# import zhinst.toolkit as tk
import zhinst.utils
import zhinst.core

import os
import time
import textwrap
import numpy as np
import time
from typing import List

class HDAWG:

    def __init__(self, device_id, server_host, server_port):

        # self.awg = tk.HDAWG('name', 'dev8181', interface = 'USB')
        api_level = 6
        # self.awg.setup()
        # self.awg.connect_device()
        (self.daq, self.device, self.props) = zhinst.utils.create_api_session(device_id, api_level, server_host, server_port)

        zhinst.utils.api_server_version_check(self.daq)

        self.set_channels()
        self.sampling_rate = 2.4e9 # Hz
        self.voltage_range = 0.8
        self.control_both_groups = -1

        zhinst.utils.disable_everything(self.daq, self.device)

        exp_setting = [
            [f"/{self.device}/awgs/0/time", 0],
            [f"/{self.device}/awgs/1/time", 0],

            [f"/{self.device}/awgs/0/auxtriggers/0/channel", 0],
            [f"/{self.device}/awgs/0/auxtriggers/1/channel", 0],
            [f"/{self.device}/awgs/0/auxtriggers/0/slope", 1],
            [f"/{self.device}/awgs/0/auxtriggers/1/slope", 1],           
            
            [f"/{self.device}/awgs/1/auxtriggers/0/channel", 2],
            [f"/{self.device}/awgs/1/auxtriggers/1/channel", 2],
            [f"/{self.device}/awgs/1/auxtriggers/0/slope", 1],
            [f"/{self.device}/awgs/1/auxtriggers/1/slope", 1],

            [f"/{self.device}/awgs/0/outputs/0/modulation/mode", 1], # set for SRS IQ modulation
            [f"/{self.device}/awgs/0/outputs/1/modulation/mode", 2], # set for SRS IQ modulation
            [f"/{self.device}/awgs/1/outputs/0/modulation/mode", 1],
            [f"/{self.device}/awgs/1/outputs/1/modulation/mode", 2],

            # for SRS IQ modulation
            [f"/{self.device}/sigouts/0/on", 1],
            [f"/{self.device}/sigouts/1/on", 1],
            [f"/{self.device}/sigouts/0/direct", 1],
            [f"/{self.device}/sigouts/1/direct", 1],

            # for electron/nuclear spin driving
            [f"/{self.device}/sigouts/2/on", 1],

            [f"/{self.device}/dios/0/output", 0],
            [f"/{self.device}/dios/0/drive", 1],

            [f"/{self.device}/sigouts/0/delay", 0],
            [f"/{self.device}/sigouts/1/delay", 0],

            [f"/{self.device}/sigouts/0/range", self.voltage_range],
            [f"/{self.device}/sigouts/1/range", self.voltage_range],
            [f"/{self.device}/sigouts/2/range", self.voltage_range],
            # [f"/{self.device}/awgs/0/outputs/{self.awg_channel}/amplitude", self.amplitude]

            # ['/%s/awgs/0/outputs/%d/amplitude' % (self.device, self.awg_channel), self.amplitude],
            # ['/%s/awgs/0/outputs/0/modulation/mode' % self.device, 0],
            # ['/%s/awgs/0/time'                 % self.device, 0],
            # ['/%s/awgs/0/userregs/0'           % self.device, 0]
        ]
        
        self.daq.set(exp_setting)

        self.awgModule = self.daq.awgModule()
        # Ensure that all settings have taken effect on the device before continuing.

        # self.daq.setInt(f"/{self.device}/system/awg/oscillatorcontrol", 1)

        # self.daq.sync()

        # program = textwrap.dedent(f"""wave waveRF = rect(120,0.2);
        # repeat (3){{
        # repeat (2){{
        # setDouble('oscs/1/freq',350000000);repeat(2){{repeat(1){{waitDigTrigger(1);playWave(1,waveRF ); waitWave();   }}}}
        # setDouble('oscs/1/freq',394444444.4444);repeat(2){{repeat(1){{waitDigTrigger(1);playWave(1,waveRF ); waitWave();   }}}}
        # setDouble('oscs/1/freq',438888888.8889);repeat(2){{repeat(1){{waitDigTrigger(1);playWave(1,waveRF ); waitWave();   }}}}
        # setDouble('oscs/1/freq',483333333.3333);repeat(2){{repeat(1){{waitDigTrigger(1);playWave(1,waveRF ); waitWave();   }}}}
        # setDouble('oscs/1/freq',527777777.7778);repeat(2){{repeat(1){{waitDigTrigger(1);playWave(1,waveRF ); waitWave();   }}}}
        # setDouble('oscs/1/freq',572222222.2222);repeat(2){{repeat(1){{waitDigTrigger(1);playWave(1,waveRF ); waitWave();   }}}}
        # setDouble('oscs/1/freq',616666666.6667);repeat(2){{repeat(1){{waitDigTrigger(1);playWave(1,waveRF ); waitWave();   }}}}
        # setDouble('oscs/1/freq',661111111.1111);repeat(2){{repeat(1){{waitDigTrigger(1);playWave(1,waveRF ); waitWave();   }}}}
        # setDouble('oscs/1/freq',705555555.5556);repeat(2){{repeat(1){{waitDigTrigger(1);playWave(1,waveRF ); waitWave();   }}}}
        # setDouble('oscs/1/freq',750000000);repeat(2){{repeat(1){{waitDigTrigger(1);playWave(1,waveRF ); waitWave();   }}}}
        # }}
        # }}
        # """)

        # # Upload settings to the device in a transaction
        # self.daq.set(exp_setting)

        # # Create an instance of the AWG Module
        # awgModule = self.daq.awgModule()
        # awgModule.set("device", self.device)
        # awgModule.set("index", 1)
        # awgModule.execute()

        # # Transfer the AWG sequencer program. Compilation starts automatically.
        # awgModule.set("compiler/sourcestring", program)
        
        # # Wait until compilation is done
        # timeout = 10  # seconds
        # start = time.time()
        # compiler_status = awgModule.getInt("compiler/status")
        # while compiler_status == -1:
        #     if time.time() - start >= timeout:
        #         raise TimeoutError("Program compilation timed out")
        #     time.sleep(0.01)
        #     compiler_status = awgModule.getInt("compiler/status")

        # compiler_status_string = awgModule.getString("compiler/statusstring")
        # if compiler_status == 0:
        #     print(
        #         "Compilation successful with no warnings, will upload the program to the instrument."
        #     )
        # if compiler_status == 1:
        #     raise RuntimeError(
        #         f"Error during sequencer compilation: {compiler_status_string:s}"
        #     )
        # if compiler_status == 2:
        #     print(f"Warning during sequencer compilation:  {compiler_status_string:s}")

        # # Wait until the sequence is correctly uploaded
        # start = time.time()
        # # Check the ready status for each core
        # while self.daq.getInt(f"/{self.device:s}/awgs/1/ready") == 0:
        #     # Timeout if all the cores doesn't report ready in time
        #     if time.time() - start >= timeout:
        #         raise TimeoutError(f"Sequence not uploaded within {timeout:.1f}s.")
        #     time.sleep(0.01)

        # print("Sequence successfully uploaded.")

        # self.daq.setInt(f"/{self.device}/awgs/1/single", 1)
        # self.daq.setInt(f"/{self.device}/awgs/1/enable", 1)

        # # self.daq.setInt(f"/{self.device:s}/awgs/1/single", True)
        # # self.daq.syncSetInt(f"/{self.device:s}/awgs/1/enable", True)


        # # AWG PARAMETERS FROM EXPERIMENT FILE 
        # self.num_pts = 0
        # self.samples_step_size = 0
        # self.ch1_amp = 1

    def set_channels(self):
        ### CHANNEL SELECTION ###
        # 'system/awg/channelgrouping' : Configure how many independent sequencers
        #   should run on the AWG and how the outputs are grouped by sequencer.
        #   0 : 4x2 with HDAWG8; 2x2 with HDAWG4.
        #   1 : 2x4 with HDAWG8; 1x4 with HDAWG4.
        #   2 : 1x8 with HDAWG8. 
        # Configure the HDAWG to use one sequencer with the same waveform on all output channels.
        self.daq.setInt(f"/{self.device}/system/awg/channelgrouping", 0)
        self.daq.setInt(f"/{self.device}/triggers/in/0/imp50", 1)
        self.daq.setInt(f"/{self.device}/triggers/in/1/imp50", 1)
        self.daq.setInt(f"/{self.device}/triggers/in/2/imp50", 1)
        self.daq.setInt(f"/{self.device}/triggers/in/3/imp50", 1)

    def set_awg_oscillator_control(self, state):
        match state:
            case 'on':
                self.daq.setInt(f"/{self.device}/system/awg/oscillatorcontrol", 1)
            case _:
                self.daq.setInt(f"/{self.device}/system/awg/oscillatorcontrol", 0)
    
    def set_sampling_rate(self, group, idx):
        try:
            self.daq.setInt(f"/{self.device}/awgs/{group}/time", {idx}),
        except Exception as e:
            print(e)

    # def set_enabled(self, group):
    #     self.daq.setInt(f"/{self.device}/awgs/{group}/single", 1)
    #     self.daq.setInt(f"/{self.device}/awgs/{group}/enable", 1)        
    #     # self.daq.setInt(f"/{self.device}/awgs/1/single", 1)
    #     # self.daq.setInt(f"/{self.device}/awgs/1/enable", 1)
    #     self.awgModule.set('awg/enable', 1)
        
    def set_group_enabled(self, group):
        self.daq.setInt(f"/{self.device}/awgs/{group}/single", 1)
        self.daq.setInt(f"/{self.device}/awgs/{group}/enable", 1)        
        # self.daq.setInt(f"/{self.device}/awgs/1/single", 1)
        # self.daq.setInt(f"/{self.device}/awgs/1/enable", 1)
    
    def set_awg_enabled(self):    
        self.awgModule.set('awg/enable', 1)

    def set_disabled(self):
        self.daq.setInt(f"/{self.device}/awgs/0/single", 0)
        self.daq.setInt(f"/{self.device}/awgs/0/enable", 0)
        self.daq.setInt(f"/{self.device}/awgs/1/single", 0)
        self.daq.setInt(f"/{self.device}/awgs/1/enable", 0)
        self.awgModule.set('awg/enable', 0)

    def set_voltage_offsets(self, i_offset, q_offset):
        self.daq.setDouble(f"/{self.device}/sigouts/1/offset", i_offset) # units of [V]
        self.daq.setDouble(f"/{self.device}/sigouts/0/offset", q_offset)

    def set_sequence(self, **kwargs):

        self.set_voltage_offsets(kwargs['i_offset'], kwargs['q_offset']) # set I and Q offset voltages to suppress carrier freq

        self.set_disabled()

        time.sleep(0.1)

        match kwargs['seq']:
            case 'Test':
                # wave I_pihalf_x = rect(10000.0, 0.5625);
                # wave Q_pihalf_x = rect(10000.0, 0.5625);

                # setSinePhase(0, 0); setSinePhase(1, 90);

                # while(1){
                # playWave(1, I_pihalf_x, 2, Q_pihalf_x); waitWave();
                
                # }
                match kwargs['channel']:
                    case 'IQ':
                        i_wave = [self.create_rect_wave("I", 10000, self.convert_mw_power(kwargs['sideband_power']))]
                        q_wave = [self.create_rect_wave("Q", 10000, self.convert_mw_power(kwargs['sideband_power']))]

                        i_phase = [self.set_sine_phase(0, kwargs['iq_phases'][0])]
                        q_phase = [self.set_sine_phase(1, kwargs['iq_phases'][1])]

                        awg_pulses: List[str] = [] # lines telling AWG to emit pulse
                        for i in range(1):
                            awg_pulses.append(f"""while(1){{{self.create_iq_pulses_no_trig(1, "I", 2, "Q")} waitWave();}}""")

                        self.awg_iq_program_text = "\n".join(i_wave + q_wave + i_phase + q_phase + awg_pulses)

                        self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                        self.control_both_groups = 0

                    case 'DEER':
                        print("running deer output")
                        deer_wave = [self.create_rect_wave("waveRF", 10000, self.convert_mw_power(kwargs['deer_power']))]

                        awg_pulses: List[str] = [] # lines telling AWG to emit pulse
                        for i in range(1):
                            awg_pulses.append(f"""while(1){{{self.create_pulses_no_trig(1, "waveRF")} waitWave();}}""")

                        self.awg_program_text = "\n".join(deer_wave + awg_pulses)

                        self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['RF_Frequency'])

                        self.control_both_groups = 1

                self.set_awg_oscillator_control('off')

            case 'Calibrate':
                i_wave = [self.create_rect_wave("I", kwargs['pi']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave = [self.create_rect_wave("Q", kwargs['pi']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]

                i_phase = [self.set_sine_phase(0, kwargs['iq_phases'][0])]
                q_phase = [self.set_sine_phase(1, kwargs['iq_phases'][1])]

                awg_pulses: List[str] = [] # lines telling AWG to emit pulse
                for i in range(kwargs['num_pts']):
                    awg_pulses.append(f"""repeat(1){{{self.create_iq_pulses(1, 1, "I", 2, "Q")} waitWave();}}""")
                
                iter_repeat = [self.repeat(kwargs['iters'])] 
                runs_repeat = [self.repeat(kwargs['runs'])]

                iters_end = [f"}}}}"]

                self.awg_iq_program_text = "\n".join(i_wave + q_wave + i_phase + q_phase + iter_repeat + runs_repeat + awg_pulses + iters_end)

                self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 0

            case 'CW ODMR':           
                odmr_side_freqs = kwargs['sideband_freqs'] # define frequency array
                
                # define the sideband frequencies for the ODMR experiment over range ("start", "stop", step defined by num_pts)
                i_wave = [self.create_rect_wave("I", kwargs['probe_length']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave = [self.create_rect_wave("Q", kwargs['probe_length']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                
                odmr_pulses: List[str] = []
                
                # loop through each sideband frequency, set lower frequency and output first then set upper frequency and output
                for i in range(kwargs['num_pts']):
                    odmr_pulses.append(f"""setDouble('oscs/0/freq', {odmr_side_freqs[i]}); 
{self.set_sine_phase(0, kwargs['iq_phases'][0])}
{self.set_sine_phase(1, kwargs['iq_phases'][1])}
{self.create_iq_pulses(1, 1, "I", 2, "Q")} waitWave();""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave + q_wave + inf_repeat + odmr_pulses + inf_end)

                self.set_awg_oscillator_control('on')

                self.control_both_groups = 0

            case 'CW ODMR central freq':           
                odmr_side_freqs = kwargs['sideband_freqs'] # define frequency array
                
                # define the sideband frequencies for the ODMR experiment over range ("start", "stop", step defined by num_pts)
                i_wave = [self.create_rect_wave("I", kwargs['probe_length']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave = [self.create_rect_wave("Q", kwargs['probe_length']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                
                odmr_pulses: List[str] = []
                
                # loop through each sideband frequency, set lower frequency and output first then set upper frequency and output
                for i in range(kwargs['num_pts']//2):
                    odmr_pulses.append(f"""setDouble('oscs/0/freq', {odmr_side_freqs[i]}); 
{self.set_sine_phase(0, kwargs['iq_phases'][0])}
{self.set_sine_phase(1, kwargs['iq_phases'][1])}
{self.create_iq_pulses(1, 1, "I", 2, "Q")} waitWave();
{self.set_sine_phase(0, kwargs['iq_phases'][2])}
{self.set_sine_phase(1, kwargs['iq_phases'][3])}
{self.create_iq_pulses(1, 1, "I", 2, "Q")} waitWave();""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave + q_wave + inf_repeat + odmr_pulses + inf_end)

                self.set_awg_oscillator_control('on')

                self.control_both_groups = 0

            case 'Rabi':           
                rabi_pulses = kwargs['pi_pulses']

                i_phase = [self.set_sine_phase(0, kwargs['iq_phases'][0])]
                q_phase = [self.set_sine_phase(1, kwargs['iq_phases'][1])]

                i_waves: List[str] = [] # defines wave structures
                q_waves: List[str] = []
                awg_pulses: List[str] = [] # lines telling AWG to emit pulse

                for i in range(kwargs['num_pts']):
                    # awg_waves.append(f"wave {i} = rect({rabi_pulses[i]}, {kwargs['awg_power']});")
                    i_waves.append(self.create_rect_wave(f"I{i}", rabi_pulses[i]*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power'])))
                    q_waves.append(self.create_rect_wave(f"Q{i}", rabi_pulses[i]*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power'])))
                    
                    awg_pulses.append(f"""{self.create_iq_pulses(1, 1, f"I{i}", 2, f"Q{i}")} waitWave();""")

                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_phase + q_phase + i_waves + q_waves + inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 0
            
            case 'Pulsed ODMR':           
                odmr_side_freqs = kwargs['sideband_freqs'] # define frequency array
                
                # define the sideband frequencies for the ODMR experiment over range ("start", "stop", step defined by num_pts)
                i_wave = [self.create_rect_wave("I_pi", kwargs['pi_pulse']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave = [self.create_rect_wave("Q_pi", kwargs['pi_pulse']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                
                odmr_pulses: List[str] = []
                
                # loop through each sideband frequency, set lower frequency and output first then set upper frequency and output
                for i in range(kwargs['num_pts']):
                    odmr_pulses.append(f"""setDouble('oscs/0/freq', {odmr_side_freqs[i]}); 
{self.set_sine_phase(0, kwargs['iq_phases'][0])}
{self.set_sine_phase(1, kwargs['iq_phases'][1])}
{self.create_iq_pulses(1, 1, "I_pi", 2, "Q_pi")} waitWave();""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave + q_wave + inf_repeat + odmr_pulses + inf_end)

                self.set_awg_oscillator_control('on')

                self.control_both_groups = 0

            case 'T1':
                i_wave = [self.create_rect_wave("I_pi", kwargs['pi_pulse']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave = [self.create_rect_wave("Q_pi", kwargs['pi_pulse']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]

                i_phase = [self.set_sine_phase(0, kwargs['iq_phases'][0])]
                q_phase = [self.set_sine_phase(1, kwargs['iq_phases'][1])]

                awg_pulses: List[str] = [] # lines telling AWG to emit pulse
                # for i in range(kwargs['num_pts']):
                awg_pulses.append(f"""repeat({kwargs['num_pts']}){{{self.create_iq_pulses(1, 1, "I_pi", 2, "Q_pi")} waitWave();}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave + q_wave + i_phase + q_phase + inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 0

            case 'DQ':
                i_wave_minus1 = [self.create_rect_wave("I_minus1", kwargs['pi_minus1']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_minus1 = [self.create_rect_wave("Q_minus1", kwargs['pi_minus1']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_plus1 = [self.create_rect_wave("I_plus1", kwargs['pi_plus1']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_plus1 = [self.create_rect_wave("Q_plus1", kwargs['pi_plus1']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]

                awg_pulses: List[str] = [] # lines telling AWG to emit pulse
                
                awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
{self.set_sine_phase(0, kwargs['iq_phases'][0])}
{self.set_sine_phase(1, kwargs['iq_phases'][1])}
repeat(4){{{self.create_iq_pulses(1, 1, "I_minus1", 2, "Q_minus1")} waitWave();}}
{self.set_sine_phase(0, kwargs['iq_phases'][2])}
{self.set_sine_phase(1, kwargs['iq_phases'][3])}
repeat(1){{{self.create_iq_pulses(1, 1, "I_plus1", 2, "Q_plus1")} waitWave();}}}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_minus1 + q_wave_minus1 + i_wave_plus1 + q_wave_plus1 + inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 0
                
            case 'T2':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]

                awg_pulses: List[str] = [] # lines telling AWG to emit pulse
                
                match kwargs['seq_dd']:
                    case 'Ramsey':
                        awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}""")
                            
                    case 'Echo':
                        awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();  
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}""")
                            
                    case 'XY4':
                        awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();        
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();}}
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();}}
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}""")
                            
                    case 'XY8':
                        awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();              
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();}}
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();}}
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}""")
                    
                    case 'YY8':
                        awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();}}
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();}}
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}""")
                            
                    case 'CPMG':
                        awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();                     
repeat({kwargs['n']}){{
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();}}
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();}}
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}""")

                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 0

            case 'DEER':  
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]

                awg_iq_pulses: List[str] = []

                awg_iq_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(2){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();     
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                iters_end = [f"}}}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_iq_pulses + inf_end)
                

                deer_freqs = kwargs['freqs'] # define frequency array
                # print("DEER FREQUENCIES: ", deer_freqs)
                #define the frequency for the deer experiment over range ("start", "stop", step defined by num_pts)
                #awg_wave = [f"wave waveRF = rect({kwargs['dark_pi']*self.sampling_rate}, {kwargs['awg_power']});"] 
                awg_wave = [self.create_rect_wave("waveRF", kwargs['pi_pulse']*self.sampling_rate, self.convert_mw_power(kwargs['mw_power']))]
                awg_pulses: List[str] = []

                for i in range(kwargs['num_pts']):
                    awg_pulses.append(f"""setDouble('oscs/1/freq', {deer_freqs[i]}); 
repeat(4){{{self.create_pulses(1, 1, "waveRF")} waitWave();}}""")

                self.awg_program_text = "\n".join(awg_wave + inf_repeat + awg_pulses + inf_end)

                self.set_awg_oscillator_control('on')
                
                self.control_both_groups = 2

            case 'DEER CD':           
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]

                awg_iq_pulses: List[str] = []

                awg_iq_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(3){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();        
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_iq_pulses + inf_end)
                
                deer_freqs = kwargs['freqs'] # define frequency array
                # print("DEER FREQUENCIES: ", deer_freqs)
                #define the frequency for the deer experiment over range ("start", "stop", step defined by num_pts)
                #awg_wave = [f"wave waveRF = rect({kwargs['dark_pi']*self.sampling_rate}, {kwargs['awg_power']});"] 
                awg_wave = [self.create_rect_wave("waveRF", kwargs['pi_pulse']*self.sampling_rate, self.convert_mw_power(kwargs['mw_power']))]
                awg_pulses: List[str] = []

                for i in range(kwargs['num_pts']):
                    awg_pulses.append(f"""setDouble('oscs/1/freq', {deer_freqs[i]}); 
repeat(2){{{self.create_pulses(1, 1, "waveRF")} waitWave();}}""")

                self.awg_program_text = "\n".join(awg_wave + inf_repeat + awg_pulses + inf_end)

                self.set_awg_oscillator_control('on')

                self.control_both_groups = 2

            case 'DEER Rabi':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]

                awg_iq_pulses: List[str] = []

                awg_iq_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(2){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();   
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_iq_pulses + inf_end)
                
                rabi_pulses = kwargs['pi_pulses']
                
                awg_waves: List[str] = [] # defines wave structures
                awg_pulses: List[str] = [] # lines telling AWG to emit pulse

                for i in range(kwargs['num_pts']):
                    # awg_waves.append(f"wave {i} = rect({rabi_pulses[i]}, {kwargs['awg_power']});")
                    awg_waves.append(self.create_rect_wave(f"w{i}", rabi_pulses[i]*self.sampling_rate, self.convert_mw_power(kwargs['mw_power'])))
                    # awg_pulses.append(f"waitDigTrigger(1); playWave(1, {i});")
                    
                    awg_pulses.append(f"""repeat(4){{
{self.create_pulses(1, 1, f"w{i}")}}}""")

                self.awg_program_text = "\n".join(awg_waves + inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['dark_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 2

            case 'DEER Corr Rabi':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]

                awg_iq_pulses: List[str] = []

                awg_iq_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(3){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();           
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();}}

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_iq_pulses + inf_end)
                
                rabi_pulses = kwargs['pi_pulses']
                
                awg_wave = [self.create_rect_wave("waveRF", kwargs['dark_pulse']*self.sampling_rate, self.convert_mw_power(kwargs['mw_power']))]

                awg_waves: List[str] = [] # defines wave structures
                awg_pulses: List[str] = [] # lines telling AWG to emit pulse

                for i in range(kwargs['num_pts']):
                    # awg_waves.append(f"wave {i} = rect({rabi_pulses[i]}, {kwargs['awg_power']});")
                    awg_waves.append(self.create_rect_wave(f"w{i}", rabi_pulses[i]*self.sampling_rate, self.convert_mw_power(kwargs['mw_power'])))
                    # awg_pulses.append(f"waitDigTrigger(1); playWave(1, {i});")
                    awg_pulses.append(f"""repeat(2){{{self.create_pulses(1, 1, "waveRF")} {self.create_pulses(1, 1, "waveRF")}
{self.create_pulses(1, 1, f"w{i}")} {self.create_pulses(1, 1, "waveRF")} {self.create_pulses(1, 1, "waveRF")}}}""")

                iters_end = [f"}}}}"]
                self.awg_program_text = "\n".join(awg_wave + awg_waves + inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['dark_freq'])

                self.set_awg_oscillator_control('off')
                
                self.control_both_groups = 2

            case 'DEER FID':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]

                awg_iq_pulses: List[str] = []

                awg_iq_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(2){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();

{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_iq_pulses + inf_end)
                
                #awg_wave = [f"wave waveRF = rect({kwargs['dark_pi']*self.sampling_rate}, {kwargs['awg_power']});"] 
                awg_wave = [self.create_rect_wave("waveRF", kwargs['pi_pulse']*self.sampling_rate, self.convert_mw_power(kwargs['mw_power']))]
                awg_pulses = [f"""repeat({kwargs['num_pts']}){{
repeat({4*kwargs['n']}){{
{self.create_pulses(1, 1, "waveRF")}}}}}"""]
                
                self.awg_program_text = "\n".join(awg_wave + inf_repeat + awg_pulses + inf_end)
                
                self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['dark_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 2

            case 'DEER FID CD':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]

                awg_iq_pulses: List[str] = []

                awg_iq_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(3){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_iq_pulses + inf_end)
                
                taus = kwargs['taus']

                #awg_wave = [f"wave waveRF = rect({kwargs['dark_pi']*self.sampling_rate}, {kwargs['awg_power']});"] 
                awg_wave_DEER = [self.create_rect_wave("waveDEER", kwargs['pi_pulse']*self.sampling_rate, self.convert_mw_power(kwargs['mw_power']))]
                
                awg_CD_waves: List[str] = [] # defines wave structures
                awg_pulses: List[str] = [] # lines telling AWG to emit pulse

                for i in range(kwargs['num_pts']):
                    awg_CD_waves.append(self.create_rect_wave(f"waveCD{i}", taus[i]*self.sampling_rate, self.convert_mw_power(kwargs['cd_mw_power'])))

                    awg_pulses.append(f"""repeat({4*kwargs['n']}){{
{self.create_pulses(1, 1, "waveDEER")}}}
repeat({4*kwargs['n']}){{
{self.create_pulses(1, 1, f"waveCD{i}")}}}""")
                
                self.awg_program_text = "\n".join(awg_wave_DEER + awg_CD_waves + inf_repeat + awg_pulses + inf_end)
                
                self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['dark_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 2

            case 'DEER Corr T1':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]

                awg_iq_pulses: List[str] = []

                awg_iq_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(4){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();

{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();}}

repeat(4){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();

{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();}}}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_iq_pulses + inf_end)
                
                #awg_wave = [f"wave waveRF = rect({kwargs['dark_pi']*self.sampling_rate}, {kwargs['awg_power']});"] 
                awg_wave = [self.create_rect_wave("waveRF", kwargs['dark_pulse']*self.sampling_rate, self.convert_mw_power(kwargs['mw_power']))]
                awg_pulses = [f"""repeat({kwargs['num_pts']}){{
repeat(1){{
repeat(19){{
{self.create_pulses(1, 1, "waveRF")}}}}}}}"""]
                
                self.awg_program_text = "\n".join(awg_wave + inf_repeat + awg_pulses + inf_end)
                
                self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['dark_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 2

            case 'NMR':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate, self.convert_mw_power(kwargs['sideband_power']))]

                awg_pulses: List[str] = [] # lines telling AWG to emit pulse

                match kwargs['seq_nmr']:
                            
                    case 'Correlation Spectroscopy':
                        awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(2){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();}}
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();}}

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();}}
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();}}
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();}}""")

                    case 'CASR':
                        pass
                    
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 0

        try:
            if self.control_both_groups == 0:
                self.awg_iq_program = textwrap.dedent(f"{self.awg_iq_program_text}")
                self.compile_sequence(0, self.awg_iq_program)

            elif self.control_both_groups == 1:
                self.awg_program = textwrap.dedent(f"{self.awg_program_text}")
                self.compile_sequence(1, self.awg_program)

            elif self.control_both_groups == 2:
                self.awg_iq_program = textwrap.dedent(f"{self.awg_iq_program_text}")
                self.compile_sequence(0, self.awg_iq_program)
                time.sleep(0.5)
                self.awg_program = textwrap.dedent(f"{self.awg_program_text}")
                self.compile_sequence(1, self.awg_program)

        except ValueError:
            raise ValueError
        else:
            time.sleep(0.1)
            self.set_awg_enabled()

    def compile_sequence(self, group, program):
        self.awgModule = self.daq.awgModule()
        self.awgModule.set("device", self.device)
        self.awgModule.set("index", group) 
        self.awgModule.execute()
        # print(program)
        self.awgModule.set("compiler/sourcestring", program)
        
        print("AWG COMPILE STATUS: ", self.awgModule.getInt("compiler/status"))

        # Compile the AWG program
        # device_type = self.daq.getString(f"/{self.device}/features/devtype")
        # options = self.daq.getString(f"/{self.device}/features/options")
        # samplerate = self.daq.getDouble(f"/{self.device}/system/clocks/sampleclock/freq")

        # elf, compiler_info = zhinst.core.compile_seqc(
        #     program, devtype=device_type, options=options, index=0, samplerate=samplerate
        # )

        # print(compiler_info)
        # assert not compiler_info[
        #     "messages"
        # ], f"There was an error during compilation: {compiler_info['messages']:s}"


        # # self.daq.setVector(f"/{self.device}/awgs/1/elf/data", elf)

        # # Wait until the sequence is correctly uploaded
        # timeout = 10.0
        # start = time.time()
        # while self.daq.getInt(f"/{self.device}/awgs/1/ready") == 0:
        #     if time.time() - start > timeout:
        #         raise TimeoutError(
        #             f"Sequence not uploaded within {timeout:.1f}s."
        #         )
        #     time.sleep(0.01)
        # print("Sequence successfully uploaded.")

        # Wait until compilation is done
        timeout = 20  # seconds
        # time.sleep(20)
        start = time.time()
        # compiler_status = awgModule.compiler.status()
        while self.awgModule.getInt("compiler/status") == -1:
            if time.time() - start >= timeout:
                raise TimeoutError("Program compilation timed out")
            time.sleep(0.01)
            # compiler_status = awgModule.compiler.status()

        print("AWG COMPILE STATUS again: ", self.awgModule.getInt("compiler/status"))

        if self.awgModule.getInt("compiler/status") == 1:
            raise RuntimeError(
                f"Error during sequencer compilation: {self.awgModule.getString('compiler/statusstring')}"
            )
        else:
            # compiler_status_string = awgModule.compiler.statusstring()
            if self.awgModule.getInt("compiler/status") == 0:
                print(
                    "Compilation successful with no warnings, will upload the program to the instrument."
                )
            
            if self.awgModule.getInt("compiler/status") == 2:
                print(f"Warning during sequencer compilation:  {self.awgModule.getString('compiler/statusstring')}")

            # Wait until the sequence is correctly uploaded
            start = time.time()
            
            # Check the ready status for each core
            while self.daq.getInt(f"/{self.device:s}/awgs/{group}/ready") == 0:
                # Timeout if all the cores doesn't report ready in time
                if time.time() - start >= timeout:
                    raise TimeoutError(f"Sequence not uploaded within {timeout:.1f}s.")
                time.sleep(0.01)

            print("Sequence successfully uploaded.")

        time.sleep(0.1)
        self.set_group_enabled(group)
        time.sleep(0.1)
        self.daq.sync()
        # self.daq.syncSetInt(f"/{self.device:s}/awgs/1/enable", True)

    def convert_mw_power(self, power):
        I = 500/np.sqrt(2)
        if power > I:
            raise Exception("Power exceeds IQ port handling for SRS 396 signal generator.")
        else:
            return power/self.voltage_range

# generate separate one-task functions for each command: create rectangular waves; repeat iterations; repeat runs; create pulses
    def create_rect_wave(self, name, length, power):
        return f"wave {name} = rect({length}, {power});"

    # def create_wave(self, name, length, power):
    #     return f"wave {name} = rect({length}, {power});"
    
    def set_sine_phase(self, ch, phase):
        return f"setSinePhase({ch}, {phase});"
    
    def set_pulse_phases(self, axis, iq_phases):
        match axis:
            case 'x':
                phase_offset = 0
            case '-x':
                phase_offset = 180
            case 'y':
                phase_offset = 90
            case '-y':
                phase_offset = 270

        return f"setSinePhase(0, {phase_offset + iq_phases[0]}); setSinePhase(1, {phase_offset + iq_phases[1]});"

    def repeat_inf(self):
        return f"while(true){{"
    
    def repeat(self, var):
        return f"repeat({var}){{"

    def runs(self, runs):
        return f"repeat({runs}){{"
    
    def create_pulses(self, trigger_num, wave_num, wave_name):
        return f"waitDigTrigger({trigger_num}); playWave({wave_num}, {wave_name});"

    def create_pulses_no_trig(self, wave_num, wave_name):
        return f"playWave({wave_num}, {wave_name});"
    
    def create_iq_pulses(self, trigger_num, i_wave_num, i_wave_name, q_wave_num, q_wave_name):
        return f"waitDigTrigger({trigger_num}); playWave({i_wave_num}, {i_wave_name}, {q_wave_num}, {q_wave_name});"

    def create_iq_pulses_no_trig(self, i_wave_num, i_wave_name, q_wave_num, q_wave_name):
        return f"playWave({i_wave_num}, {i_wave_name}, {q_wave_num}, {q_wave_name});"
    
    def create_gauss_wave(self, name, amplitude, sample_num, center_position, standard_deviation):
        return f"wave {name} = {amplitude}*gauss({sample_num}, {center_position}, {standard_deviation});"
    
    



    def set_voltage_range(self):
        pass

    def set_sine_generators_state(self):
        # TODO:
        # control when to bypass internal sine generators
        pass

