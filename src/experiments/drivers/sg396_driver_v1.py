'''
Stanford RS SG396 Driver for Updated Nspyre

Based on: Implementation of SG396 signal generator
          Author: Kevin Miao & Berk Diler
          Date: 12/15/2015 & 8/21/2017

Author: Evan Villafranca
Date: 8/25/2022
'''

import logging
logger = logging.getLogger(__name__)

from collections import OrderedDict

from pyvisa import ResourceManager

logger = logging.getLogger(__name__)

from math import log10
def W_to_dBm(power):
    return 10*log10(1000*power)

class SG396:
    DEFAULTS = {
        'COMMON': {
            'write_termination': '\r\n',
            'read_termination': '\r\n',
        }
    }
    MODULATION_TYPE = OrderedDict([
        ('AM', 0),
        ('FM', 1),
        ('Phase', 2),
        ('Sweep', 3),
        ('Pulse', 4),
        ('Blank', 5),
        ('QAM', 7),
        ('CPM', 8),
        ('VSB', 9)
    ])
    MODULATION_FUNCTION = OrderedDict([
        ('sine', 0),
        ('ramp', 1),
        ('triangle', 2),
        ('square', 3),
        ('noise', 4),
        ('external', 5)
    ])

    def __init__(self, address):

        '''
        add pyvisa open function to connect to IP address
        '''

        self.rm = ResourceManager('@py')
        self.address = address
        self.device = self.rm.open_resource(self.address)
        print(f"connected to SG396 [{self.address}]")
        logger.info(f"Connected to SG396 [{self.address}].")

        self.output_en = False
        self._amplitude = 0.0
        self._frequency = 100e3


    def get_lf_amplitude(self):
        """
        low frequency amplitude (BNC output)
        """
        return float(self.device.query('AMPL?'))

    def set_lf_amplitude(self, value):
        self.device.write('AMPL{:.2f}'.format(W_to_dBm(value)))


    def get_rf_amplitude(self):
        """
        RF amplitude (Type N output)
        """
        return float(self.device.query('AMPR?'))

    def set_rf_amplitude(self, value):
       self.device.write(f"AMPR{W_to_dBm(value)}")    


    def get_lf_toggle(self):
        """
        low frequency output state
        """
        return self.device.query('ENBL?')
    
    def set_lf_toggle(self, value):
        self.device.write(f"ENBL{value}")


    def get_rf_toggle(self):
        """
        RF output state
        """
        return self.device.query('ENBR?')

    # 1 = True (on), 0 = False (off)
    def set_rf_toggle(self, value):
        self.device.write(f"ENBR{value}")

    def get_lf_offset(self):
        """
        low frequency offset voltage
        """
        return self.device.query('OFSL?')

    def set_lf_offset(self, value):
        self.device.write(f"OFSL{value}")

    def get_phase(self):
        """
        carrier phase
        """
        return self.device.query('PHAS?')

    def set_phase(self, value):
        self.device.write(f"PHAS{value}")
        

    def set_rel_phase(self):
        """
        sets carrier phase to 0 degrees
        """
        self.device.write('RPHS')

    def get_mod_toggle(self):
        """
        Modulation State
        """
        return int(self.device.query('MODL?'))

    def set_mod_toggle(self, value):
        self.device.write(f"MODL {value}")

    def get_mod_type(self):
        """
        Modulation State
        """
        return int(self.device.query('TYPE?'))

    def set_mod_type(self, value):
        self.device.write(f"TYPE {value}")

    def set_mod_subtype(self, value):
        self.device.write(f"STYP {value}")

    def get_mod_function(self):
        """
        Modulation Function
        """
        return int(self.device.query('MFNC?'))

    def set_mod_function(self, exp, value):
        match exp:
            case 'FM':
                self.device.write(f"MFNC {value}")
            case 'IQ':
                self.device.write(f"QFNC {value}")

    # units = "Hz", limits = (0.1, 100.e3))
    def get_mod_rate(self):
        """
        Modulation Rate
        """
        return float(self.device.query('RATE?'))
    
    def set_mod_rate(self, value):
        self.device.write(f"RATE {value}")

    def get_FM_mod_dev(self):
        """
        FM Modulation Deviation
        """
        return float(self.device.query("FDEV?"))
    
    def set_FM_mod_dev(self, value):
        self.device.write(f"FDEV {value}")

    # def get_mod_sweep_function(self):
    #     """
    #     Modulation Function
    #     """
    #     return int(self.device.query('SFNC?'))

    # def set_mod_sweep_function(self, value):
    #     self.device.write(f"SFNC {value}")

    # def get_mod_sweep_rate(self):
    #     return float(self.device.query('SRAT?'))
    
    # def set_mod_sweep_rate(self, value):
    #     self.device.write(f"SRAT {value}")

    # def get_mod_sweep_dev(self):
    #     return float(self.device.query('SDEV?'))
    
    # def set_mod_sweep_dev(self, value):
    #     self.device.write(f"SDEV {value}")
 
    # units = "pc", limits = (0., 100.))  
    # created percentage unit 'pc' to bypass problems with the instrument manager and pint
    def get_AM_mod_depth(self):
        """
        AM Modulation Depth
        """
        return float(self.device.query('ADEP?'))

    def set_AM_mod_depth(self, value):
        self.device.write(f"ADEP {value}")
    
    # units = "Hz", limits = (0.1, 8.e6))
    
    def get_frequency(self):
        """
        signal frequency
        """
        return self.device.query('FREQ?')
    
    def set_frequency(self, value):
        """Change the frequency (Hz)"""
        if value < 100e3 or value > 6e9:
            raise ValueError("Frequency must be in range [100 kHz, 6 GHz].")
        
        self.device.write(f"FREQ{value}")
        
        logger.info(f"Set frequency to {value} Hz")

    def amplitude(self):
        return self._amplitude

    def set_amplitude(self, value):
        """Change the amplitude (dBm)"""
        if value < -30 or value > 10:
            raise ValueError("Amplitude must be in range [-30 dBm, 10 dBm].")

        logger.info(f"Set amplitude to {value} dBm")

    def calibrate(self):
        logger.info("SG396 calibration succeeded.")
