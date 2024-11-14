"""
Python wrapper to control a single oxxius laser.
Reference: 
    [1] Oxxius LaserBoxx LBX user's guide (December 2014, ID: CO-1499-E)

Author: ChunTung Cheung 
Email: ctcheung1123@gmail.com
Created:  2023-01-04
Modified: 2023-01-07
"""

from oxxius.classeLaser import LasersList, Laser
import logging

logger = logging.getLogger(__name__)

ANALOG_CONTROL_MODE_TABLE = {"0":"power", 0:"power", "1":"current", 1:"current"}
STATUS_TABLE = {
    1:"Warm Up", 2:"Standby", 3:"Laser ON", 4:"Error", 5:"Alarm", 6:"Sleep", 7:"Searching SLM point",
    "1":"Warm Up", "2":"Standby", "3":"Laser ON", "4":"Error", "5":"Alarm", "6":"Sleep", "7":"Searching SLM point"
}
class LaserControl(Laser):
    def __init__(self, serial_num=""):
        lasers = LasersList()
        if serial_num == "":
            # connect to the first laser
            laser_infos = lasers.get_list()[0]
        else:
            # connect to the specific laser
            laser_infos = lasers.find_serial_number(serial_num)
        super().__init__(laser_infos)
        self.open()
        # self.close()

    def send_command(self, input):
        rep = super().send(input)
        if rep == input:
            return True
        else:
            logger.warning(f"Failed to set command '{input}'")
            logger.warning(f"Response from Device :'{rep}'")
            return False
    
    def send_query(self, input):
        return super().send(input).encode('ASCII')

    # wrappers of commands -----------------------------------------------------
    def set_analog_control_mode(self, mode):
        # set constant power or constant current mode
        mode = mode.lower()
        if mode == "power":
            logger.info("Analog control mode set to 'power'")
            self.send_command("ACC=0")
        elif mode == "current":
            logger.info("Analog control mode set to 'current'")
            self.send_command("ACC=1")
        else:
            logger.warning("Invalid argument. Please enter either 'power' or 'current'.")

    def set_analog_modulation(self, extint):
        extint = extint.lower()
        if extint == "ext":
            logger.info("Analog modulation set to 'external'")
            self.send_command("EXT=1")
        elif extint == "int":
            logger.info("Analog modulation set to 'internal'")
            self.send_command("EXT=0")
        else:
            logger.warning("Invalid argument. Please enter either 'ext' or 'int'.")

    def set_diode_current_memory(self, current_percent):
        # set the laser diode current to "current_percent" % (0 to 125%)
        # save the value to memory
        current_percent_int = round(current_percent, 2)
        self.send_command(f"C={current_percent_int}")
        logger.info(f"Laser diode current set to '{current_percent_int}%'")

    def set_CDRH_state(self, delay=False):
        if delay:
            logger.info(f"CDRH state set to 'delay ON'")
            self.send_command("CDRH=1")
        else:
            logger.info(f"CDRH state set to 'delay OFF'")
            self.send_command("CDRH=0")

    def set_diode_current_realtime(self, current_percent):
        # set the laser diode current to "current_percent" % (0 to 125%) without saving the value to memory 
        current_percent_int = round(current_percent, 2)
        logger.info(f"Laser diode current set to '{current_percent_int}%'")
        self.send_command(f"CM={current_percent_int}")

    def set_modulation_state(self, state):
        state = state.lower()
        if state == "cw":
            logger.info(f"Modulation state set to 'continuous'")
            self.send_command("CW=1")
        elif state == "pulsed" or "modulated":
            logger.info(f"Modulation state set to 'modulated'")
            self.send_command("CW=0")
        else:
            logger.warning(f"Invalid argument.  Please enter 'cw', 'pulsed' or 'modulated'. ")

    def set_laser_emission_activation(self, on):
        if on:
            logger.info(f"Laser emission is ON")
            self.send_command("L=1")
        else:
            logger.info(f"Laser emission is OFF")
            self.send_command("L=0")

    def set_laser_power_memory(self, power):
        # set the laser and save the value in memory 
        # power: specify the laser power [mW] in the range of 0 to ?MAXLP
        power_int = int(round(power))
        logger.info(f"Laser power (saved in memory) set to '{power_int}'")
        self.send_command(f"P={power_int}")

    def set_laser_power_realtime(self, power):
        # set the laser without saving the value in memory 
        # allowing realtime modification in a max. freq. of 50Hz (RS-232) or 1kHz (USB)
        # power: specify the laser power [mW] in the range of 0 to ?MAXLP
        power_int = int(round(power))
        logger.info(f"Laser power (real time) set to '{power_int}'")
        self.send_command(f"PM={power_int}")

    def reset_alarm(self):
        logger.info("Alarm resetted")
        self.send_command("RST")

    def set_TEC_enable(self, enable=True):
        if enable:
            logger.info(f"TEC state is ENABlE")
            self.send_command("T=1")
        else:
            logger.info(f"TEC state is DISABlE")
            self.send_command("T=0")

    def set_digital_modulation(self, extint):
        extint = extint.lower()
        if extint == "ext":
            logger.info("Digital modulation set to 'external'")
            self.send_command("TTL=1")
        elif extint == "int":
            logger.info("Digital modulation set to 'internal'")
            self.send_command("TTL=0")
        else:
            logger.warning("Invalid argument. Please enter either 'ext' or 'int'.")

    # some query commands ----------------------------------------------------
    # TO DO: some query commands are not wrapped (2023-01-04)-----------------
    def get_analog_control_mode(self):
        # return 'power' or 'current'
        rep = int(self.send_query("?ACC"))
        return ANALOG_CONTROL_MODE_TABLE[rep]

    def get_analog_modulation(self):
        # return 'power' or 'current'
        rep = int(self.send_query("?EXT"))
        return "ext" if rep == 1 else "int"
        # return ANALOG_MODULATION_TABLE[rep]

    def get_diode_current(self):
        # retrieve the measured current of the laser diode [mA]
        return float(self.send_query("?C"))

    def get_CDRH_state(self):
        # get the CDRH delay state
        return bool(int(self.send_query("?CDRH")))

    def get_modulation_state(self):
        rep = int(self.send_query("?CW"))
        return "cw" if rep == 1 else "modulated"

    def get_baseplate_temp(self):
        # get the measured temperature of the laser base plate [°C] 
        return float(self.send_query("?BT"))

    def get_diode_temp(self):
        # get the measured temperature of the laser diode [°C] 
        return float(self.send_query("?DT"))

    def get_interlock_state(self):
        # get the state of the interlock, 'open' or 'closed'
        rep = self.send_query("?INT")
        # logger.info(f"Interlock state is '{rep}'")
        return rep

    def get_laser_emission_activation(self):
        rep = int(self.send_query("?L"))
        return bool(rep)

    def get_max_laser_current(self):
        # return the maximum laser current availabe [mA]
        return float(self.send_query("?MAXLC"))

    def get_max_laser_power(self):
        # return the maximum laser power availabe [mW]
        return float(self.send_query("?MAXLP"))

    def get_laser_power(self):
        # return power [mW]
        return float(self.send_query("?P"))

    def get_current_setpoint(self):
        # retrieve the laser current setpoint setting [mA]
        return float(self.send_query("?SC"))

    def get_power_setpoint(self):
        # retrieve the laser power setpoint setting [mW]
        return float(self.send_query("?SP"))

    def get_status(self):
        # retrieve the laser power setpoint setting [mW]
        rep = int(self.send_query("?STA"))
        return STATUS_TABLE[rep]

    def get_TEC_state(self):
        # retrieves the TEC status, enabled=true, disabled=false
        return bool(int(self.send_query("?SS")))

    def get_digital_modulation(self):
        rep = int(self.send_query("?TTL"))
        return "ext" if rep==1 else 'int'

    # some helper methods------------------------------------------------------------
    def set_analog_modulation_external(self):
        self.set_analog_modulation("external")

    def set_laser_power(self, power, save_memory=False):
        if not save_memory:
            self.set_laser_power_realtime(power)
        else:
            self.set_laser_power_memory(power)

    def set_diode_current(self, power, save_memory=False):
        if not save_memory:
            self.set_diode_current_realtime(power)
        else:
            self.set_laser_power_realtime(power)
    
    def set_cw(self):
        self.set_modulation_state("cw")

    def set_pulsed(self):
        self.set_modulation_state("pulsed")
    
    def laser_on(self):
        self.set_laser_emission_activation(True)

    def laser_off(self):
        self.set_laser_emission_activation(False)
    
    def set_TTL_ext(self):
        self.set_digital_modulation("ext")

    def set_TTL_int(self):
        self.set_digital_modulation("int")

    def is_laser_on(self):
        return self.get_laser_emission_activation()

if __name__ == "__main__":
    # for testing only
    import time

    # !!! before all the tests: close the oxxius software
    laser = LaserControl()
    laser.open()

    # test laser on/off ------------------------------
    laser.laser_on()
    assert laser.is_laser_on()
    laser.laser_off()
    assert not laser.is_laser_on()
    assert laser.get_status() == "Standby"


    # test constant mode ---------------------------
    laser.set_analog_control_mode("power")
    assert laser.get_analog_control_mode() == "power"
    laser.set_analog_control_mode("current")
    assert laser.get_analog_control_mode() != "power"

    # test analog modulation -----------------------
    laser.set_analog_modulation("iNt")
    assert laser.get_analog_modulation() == "int"
    laser.set_analog_modulation("Ext")
    assert laser.get_analog_modulation() != "int"

    # test diode current ----------------------------------
    current_percent = 20.75
    laser.set_analog_control_mode("current")
    laser.set_diode_current(current_percent, save_memory=False)
    laser.laser_on()
    time.sleep(10) # allow laser current to settle
    cp_read = laser.get_diode_current()/laser.get_max_laser_current()
    laser.laser_off()
    assert round(current_percent, -1) == round(cp_read*100, -1)

    # test CDRH state ---------------------------------
    laser.set_CDRH_state(True)
    assert laser.get_CDRH_state()
    laser.set_CDRH_state(False)
    assert laser.get_CDRH_state() == False

    # test modulation type cw/modulated -------------------------
    laser.set_modulation_state("CW")
    assert laser.get_modulation_state() == "cw"
    laser.set_modulation_state("Pulsed")
    assert laser.get_modulation_state() == "modulated"

    # test laser power setting ----------------------------
    power = 32.75 # [mW]
    laser.set_analog_control_mode("power")
    laser.set_laser_power(power, save_memory=True)
    laser.laser_on()
    time.sleep(10) # allow laser current to settle
    power_read = laser.get_laser_power()
    laser.laser_off()
    assert round(power) == round(power_read)

    # test alarm reset --------------------------------------
    laser.laser_off()
    laser.reset_alarm()
    time.sleep(10) # wait for reset
    assert laser.get_status() == "Laser ON"
    laser.laser_off()
    assert laser.get_status() == "Standby"

    # test TEC states -------------------------------
    laser.set_TEC_enable(enable=False)
    assert laser.get_TEC_state() == False
    assert laser.get_status() == "Sleep"
    laser.set_TEC_enable()
    assert laser.get_TEC_state()

    # test TTL digital modulation ---------------------
    laser.set_analog_control_mode("power")
    laser.set_digital_modulation("ExT")
    assert laser.get_digital_modulation() != "ext"
    laser.set_analog_control_mode("current")
    laser.set_digital_modulation("ExT")
    assert laser.get_digital_modulation() == "ext"
    laser.set_digital_modulation("Int")
    assert laser.get_digital_modulation() == 'int'

    laser.close() # IMPORTANT to close the connection!!