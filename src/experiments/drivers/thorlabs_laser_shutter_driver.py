import os
import time
import sys
import clr

clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.KCube.SolenoidCLI.dll")
from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.KCube.SolenoidCLI import *
from System import Decimal  # necessary for real world units

class LaserShutter:
    """The main entry point for the application"""

    # Uncomment this line if you are using simulation
    # SimulationManager.Instance.InitializeSimulations()

    def __init__(self, serial_id):
        
        DeviceManagerCLI.BuildDeviceList()

        # create new device
        self.serial_no = serial_id  # assigns proper serial no. to class param
        self.millisecs = 250

        # Connect
        self.device = KCubeSolenoid.CreateKCubeSolenoid(self.serial_no)
        self.device.Connect(self.serial_no)

        # Ensure that the device settings have been initialized
        if not self.device.IsSettingsInitialized():
            self.device.WaitForSettingsInitialized(10000)  # 10 second timeout
            assert self.device.IsSettingsInitialized() is True

        # Start polling and enable
        self.device.StartPolling(self.millisecs)  # 250 ms polling rate
        time.sleep(0.25)
        self.device.EnableDevice()
        time.sleep(0.5)  # Wait for device to enable

        # Get Device Information and display description
        self.device_info = self.device.GetDeviceInfo()
        # print(device_info.Description)

        # set laser shutter to "Manual" mode --> this is only mode that will need to be used
        self.device.SetOperatingMode(SolenoidStatus.OperatingModes.Manual)

    def open_shutter(self):
        self.device.SetOperatingState(SolenoidStatus.OperatingStates.Active)

    def close_shutter(self):
        self.device.SetOperatingState(SolenoidStatus.OperatingStates.Inactive)

    def disconnect(self):
        # Stop Polling and Disconnect
        self.device.StopPolling()
        self.device.Disconnect()

