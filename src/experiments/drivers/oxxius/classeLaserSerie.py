import serial
import sys
import glob

TIMEOUT = 0.5
TIMEOUT_INIT = 0.1


class LasersList():
    def __init__(self):
        self.count = 0
        self.laserslist = []
        self.ports_com = self.serial_ports()
        for com in self.ports_com:
            ser = serial.Serial()
            ser.baudrate = 38400
            ser.port = com
            ser.timeout = TIMEOUT_INIT
            
            hid = self.send(ser, "?HID", True)
            if hid.startswith("LAS"):
                serial_number = (hid.split(','))[0]
                inf = self.send(ser, "inf?", False)
                infos = inf.split('-')
                type = infos[0]
                couleur = infos[1]
                puissance = infos[2]
                self.laserslist.append([serial_number, type, couleur, puissance, com, ser.baudrate])
            else:
                ser.baudrate = 19200
                hid = self.send(ser, "?HID", True)
                if hid.startswith("LAS"):
                    serial_number = hid.split(',')[0]
                    inf = self.send(ser, "inf?", False)
                    infos = inf.split('-')
                    type = infos[0]
                    couleur = infos[1]
                    puissance = infos[2]
                    self.laserslist.append([serial_number, type, couleur, puissance, com, ser.baudrate])
    
    def send(self,ser, command, init):
        ser.open()
        if init:
            ser.write(b"dummy\r\n")
            rep = ser.readline().decode()
        ser.write((command+"\r\n").encode('ASCII'))
        rep =  ser.readline().decode().replace("\r","").replace("\n","")
        ser.close()
        return rep
    
    def serial_ports(self):
        """ Lists serial port names

            :raises EnvironmentError:
                On unsupported or unknown platforms
            :returns:
                A list of the serial ports available on the system
        """
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')

        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                pass
        return result
    
    def get_list(self):
        return self.laserslist
        
    def get_serial_numbers(self):
        list = []
        for laser in self.laserslist:
            list.append(laser[0])
        return list
    def get_types(self):
        list = []
        for laser in self.laserslist:
            list.append(laser[1])
        return list
    def get_colors(self):
        list = []
        for laser in self.laserslist:
            list.append(laser[2])
        return list
    def get_powers(self):
        list = []
        for laser in self.laserslist:
            list.append(laser[3])
        return list  
    def get_coms(self):
        list = []
        for laser in self.laserslist:
            list.append(laser[4])
        return list  
    def find_serial_number(self, serial_number):
        for i,sn in enumerate(self.get_serial_numbers()):
            if sn == serial_number:
                return self.laserslist[i]
        return []
    def find_color(self, color):
        for i,co in enumerate(self.get_colors()):
            if co == color:
                return self.laserslist[i]
        return []
    def find_com(self, com):
        for i,co in enumerate(self.get_coms()):
            if co == com:
                return self.laserslist[i]
        return []       


class Laser():
    def __init__(self, laser_infos):
        self.serial_number = laser_infos[0]
        self.type = laser_infos[1]
        self.color = laser_infos[2]
        self.power = laser_infos[3]
        
        self.ser = serial.Serial()
        self.ser.port = laser_infos[4]
        self.ser.baudrate = laser_infos[5]
        self.ser.timeout = TIMEOUT
        
    def open(self):
        self.ser.open()
        self.ser.write(b"dummy\r\n")
        rep = self.ser.readline()
        
    def close(self):
        self.ser.close()
        
    def send(self, command):
        self.ser.write((command+"\r\n").encode('ASCII'))
        rep = self.ser.readline().decode().replace("\r", "").replace("\n", "")
        return rep

