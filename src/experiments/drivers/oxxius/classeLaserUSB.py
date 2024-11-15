import usb


class LasersList(object):
    def __init__(self):
        self.las = 0
        self.laserslist = []
        self.LisLas = self.find_devices()
        for item in self.LisLas:
            hid = self.send(item, '?HID')
            if hid.startswith("LAS"):
                serial_number = (hid.split(','))[0]
                inf = self.send(item, "inf?")
                infos = inf.split('-')
                type_laser = infos[0]
                couleur = infos[1]
                puissance = infos[2]
                self.laserslist.append([serial_number, type_laser, couleur, puissance,item])
        # print(self.laserslist)
    #def update(self):
        
    def find_devices(self):
        """Returns a list of the available devices"""
        #try:
        #    for item in self.LisLas:
        #        usb.util.dispose_resources(item)
        #except:
        #    "Keep going"
        self.las = usb.core.find(find_all=1, idVendor=0x0403, idProduct=0x90D9)
        self.LisLas = []
        for item in self.las:
            self.LisLas.append(item)
        return self.LisLas

    def send(self,las,command):
        "Writes a command to the laser"
        las.set_configuration()
        #if init:
        #    las.write(0x02, b'dummy\0', 100)
        #    rep = las.read(0x81, 30, 1000)
        ttt = las.write(0x02, (command+'\0').encode(), 100)
        rep = las.read(0x81, 30, 1000)
        usb.util.dispose_resources(las)
        rep = bytearray(rep).decode().replace("\0", "")
        #print(rep)
        return rep

    def usb_ports(self):
        "Returns a list of usb ports available on the machine"

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

    def get_usbs(self):
        list = []
        for laser in self.laserslist:
            list.append(laser[4])
        return list

    def find_serial_number(self, serial_number):
        for i, sn in enumerate(self.get_serial_numbers()):
            if sn == serial_number:
                return self.laserslist[i]
        return []

    def find_color(self, color):
        for i, co in enumerate(self.get_colors()):
            if co == color:
                return self.laserslist[i]
        return []

    def find_usbs(self, com):
        for i, co in enumerate(self.get_usbs()):
            if co == com:
                return self.laserslist[i]
        return []


class Laser(object):
    def __init__(self, laser_infos):
        self.state = 0
        # print(laser_infos)
        self.serial_number = laser_infos[0]
        self.type = laser_infos[1]
        self.color = laser_infos[2]
        self.power = laser_infos[3]
        self.dev = laser_infos[4]
        
        #self.port = laser_infos[4]

    def open(self):
        self.dev.set_configuration()

    def close(self):
        usb.util.dispose_resources(self.dev)

    def send(self, input):
        try:
            chaine = input + '\0'
            self.dev.write(0x02, chaine.encode(), 100)
            rep = self.dev.read(0x81, 30, 1000)
            return bytearray(rep).decode().split('\0')[0]
        except:
            print("Unable to send instructions to the laser")
            return -1
