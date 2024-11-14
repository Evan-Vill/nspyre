import oxxius.classeLaserSerie as classeLaserSerie
import oxxius.classeLaserUSB as classeLaserUSB

#from laser.classeLaserSerie import classeLaserSerie
#from laser.classeLaserUSB import classeLaserUSB
#from laser.classeLnCcUSB import classeLnCcUSB


class LasersList(object):
    def __init__(self):
        self.laserslist = []
        liste_usb = classeLaserUSB.LasersList()
        liste_serie = classeLaserSerie.LasersList()
        for item in liste_usb.get_list():
            las = ['usb'] + item
            self.laserslist.append(las)
        for item in liste_serie.get_list():
            las = ['rs232'] + item
            self.laserslist.append(las)
        # print(self.laserslist)

    def get_list(self):
        return self.laserslist

    def get_serial_numbers(self):
        list = []
        for laser in self.laserslist:
            list.append(laser[1])
        return list

    def get_types(self):
        list = []
        for laser in self.laserslist:
            list.append(laser[2])
        return list

    def get_colors(self):
        list = []
        for laser in self.laserslist:
            list.append(laser[3])
        return list

    def get_powers(self):
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
        # print("LASER INFO: ", laser_infos)
        self.isSerie = ('rs232' in laser_infos[0])
        # print(self.isSerie)
        las_infos = laser_infos[1:]
        # print(las_infos)
        # if self.isSerie:
        #     self.las_serie = classeLaserSerie.Laser(las_infos)
        # else:
        #     self.las_usb = classeLaserUSB.Laser(las_infos)
        if self.isSerie:
            self.las = classeLaserSerie.Laser(las_infos)
        else:
            self.las = classeLaserUSB.Laser(las_infos)


    def open(self):
        # if self.isSerie:
        #     self.las_serie.open()
        # else:
        #     self.las_usb.open()
        self.las.open()

    def close(self):
        # if self.isSerie:
        #     self.las_serie.close()
        # else:
        #     self.las_usb.open()
        self.las.close()

    def send(self, command):
        # if self.isSerie:
        #     return self.las_serie.send(command)
        # else:
        #     return self.las_usb.send(command)
        return self.las.send(command)
