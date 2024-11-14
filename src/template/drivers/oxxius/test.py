#from laserClass import Devices, LaserBoxxCom
import classeLaser

# test LaserList
listlasers = classeLaser.LasersList()
print(listlasers.get_list())

#print(listlasers.find_serial_number('LAS-03957'))
#print(listlasers.find_color('561'))
las1_infos = listlasers.get_list()[0]
las2_infos = listlasers.get_list()[1]
#print(las1_infos, las2_infos)
las1 = classeLaser.Laser(las1_infos)
las2 = classeLaser.Laser(las2_infos)
las1.open()
las2.open()
print(las1.send('?sv').encode('ASCII'))
print(las2.send('?iv').encode('ASCII'))
las1.close()
las2.close()

