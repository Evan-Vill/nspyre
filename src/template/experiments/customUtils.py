from os import path, mkdir
from datetime import datetime
from itertools import count
from rpyc.utils.classic import obtain

from nspyre.gui.widgets.save import save_json
from nspyre import DataSink
from nspyre import ProcessRunner




def flexSave(datasetName:str, expType:str, filename:str, dirs:list = ['E:\\Data\\']): #TODO: Make it so this hangs up data acquisition. Will need to use ProcessRunner, multithread acq and saving, and then make them talk to each other nicely. Gross
    '''Creates a save of the data a specified directory(ies) in a Dir\\DATE(YYMMDD)\\EXPERIMENT_TYPE\\EXP_TYPE TIME(HHMMSS) SAVE_TYPE.json structure
    Arguments:  *datasetName:str, name of data to be saved from dataserv
                *expType:str, name of the experiment (or name of folder to save in under the date)
                *filename:str, file name entered in experiment GUI
                *OBSOLETE: saveType:str, typically something like auto, closeout, final
                *dirs:list, dirs ending in \\ to save data to. Default is Jasper's Data driver'''

    if not len(dirs) > 0:
        raise ValueError('No directories specified for custom autosaver')
    
    now = datetime.now()
    with DataSink(datasetName) as dataSink:
        try:
            print("Trying to save in flexSave...")
            dataSink.pop(1)
            for dir in dirs:
                datePath = datePath = dir + now.strftime('%y_%m_%d') + '//' #dir for the date
                expPath = datePath + expType + '//' #dir for the exp on date
                filePath = expPath + expType + '_' + filename + '.json' #filename for the json, assumes autosaving is at <1Hz to avoid name collisions
                print(datePath, expPath, filePath)
                if not path.isdir(datePath): #create the date dir if it doesn't already exist
                    mkdir(datePath)
                if not path.isdir(expPath): #create the exp dir inside of date dir if it doesn't already exist
                    mkdir(expPath)
                if path.isfile(filePath):
                    # filePath = filePath + '_1'
                    filename = filename + '_1'
                    
                # print("obtained data: ", obtain(dataSink.data))
                save_json(filePath, obtain(dataSink.data)) #actually save the data as a json there
        except TimeoutError:
            raise ValueError(f'No data with name \'{datasetName}\' to save (or timed out)')



def saveInNewProc(datasetName:str, expNameForAutosave:str, saveType:str, dirs:list=None):
        '''Starts a new process (that shouldn't kill current processes) to save the data to a specified directory(ies) in a Dir\\DATE(YYMMDD)\\EXPERIMENT_TYPE\\EXP_TYPE TIME(HHMMSS) SAVE_TYPE.json structure
        Arguments:  *datasetName:str, name of data to be saved from dataserv
                    *expNameForAutosave:str, str that will used in saved file
                    *saveType:str, typically something like auto, closeout, final
                    *dirs:list, dirs ending in \\ to save data to. Default is None, which uses the default save directories from flexSave in CustomUtils.py. Default is None'''
        saveProc = ProcessRunner(kill=False)
        if dirs is not None:
            saveProc.run(flexSave, datasetName, expNameForAutosave, saveType, dirs)
        else:
            saveProc.run(flexSave, datasetName, expNameForAutosave, saveType)
        return(saveProc)



def setupIters(maxIterations):
    if maxIterations < 0:
        iters = count() #infinite iterator
    else:
        iters = range(maxIterations) 
    return(iters)


def setupLaser(gw, laserPower):
    if not gw.laser.is_on():
        raise(ValueError('LASER IS NOT ON'))
    gw.laser.set_power(laserPower)


def setupSigGen(gw, freq, rfPower, sigGenName:str='vaunix'):
    sigGen = getattr(gw, sigGenName)
    sigGen.setPwrLvl(rfPower)
    sigGen.setFreq(freq)
    sigGen.setIfRFOut(True)