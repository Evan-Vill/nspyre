import numpy as np
import pandas as pd

class BFieldCalibrate:
    """
    B Field Calibration class to interface with NV widefield microscope.
    """
    def __init__(self, file):
        
        self.df = pd.read_excel(file, index_col=0)

    def find_nearest(self, colname, field):
        """
        Finds the nearest B field value to user input
        :return: Zaber position, B field value
        """
        df_sort = self.df.iloc[(self.df[colname] - field).abs().argsort()[1]] # find closest B field value in calibration data

        nearest_pos = df_sort.name
        nearest_val = int(df_sort[colname].tolist())

        return nearest_pos, nearest_val
    
    def find_value(self, colname, pos):
        """
        Find B field value for a given Zaber stage position
        :return: B field value
        """
        return self.df[colname][pos]
    
