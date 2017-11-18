'''
Collection of little pythonic tools. Might need to organize this better in the future. 

@author: danielhernandez
'''

import datetime
import string

def addDateTime(s = ""):
    """
    Adds the current date and time at the end of a string.
    Inputs:
        s -> string
    Output:
        S = s_Dyymmdd_HHMM
    """
    date = str(datetime.datetime.now())
    allchars = string.maketrans('','')
    nodigs = allchars.translate(allchars, string.digits)
    date = date.translate(allchars, nodigs)
    return s + '_D' + date[2:8] + '_' + date[8:12]




if __name__ == "__main__":
    print addDateTime('Hello')
    print addDateTime()