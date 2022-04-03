# do not move this file --> needed for ROOT Variable
from datetime import datetime
import os
# global constants
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

#usefull helper function
def get_current_datetime_as_str():
    return "{:%Y_%m_%d_%H_%M}".format(datetime.now())