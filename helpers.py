# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 14:22:45 2023
 
@author: eacosta
"""
# Import Libraries

import os
import datetime
from config import LOG_FILE

# HELPER METHODS
def qcprint(string):
    with open(LOG_FILE, 'a' if os.path.exists(LOG_FILE) else 'w') as f:
        now = datetime.datetime.now()
        str2print = now.strftime("%Y-%m-%d %H:%M:%S")
        print(str2print, file=f)
        print(string, file=f)
