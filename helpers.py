# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 14:22:45 2023
 
@author: eacosta
"""
# Import Libraries

import os
from config import LOG_FILE

# HELPER METHODS
def qcprint(str):
    with open(LOG_FILE, 'a' if os.path.exists(LOG_FILE) else 'w') as f:
        print(str, file=f)
