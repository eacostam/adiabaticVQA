# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 14:22:45 2023
 
@author: eacosta
"""
from io import StringIO
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split

from config import LOAD_DATA_FROM_URL, LOAD_DATA_FROM_FILE, LOAD_DATA_FROM_HARDCODED
from config import DEBUG_PRINT_LEN, QRB_REP, SYMBOL
from helpers import qcprint

# DATA MANAGEMENT
# data load
def data_load(load_from):
    qcprint("Loading data for " + str(SYMBOL))
    if (load_from == LOAD_DATA_FROM_URL):
      todays_date = int(time.time())
      YAHOO_URL = "https://query1.finance.yahoo.com/v7/finance/download/"+SYMBOL+"?period1=554601600&period2=" + str(todays_date) + "&interval=1d&events=history&includeAdjustedClose=true"
      qcprint("Getting data from: " + str(YAHOO_URL))
      DATA_RAW = pd.read_csv(YAHOO_URL)
    elif (load_from == LOAD_DATA_FROM_FILE):
      DATA_RAW = pd.read_csv("dataset/SAN.csv")
    elif (load_from == LOAD_DATA_FROM_HARDCODED):
      data_str = StringIO("""Date,Open,High,Low,Close,Adj Close,Volume
    1987-07-30,0.000000,4.391340,4.319853,4.350490,0.723851,5826240
    1987-07-31,0.000000,4.370915,4.319853,4.330065,0.720452,2104056
    1987-08-03,0.000000,4.330065,4.309641,4.319853,0.718753,965736
    1987-08-04,0.000000,4.319853,4.227941,4.227941,0.703461,1479816
    1987-08-05,0.000000,4.217729,4.084967,4.105392,0.683070,1068552
    1987-08-06,0.000000,4.217729,4.115605,4.207516,0.700062,1429632
    1987-08-07,0.000000,4.227941,4.166667,4.166667,0.693265,820080
    1987-08-10,0.000000,4.176879,4.146242,4.166667,0.693265,835992
    1987-08-11,0.000000,4.197304,4.166667,4.197304,0.698363,192168
    1987-08-12,0.000000,4.227941,4.187092,4.217729,0.701761,235008
    1987-08-13,0.000000,4.207516,4.197304,4.197304,0.698363,298656
    1987-08-14,0.000000,4.207516,4.197304,4.197304,0.698363,462672
    1987-08-17,0.000000,4.217729,4.187092,4.187092,0.696664,1266840
    1987-08-18,0.000000,4.176879,4.146242,4.166667,0.693265,173808
    1987-08-19,0.000000,4.166667,4.146242,4.166667,0.693265,36720
    1987-08-20,0.000000,4.156454,4.146242,4.156454,0.691566,110160
    1987-08-21,0.000000,4.146242,4.125817,4.146242,0.689867,129744
    1987-08-24,0.000000,4.176879,4.156454,4.166667,0.693265,378216
    1987-08-25,0.000000,4.166667,4.146242,4.146242,0.689867,198288
    1987-08-26,0.000000,4.125817,4.064542,4.064542,0.676273,613224
    1987-08-27,0.000000,4.115605,4.044118,4.105392,0.683070,1159128
    1987-08-28,0.000000,4.248366,4.125817,4.227941,0.703461,429624
    1987-08-31,0.000000,4.248366,4.207516,4.248366,0.706859,126072""")
      DATA_RAW= pd.read_table(data_str, sep=',')
    else:
      DATA_RAW = []
    
    # Print list for verification
    qcprint("Total records: " + str(len(DATA_RAW)))
    qcprint(DATA_RAW.iloc[:DEBUG_PRINT_LEN])
      
    return DATA_RAW

# data preparation
def data_prep(data_set, max_rows):
    # Data Preparation
    # 1. Reverse ordering, for latest tick to be on top
    DATA_RAW = data_set.iloc[::-1]
    # Cut at MAX_ROWS to expedite development
    DATA_RAW = DATA_RAW[:max_rows]
    # 2. Add Variation column as the difference of closing price against previous day
    DATA_RAW['Variation'] = DATA_RAW['Close'] - DATA_RAW['Close'].shift(-1)
    DATA_RAW['Close_norm'] = (2* ((DATA_RAW['Close'] - min(DATA_RAW['Close'])) / (max(DATA_RAW['Close'])-min(DATA_RAW['Close']))) ) -1
    
    # 3. Add Label.  0=decrease, 1=increase.
    DATA_RAW['Label'] = np.where(DATA_RAW['Variation']<0, 0, 1)
    # 4. Add Weekday column, where 0:Monday, 6:Sunday
    DATA_RAW['Weekday'] = ( pd.to_datetime(DATA_RAW['Date']).dt.dayofweek )
    
    qcprint("Records loaded: " + str(len(DATA_RAW)))
    
    # Train and test data preparation
    # Remove headers
    DATA_RAW_TRAIN, DATA_RAW_TEST = train_test_split(DATA_RAW, test_size=0.2)
    TRAIN_DATA_RAW = DATA_RAW_TRAIN[:].Close_norm.tolist()
    TRAIN_LABELS = DATA_RAW_TRAIN[:].Label.tolist()
    TEST_DATA_RAW = DATA_RAW_TEST[:].Close_norm.tolist()
    TEST_LABELS = DATA_RAW_TEST[:].Label.tolist()

    qcprint("Training records: " + str(len(TRAIN_DATA_RAW)))
    qcprint(TRAIN_DATA_RAW[:DEBUG_PRINT_LEN])
    qcprint(TRAIN_LABELS[:DEBUG_PRINT_LEN])
    qcprint("Test records: " + str(len(TEST_DATA_RAW)))
    qcprint(TEST_DATA_RAW[:DEBUG_PRINT_LEN])
    qcprint(TEST_LABELS[:DEBUG_PRINT_LEN])
    
    # Prepare data with historical depth. X historical entry values.
    TRAIN_DATA = []
    index = 0
    while index < len(TRAIN_DATA_RAW)-(QRB_REP-1):
      features_list = []
      for i in range(QRB_REP):
          if not (pd.isna(TRAIN_DATA_RAW[index+i])):
            features_list.append(TRAIN_DATA_RAW[index+i])
    
      if (len(features_list)==QRB_REP):
        TRAIN_DATA.append(features_list)
      index = index + 1
    TRAIN_LABELS = TRAIN_LABELS[:len(TRAIN_DATA)]
    
    TEST_DATA = []
    index = 0
    while index < len(TEST_DATA_RAW)-(QRB_REP-1):
      features_list = []
      for i in range(QRB_REP):
          if not (pd.isna(TEST_DATA_RAW[index+i])):
            features_list.append(TEST_DATA_RAW[index+i])
    
      if (len(features_list)==QRB_REP):
        TEST_DATA.append(features_list)
      index = index + 1
    TEST_LABELS = TEST_LABELS[:len(TEST_DATA)]
    
    qcprint("Training records normalized to QRB repetitions:" + str(len(TRAIN_DATA)))
    qcprint("Test records normalized to QRB repetitions:" + str(len(TEST_DATA)))
      
    return TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS
