# -*- coding: utf-8 -*-
"""
Created on May 2024
 
@author: eacosta
"""
import time 
import os

# CONSTANTS
LOAD_DATA_FROM_URL = 'URL'
LOAD_DATA_FROM_FILE = 'FILE'
LOAD_DATA_FROM_HARDCODED = 'HARDCODED'

# Configuration Parameters 
SYMBOL = "SAN.MC"# "SAN.MC" Banco Santander, "RVPH" Reviva Pharmaceuticals, "T" AT&T
MAX_ROWS = 250 #25 min, -1 loads the entire dataset
DEBUG_PRINT_LEN = 10
LOG_DIR = "./log/"
LOG_FILE = LOG_DIR + "log{}.txt".format(time.time())
LOAD_DATA_FROM = LOAD_DATA_FROM_URL
MAX_ITER = 100 #10 min, 100 ok, Number of training iteration
SHOTS = 1024 #10 min, 1024 ook
IN_QUBITS = 2 #Input data qubits
CIRCUIT_SIZE = IN_QUBITS * 2
QRB_REP = 3 # Repetitions of the Quantum Recurrent Block
ANGLE_SHIFT = 0.001 #Small shift to apply on further repetitions
VQC_RANDOM_INIT = False
DISCRETE_EXT_PARTS = 7 #7 takes very long time and memory in simulation to finish
QISKIT_REAL = False # True to run on physical hardware, False for simulation
ADIABATIC_REAL = False # True to run on physical hardware, False for simulation

QISKIT_HUB = 'ibm-q'
QISKIT_GROUP = 'open'
QISKIT_PROJECT = 'main'
DWAVE_TIMELIMIT = 60

QISKIT_TOKEN = 'toktok'  # Put your IBM Quantum token
QISKIT_BACKEND = 'ibm_brisbane'#ibm_nairobi' #Put the backend to use
DWAVE_TOKEN = 'toktok'  #Put your DWave token


if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
