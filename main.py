# -*- coding: utf-8 -*-
"""
Created on May 2024
 
@author: eacosta

Required libraries
!pip install qiskit
!pip install qiskit_machine_learning
!pip install qiskit[visualization]
!pip install qiskit-algorithms
!pip install qiskit-ibm-runtime
!pip install qiskit-ibm-provider
!pip install pylatexenc
!pip install qiskit-symb
!pip install dwave-system
!pip install tensorflow
!pip install dimod
"""

# Import Libraries
import sys
import numpy as np

from config import LOAD_DATA_FROM, MAX_ROWS, IN_QUBITS, CIRCUIT_SIZE, MAX_ITER 
from config import QRB_REP,VQC_RANDOM_INIT, SHOTS
from config import DISCRETE_EXT_PARTS, QISKIT_REAL
from datamgt import data_load, data_prep
from qrnn import QRNN, get_input_StateVector, get_output_StateVector
from qrnn import qrnn_validation
from qrnn_train_classic import QRNN_optimizer_classic#, OptimizerLog
from qrnn_train_adiabatic import QRNN_optimizer_adiabatic
from ann import ANN

# MAIN
def main(run_qrnn_classical, run_qrnn_adiabatic, run_ann):
    # 1. Load and Prepare Data
    data_set = data_load(LOAD_DATA_FROM)
    train_data, train_labels, test_data, test_labels = data_prep(data_set, MAX_ROWS)
    
    # 2. Build QRNN circuit
    qrnn = QRNN(IN_QUBITS, CIRCUIT_SIZE, QRB_REP, 1)
    init_ansatz_params = np.array([2.19337516, 0.68064902, 2.38983917, 1.84192342])
    if VQC_RANDOM_INIT:
        init_ansatz_params = np.random.random(qrnn.ansatz_arr[0].num_parameters)
    qrnn.update(train_data[0], init_ansatz_params)

    
    # 3. Train QRNN - Classically
    if run_qrnn_classical==True or run_qrnn_classical=="True":
        qrr_trainer_c = QRNN_optimizer_classic(train_data, train_labels, qrnn, SHOTS)
        opt_var_c = qrr_trainer_c.train(init_ansatz_params)
        
        # Test
        accuracy_c, predictions = qrnn_validation(test_data, test_labels, np.abs(opt_var_c), qrnn, SHOTS, QISKIT_REAL)
        print("QRNN classical training accuracy: " + str(accuracy_c))
    
    # 4. Train QRNN - Adiabatic
    if run_qrnn_adiabatic==True or run_qrnn_adiabatic=="True":
        qrnn_optimizer_adiabatic = QRNN_optimizer_adiabatic(DISCRETE_EXT_PARTS, 
                                                            [np.real(get_input_StateVector(i, IN_QUBITS)) for i in train_data],
                                                            [np.real(get_output_StateVector(j)) for j in train_labels] )
        optimal = qrnn_optimizer_adiabatic.optimize()
        opt_var_a = qrnn_optimizer_adiabatic.get_angles(optimal)
        
        # Test 
        accuracy_a, predictions = qrnn_validation(test_data, test_labels, opt_var_a, qrnn, SHOTS, QISKIT_REAL)
        print("QRNN adiabatical training accuracy: " + str(accuracy_a))
 
    # 5 Classical Artificial Neural Network
    if run_ann==True or run_ann=="True":
        ann = ANN(train_data, train_labels, test_data, test_labels)
        ann.train(MAX_ITER)
        ann_accuracy = ann.validate()
        print("ANN accuracy: " + str(ann_accuracy))  

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])

#main(True, False, False)
