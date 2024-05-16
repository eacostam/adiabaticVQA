# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 14:22:45 2023
 
@author: eacosta
"""
# Import Libraries
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit import ParameterVector
from qiskit import BasicAer 
from qiskit_symb import Operator as SymbOperator
from qiskit import transpile, IBMQ

import time

from config import ANGLE_SHIFT, LOG_DIR
from config import QISKIT_HUB, QISKIT_GROUP, QISKIT_PROJECT, QISKIT_BACKEND, QISKIT_TOKEN
from helpers import qcprint

# Uin Class
class Uin_FM():
  ordered_parameters = []

  def __init__(self, num_qubits, param_prefix):
    self.uin_qreg = QuantumRegister(num_qubits, "q")
    self.uin_circuit = QuantumCircuit(self.uin_qreg, name="Uin")

    self.ordered_parameters = ParameterVector(param_prefix, num_qubits) 

    param_ind = 0
    for q in range(num_qubits):
      self.uin_circuit.h(q)
      self.uin_circuit.ry(self.ordered_parameters[param_ind], q)
      param_ind += 1
    
  def ordered_parameters(self):
    return self.params

  def circuit(self):
    return self.uin_circuit

# Ansatz Class
class Ansatz():
  ordered_parameters = []
  
  def __init__(self, num_qubits, param_prefix):
    self.a_qreg = QuantumRegister(num_qubits)
    self.a_circuit = QuantumCircuit(self.a_qreg, name="Ansatz")

    self.ordered_parameters = ParameterVector(param_prefix, num_qubits)
    
    self.a_circuit.cx(3,2)
    self.a_circuit.rz(self.ordered_parameters[0],2)
    self.a_circuit.cx(3,2)
    self.a_circuit.barrier()

    self.a_circuit.cx(2,1)
    self.a_circuit.rz(self.ordered_parameters[1],1)
    self.a_circuit.cx(2,1)
    self.a_circuit.barrier()

    self.a_circuit.cx(1,0)
    self.a_circuit.rz(self.ordered_parameters[2],0)
    self.a_circuit.cx(1,0)
    self.a_circuit.barrier()

    self.a_circuit.cx(0,3)
    self.a_circuit.rz(self.ordered_parameters[3],3)
    self.a_circuit.cx(0,3)
    self.a_circuit.barrier()

  def ordered_parameters(self):
    return self.ordered_parameters

  def circuit(self):
    return self.a_circuit

  def get_operator(self):
    opqrnn = SymbOperator(self.a_circuit)
    oper = opqrnn.to_sympy()
    operArr = np.array(oper)
    qcprint("Type: " + str(type(operArr)))
    qcprint("Shape: " + str(np.shape(operArr)))
    qcprint(operArr)
    
    return operArr
  
# QRNN CIRCUIT
class QRNN():
    qubits = None
    cbits = None
    qrb_rep = 2
    circuit = QuantumCircuit()
    feature_map_arr = []
    ansatz_arr = []
    QRB_arr = []
    parameters = {}
    
    # builds the circuit
    def __init__(self, input_qubits, circuit_size, qrb_rep, measured_qubits):
        qcprint("Building Circuit with " + str(qrb_rep) + " QRB repetitions")
        self.qrb_rep = qrb_rep
        
        #FEATURE_MAP = []
        indFM=0
        for fm in range(qrb_rep):
          fm = Uin_FM(input_qubits, 'x'+str(indFM)) 
          self.feature_map_arr.append(fm)
          indFM = indFM+1
        
        #ANSATZ = []
        indAnsatz=0
        for fm in range(qrb_rep):
          ansatz = Ansatz(circuit_size, 'θ'+str(indAnsatz))
          self.ansatz_arr.append(ansatz)
          indAnsatz = indAnsatz+1
        
        self.qubits = QuantumRegister(circuit_size, "q")
        self.cbits = ClassicalRegister(measured_qubits, "c")
        self.circuit = QuantumCircuit(self.qubits, self.cbits)
        
        #QRB = []
        ind = 0
        for fm in range(qrb_rep):
          qrb = QuantumCircuit(self.qubits, self.cbits, name=("QRB"+str(ind)))
          qrb.append(self.feature_map_arr[ind].circuit(), [x for x in range(circuit_size-input_qubits, circuit_size, 1)])
          qrb.barrier()
          qrb.append(self.ansatz_arr[ind].circuit(), [x for x in range(0, circuit_size, 1)])
          qrb.measure(self.qubits[circuit_size-1], self.cbits[0])
          qrb.barrier()
          for i in range(circuit_size-1, input_qubits-1, -1):
            qrb.reset(i)
          qrb.barrier()
          self.QRB_arr.append(qrb)
          ind = ind+1
        
        for qrbi in self.QRB_arr:
          self.circuit.append(qrbi, [q for q in range(circuit_size)], [c for c in range(measured_qubits)])
        
        printable = self.circuit.decompose()
        printable.draw(output='mpl', reverse_bits = True, filename=LOG_DIR+("qrnn_circuit_comprised_{}.png".format(time.time())))
        printable = printable.decompose().decompose()
        printable.draw(output='mpl', reverse_bits = True, filename=LOG_DIR+("qrnn_circuit_decomposed_{}.png".format(time.time())))
        
    
    # Circuit update
    def update(self, data, ansatz_params):
        # Update Feature Map parameters
        data_ind = 0
        for fm in self.feature_map_arr:
          for p in fm.ordered_parameters:
            self.parameters[p] = np.arccos(data[data_ind])
          data_ind += 1

        # Update Ansatz parameters
        qrb_ind = 0
        for qrb_unit in self.ansatz_arr:
          for i, p in enumerate(qrb_unit.ordered_parameters):
              self.parameters[p] = ansatz_params[i]*(qrb_ind*ANGLE_SHIFT)
          qrb_ind += 1
              
        out =  self.circuit.assign_parameters(self.parameters, inplace = False)
        return out

# Circuit run
def label_execution(results):
    shots = sum(results.values())
    probabilities = {0: 0, 1: 0}

    for bitstring, counts in results.items():
     probabilities[int(bitstring)] += counts / shots

    return probabilities

# Circuit run on Q Gate Computer or Simulator
def circuits_execution(data, ansatz_params, qrnn, shots, realhw=False):
    circuits = []

    for d in data:
      circuits.append(qrnn.update(d, ansatz_params))
    
    circ_ind = 0
    for c in circuits:
        c.name = "QRNN-" + str(circ_ind)
        circ_ind += 1
    
    if not realhw: # run on simulator
        backend = BasicAer.get_backend('qasm_simulator')
        
        start_time = time.time()
        results = execute(circuits, backend, shots = shots).result()
        qcprint("Execute the circuits on " + str(len(data)) + " registers, and " + str(shots) + " shots, took %s sec" % (time.time() - start_time))
    else: # run on real hardware
        IBMQ.save_account(QISKIT_TOKEN, overwrite=True)
        provider = IBMQ.load_account()
        provider = IBMQ.get_provider(hub=QISKIT_HUB, group=QISKIT_GROUP, project=QISKIT_PROJECT)
        backend = provider.get_backend(QISKIT_BACKEND)
        transpiled = transpile(circuits, backend=backend)
        job = backend.run(transpiled, job_name="experiment_" + str(qrnn.qrb_rep) + "_qrbreps")

        results = job.result()
        counts = results.get_counts(circuits)
        print("Result: " + str(counts))
    
    classification = [label_execution(results.get_counts(c)) for c in circuits]
    
    return classification

# QRNN TEST
def qrnn_validation(data, labels, variational, qrnn, shots, realhw):
    qcprint("Testing with angles: " + str(variational))
    if not len(data) > 0:
        qcprint("Not enough data to validate the QRNN: " + len(data))
        return 0

    probability = circuits_execution(data, variational, qrnn, shots, realhw)

    predictions = [0 if p[0] >= p[1] else 1 for p in probability]
    accuracy = 0
    for i, prediction in enumerate(predictions):
      if prediction == labels[i]:
        accuracy += 1
    accuracy /= len(labels)
    qcprint("Accuracy: " + str(accuracy))
    return accuracy, predictions

# Get State Vector for input data
def get_input_StateVector(inputval, in_qubits):
  myq_reg = QuantumRegister(4, "q")
  myCircuit = QuantumCircuit(myq_reg)

  myFEATURE_MAP = Uin_FM(in_qubits, 'x'+str(0))
  myCircuit.append(myFEATURE_MAP.circuit(), [x for x in range(2,4,1)])
  myparameters = [np.arccos(inputval[0]) for i in myFEATURE_MAP.ordered_parameters]
  
  myCircuit = myCircuit.assign_parameters(myparameters)

  back = BasicAer.get_backend('statevector_simulator')
  result = execute(myCircuit, back, shots = 100).result()
  state_vector = result.get_statevector(myCircuit, decimals=2)

  return state_vector

def get_output_StateVector(output):
  # Create a QuantumCircuit with 2 qubits and 2 classical bits
  myq_reg = QuantumRegister(4, "q")
  myCircuit = QuantumCircuit(myq_reg)

  # Add quantum gates to the circuit
  if (output==1):
    myCircuit.x(3)

  back = BasicAer.get_backend('statevector_simulator')
  result = execute(myCircuit, back, shots = 100).result()
  state_vector = result.get_statevector(myCircuit, decimals=2)

  return state_vector