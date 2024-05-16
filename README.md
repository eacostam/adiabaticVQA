ERNESTO ACOSTA MARTIN

Subject:  Adiabatic Training for Variational Quantum Algorithms
 
Universidad de Granada
Spain

Required libraries installation:
- !pip install qiskit
- !pip install qiskit_machine_learning
- !pip install qiskit[visualization]
- !pip install pylatexenc
- !pip install qiskit-symb
- !pip install dwave-system
- !pip install tensorflow
- !pip install dimod
- !pip install qiskit-ibm-runtime
- !pip install qiskit-ibmq-provider


In order to execute the program, invoke main() method on main.py module
specifying whether you want to execute either of the three available methods:
- QRNN with classical training
- QRNN with adiabatic training
- Clasical ANN

if you want to execute the three methods, run from the command line:
$ python main.py True True True

if you want to execute only the ANN methods, run from the command line:
$ python main.py False False True

The configuration parameters can be adjusted in config.py, like which company
you want to use their stock prices for training, how much data to consider,
how many QRB repetitions, and so on.
