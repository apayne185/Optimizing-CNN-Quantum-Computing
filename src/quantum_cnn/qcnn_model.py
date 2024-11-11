import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy 
import numpy as np
from  qcnn_layers import quantum_conv_grover


'''creates qubits & readout operators in Cirq'''
#cluster_state_bits = cirq.GridQubit.rect(1, 8)   # array of 8 qubits  



'''Builds a (full) sequential model, 
    Uses Iterative Search --> Grovers Search to narrow down solutions'''
def build_qcnn(bits, target_state, symbols, num_classes):
    quantum_circuit = quantum_conv_grover(bits, symbols, target_state)

    

if __name__ == "__main__":
    target_state = [1,0,0,1,1,0,1,0]        #example target state
    qcnn_model = build_qcnn(cluster_state_bits, target_state)









#https://www.geeksforgeeks.org/introduction-to-grovers-algorithm/