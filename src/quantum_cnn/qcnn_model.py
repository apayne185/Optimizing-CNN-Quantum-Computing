import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy 
from qcnn_layers import *
import yaml



with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

output_activation = config['output_activation']


class QuantumToClassical(tf.keras.layer.Layer):
    def __init__(self, quantum_layer): 
        super(QuantumToClassical, self).__init__()
        self.quantum_layer = quantum_layer

    def call(self, inputs):
        #performs quantum measurement --> circuit execution
        quantum_output = self.quantum_layer(inputs)

        #converts qmeasurement into classicial data --> expectation vakue
        classical_output = tfq.layers.Expectation()(quantum_output)

        return classical_output 
        



'''Builds a (full) sequential model, 
    Uses Iterative Search --> Grovers Search to narrow down solutions'''
def build_qcnn(bits, target_state, symbols, num_classes):
    quantum_circuit = quantum_conv_grover(bits, symbols, target_state)
    quantum_layer = tfq.layers.PQC(quantum_circuit, cirq.Z(bits[-1]))

    #conversion layer
    q_to_c = QuantumToClassical(quantum_layer)

    #classical part of model
    model = tf.keras.Sequential([
        q_to_c, 
        tf.keras.layers.Dense(num_classes, output_activation)
    ])

    return model
    



