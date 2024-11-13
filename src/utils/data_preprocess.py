'''we need to preprocess data but using quantum computing (grovers search)
to optimize preprocessing'''

import numpy as np
import tensorflow as tf
import cirq
import sympy
import tensorflow_quantum as tfq
from multiprocessing import Pool
from quantum_cnn.qcnn_layers import quantum_conv_grover



"loads, normalizes data"
def normalize_data(data_path):
    data = np.load(data_path)
    data = data.astype('float32')/255.0
    return tf.convert_to_tensor(data) 



'''encodes data as quantum state (rotations) -- quantum feature transformation'''
def encode_quantum_state(data_sample, qubits): 
    circuit = cirq.Circuit()
    for i, bit in enumerate(data_sample):
        circuit.append(cirq.rx(bit * np.pi).on(qubits[i]))
    return circuit



'''quantum feature transformation to dataset by encoding each sample with qconvolution &
grovers search'''
def quantum_preprocess(data, qubits, symbols):
    processed_data = []
    target_state = [1,0,0,1]   #modify this as needed

    for sample in data:
        encoding_circuit = encode_quantum_state(sample, qubits)
        processing_circuit = quantum_conv_grover(qubits, symbols, target_state=target_state)
        full_circuit = encoding_circuit + processing_circuit
        
        quantum_tensor = tfq.convert_to_tensor([full_circuit])
        processed_data.append(quantum_tensor)

    return tf.concat(processed_data, axis=0)



'''hpc parallelization for q circuit execution'''
def quantum_parallel(data, qubits, symbols, processes=4):
    data_chunks = np.array_split(data, processes)
    with Pool(processes=processes) as pool:
        results =pool.starmap(quantum_preprocess, [(chunk, qubits, symbols) for chunk in data_chunks])
    return tf.concat(results, axis=0)


'''complete preprocessing pipeline
    --> load data, normalize it, perform qpreprocessing'''
def preprocess_pipeline(data_path, qubits, symbols, hpc_enabled=False, processes=4):
    data = normalize_data(data_path)
    if hpc_enabled:
        processed_data = quantum_parallel(data, qubits, symbols, processes=processes)
    else: 
        processed_data = quantum_preprocess(data, qubits, symbols)
    
    return processed_data







