import cirq
import sympy
import tensorflow_quantum as tfq
import yaml



with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


def cluster_state_circuit(bits):
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(bits))
    for this_bit, next_bit in zip(bits, bits[1:] + [bits[0]]):
        circuit.append(cirq.CZ(this_bit, next_bit))
    return circuit



'''rotation of the bloch sphere about the X, Y, Z axis that depends on values in symbols.
   
    Applies series of rotations to single qubit (represneted by bit) for flexible control
    over qubits rotation on Bloch sphere (geometric representation of qubits state to allow
    for superpositions) '''
def one_qubit_unitary(bit, symbols):
    return cirq.Circuit(
        cirq.X(bit)**symbols[0],
        cirq.Y(bit)**symbols[1],
        cirq.Z(bit)**symbols[2]
    )



''' quantum circuit that creates and arbitrary two qubit unitary,
    applies series of rotations/entangling operations to two qubits
'''
def two_qubit_unitary(bits, symbols):
    circuit = cirq.Circuit()

    #series of rotations along X, Y, Z axes with each rotation controlled by parameter from symbols
    circuit += one_qubit_unitary(bits[0], symbols[0:3])
    circuit += one_qubit_unitary(bits[1], symbols[3:6])

    #applies 3 types of 2-qubit gates ZZ, YY, XX which creates entanglement between the 2 qubits
    circuit += [cirq.ZZ(*bits)**symbols[6]]
    circuit += [cirq.YY(*bits)**symbols[7]]
    circuit += [cirq.XX(*bits)**symbols[8]]

    #another set of rotations applied to each qubit to complete the circuit
    circuit += one_qubit_unitary(bits[0], symbols[9:12])
    circuit += one_qubit_unitary(bits[1], symbols[12:])  

    return circuit




'''makes quantum circuit to parameterize 'pooling' operation,
    which combines the info from 2 qubits (reduces amount of info-reduces dimensions)
'''
def two_qubit_pool(source_qubit, sink_qubit, symbols):
    pool_circuit = cirq.Circuit()     #empty qcircuit
    #basis selector means 1 qubit unitary transformation/rotation, parameterized by the first 3 symbols
    pool_circuit.append(one_qubit_unitary(sink_qubit, symbols[0:3]))
    pool_circuit.append(one_qubit_unitary(source_qubit, symbols[3:6]))
    #CNOT creates entanglement
    pool_circuit.append(cirq.CNOT(source_qubit, sink_qubit))
    #basis selector is applied to reverse the effect of the sink_basis_selector --POOLS
    pool_circuit.append(one_qubit_unitary(sink_qubit, symbols[0:3])**-1)
    return pool_circuit




''' defines 1D quantum convolution as applicaiton of 2 qubit parameterized unitary
    to each pair of adjacent qubits with a stride of 1
'''
def quantum_conv_circuit(bits, symbols):
    circuit = cirq.Circuit()
    for first, second in zip(bits[0::2], bits[1::2]):
        circuit += two_qubit_unitary([first, second], symbols)
    for first, second in zip(bits[1::2], bits[2::2]+[bits[0]]):
        circuit += two_qubit_unitary([first, second], symbols)
    return circuit



'''Grovers search w quantum assisted data preprocessing'''
def grover_search(bits, symbols, target_state): 
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(bits))

    #apply oracle to flip phase of target state
    for qubit, state in zip(bits, target_state):
        if state == 0:
            circuit.append(cirq.X(qubit))

    #marks target state
    circuit.append(cirq.Z(bits[-1]).controlled_by(*bits[:-1]))

    #reverses oracle modifications
    for qubit,state in zip(bits, target_state):
        if state ==0:
            circuit.append(cirq.X(qubit))

    #diffusion operator to amplify results
    circuit.append(cirq.H.on_each(bits), cirq.X.on_each(bits))
    circuit.append(cirq.Z(bits[-1]).controlled_by(*bits[:-1]))
    circuit.append(cirq.X.on_each(bits), cirq.H.on_each(bits))


    return circuit

    

'''Combination allows hybrid quantum preprocessing layer'''
def quantum_conv_grover(bits, symbols, target_state):
    circuit = quantum_conv_circuit(bits, target_state)
    circuit += grover_search(bits, symbols, target_state)

    return circuit 