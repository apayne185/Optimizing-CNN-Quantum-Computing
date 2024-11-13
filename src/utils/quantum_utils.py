import cirq
from qisket import IBMQ
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def initialize_ibmq(api_token=None):
    api_token = api_token or config['quantum_backend'].get('api_token')
    if api_token: 
        IBMQ.save_account(api_token=None)
    provider = IBMQ.load_account()
    backend_name = config['quantum_backend'].get('backend_name', 'ibmq_qasm_simulator')
    return provider.get_backend(backend_name)

def create_qubit(num_qubits):
    num_qubits = num_qubits or config['quantum_preprocessing'].get('num_qubits',4)
    return [cirq.GridQubits(0,i) for i in range(num_qubits)]