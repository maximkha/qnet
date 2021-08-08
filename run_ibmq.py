import enum
from qiskit import IBMQ
import qiskit
from CREDS import TOKEN
from qiskit import QuantumCircuit, transpile
import numpy as np
from numpy.linalg import norm
from qiskit.test.mock import FakeBoeblingen, FakeAthens, FakeQasmSimulator, FakeYorktown
from pathlib import Path
import re
from qiskit.providers.aer import AerSimulator

# 0 - perfect
# 1 - clone from source
# 2 - real
M_TYPE = 2
HOST = 'ibmq_bogota'

JOBS = 3

def vecs_to_circs(vecs):
    input_circuits = []
    state_vec = np.array([v/norm(v) for v in vecs], dtype=complex)
    print(state_vec)
    print(np.array([norm(v) for v in state_vec]))
    for X in state_vec:
        circuit = QuantumCircuit(int(np.log2(vecs.shape[-1])))
        circuit.initialize(X)
        circuit = transpile(circuit, optimize_backend)
        input_circuits.append(circuit)
    return input_circuits

# provider = IBMQ.enable_account(TOKEN)
# backend = provider.get_backend('ibmq_manila')

# qnn_inputs = np.array([[0,1,0,1],[1,0,0,1],[0,0,0,1],[1,1,0,1]], dtype=complex)
# qnn_inputs = np.array([[1,0],[0,1]], dtype=complex)
qnn_inputs = np.array([[0,1,0,1],[1,0,0,1],[0,0,0,1],[1,1,0,1]], dtype=complex)

optimize_backend = FakeAthens() #FakeAthens()
run_backend = optimize_backend #FakeAthens()# FakeQasmSimulator() # optimize_backend
provider = IBMQ.enable_account(TOKEN)
if M_TYPE == 2:
    run_backend = provider.get_backend(HOST) #provider.get_backend('ibmq_manila')
    optimize_backend = run_backend
elif M_TYPE == 1:
    real = provider.get_backend(HOST)
    optimize_backend = AerSimulator.from_backend(real)
    run_backend = optimize_backend
elif M_TYPE == 0:
    real = provider.get_backend(HOST)
    optimize_backend = AerSimulator.from_backend(real)
    run_backend = FakeQasmSimulator()

layer_circuits = []

for layer_file in (Path(".") / "intermediate").glob("mat*.qasm"):
    qc = QuantumCircuit.from_qasm_file(layer_file)
    print(f"Importing {layer_file}")
    layer_circuits.append(qc)

ins = qnn_inputs
for layer_circuit in layer_circuits:
    new_ins = []
    print(f"Encoding qnn inputs and running layer")
    for i, input_circuit in enumerate(vecs_to_circs(ins)):
        print(f"Running input {i}")
        #HACK !!!
        qasm_str = input_circuit.qasm()
        qasm_str += "\n//circ"
        qubit_n = int(np.log2(qnn_inputs.shape[-1]))
        qasm_str = re.sub("qreg q\[[0-9]*];", f"qreg q[{qubit_n}];", qasm_str)
        #re.sub("creg c\[[0-9]*];", f"creg c[{int(np.sqrt(qnn_inputs.shape[-1]))}]")
        qasm_str += f"\ncreg c[{qubit_n}];\n"
        qasm_str += ''.join(layer_circuit.qasm().splitlines(keepends=True)[3:])

        for i in range(qubit_n):
            qasm_str += f"\nmeasure q[{i}] -> c[{i}];"
        
        print(qasm_str)

        t_qc = QuantumCircuit.from_qasm_str(qasm_str)
        print(t_qc)
        counts_arr = np.zeros(2**qubit_n,)
        
        for jobn in range(JOBS):
            job = qiskit.execute(t_qc, run_backend, shots=8192)#100*1024)
            result = job.result()
            counts = result.get_counts()

            print(f"{jobn=}")
            
            for key, val in counts.items():
                counts_arr[int(key, 2)] += val
            
            print(f"Updated vec:")
            print(f"{counts_arr/counts_arr.sum()}")

        print(counts)
        print(counts_arr)
        counts_arr /= counts_arr.sum()
        print(counts_arr)
        new_ins.append(counts_arr)
        
    ins = np.array(new_ins)

print(ins)

# # Execute the circuit and show the result.
# job = execute(qc, backend)
# result = job.result()
# print(result.get_counts())