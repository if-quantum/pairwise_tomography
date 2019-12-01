
# pylint: disable=missing-docstring
import unittest

import itertools
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit, execute, Aer
from qiskit.quantum_info import state_fidelity
from qiskit.tools.qi.qi import partial_trace
from qiskit.quantum_info.states import DensityMatrix

from pairwise_tomography.pairwise_state_tomography_circuits import pairwise_state_tomography_circuits
from pairwise_tomography.pairwise_fitter import PairwiseStateTomographyFitter

n_list = [2, 4]
nshots = 5000

pauli = {'I': np.eye(2),
         'X': np.array([[0, 1], [1, 0]]),
         'Y': np.array([[0, -1j], [1j, 0]]),
         'Z': np.array([[1, 0], [0, -1]])}

def pauli_expectation(rho, i, j):
    # i and j get swapped because of qiskit bit convention
    return np.real(np.trace(np.kron(pauli[j], pauli[i]) @ rho))

class TestPairwiseStateTomography(unittest.TestCase):
    def test_pairwise_tomography(self):
        for n in n_list:
            with self.subTest():
                self.tomography_random_circuit(n)

    def tomography_random_circuit(self, n):
        q = QuantumRegister(n)
        qc = QuantumCircuit(q)

        psi = ((2 * np.random.rand(2 ** n) - 1) 
               + 1j * (2 *np.random.rand(2 ** n) - 1))
        psi /= np.linalg.norm(psi)

        qc.initialize(psi, q)
        rho = DensityMatrix.from_instruction(qc).data

        circ = pairwise_state_tomography_circuits(qc, q)
        job = execute(circ, Aer.get_backend("qasm_simulator"), shots=nshots)
        fitter = PairwiseStateTomographyFitter(job.result(), circ, q)
        result = fitter.fit()
        result_exp = fitter.fit(output='expectation')

        # Compare the tomography matrices with the partial trace of 
        # the original state using fidelity
        for (k, v) in result.items():
            trace_qubits = list(range(n))
            trace_qubits.remove(k[0])
            trace_qubits.remove(k[1])
            rhok = partial_trace(rho, trace_qubits)
            try:
                self.check_density_matrix(v, rhok)
            except:
                print("Problem with density matrix:", k)
                raise
            try:
                self.check_pauli_expectaion(result_exp[k], rhok)
            except:
                print("Problem with expectation values:", k)
                raise
    
    def check_density_matrix(self, item, rho):
        fidelity = state_fidelity(item, rho)
        try:
            self.assertAlmostEqual(fidelity, 1, delta=4 / np.sqrt(nshots))
        except AssertionError:
            print(fidelity)
            raise

    def check_pauli_expectaion(self, item, rho):
        for (a, b) in itertools.product(pauli.keys(), pauli.keys()):
            if not (a == "I" and b == "I"):
                correct = pauli_expectation(rho, a, b)
                tomo = item[(a, b)]
                delta = np.sqrt(16 * (1 - correct ** 2) / nshots)
                try:
                    self.assertAlmostEqual(tomo, correct, delta=delta)
                except AssertionError:
                    print(a, b, correct, tomo)
                    raise
                
    def test_meas_qubit_specification(self):
        n = 4

        q = QuantumRegister(n)
        qc = QuantumCircuit(q)

        psi = ((2 * np.random.rand(2 ** n) - 1) 
            + 1j * (2 *np.random.rand(2 ** n) - 1))
        psi /= np.linalg.norm(psi)

        qc.initialize(psi, q)
        rho = DensityMatrix.from_instruction(qc).data

        measured_qubits = [q[0], q[2], q[3]]
        circ = pairwise_state_tomography_circuits(qc, measured_qubits)
        job = execute(circ, Aer.get_backend("qasm_simulator"), shots=nshots)
        fitter = PairwiseStateTomographyFitter(job.result(), circ, measured_qubits)
        result = fitter.fit()
        result_exp = fitter.fit(output='expectation')

        # Compare the tomography matrices with the partial trace of 
        # the original state using fidelity
        for (k, v) in result.items():
            #TODO: This method won't work if measured_qubits is not ordered in 
            # wrt the DensityMatrix object.
            trace_qubits = list(range(n))
            trace_qubits.remove(measured_qubits[k[0]].index)
            trace_qubits.remove(measured_qubits[k[1]].index)
            rhok = partial_trace(rho, trace_qubits)
            try:
                self.check_density_matrix(v, rhok)
            except:
                print("Problem with density matrix:", k)
                raise
            try:
                self.check_pauli_expectaion(result_exp[k], rhok)
            except:
                print("Problem with expectation values:", k)
                raise

    def test_multiple_registers(self):
        n = 4

        q = QuantumRegister(n / 2)
        p = QuantumRegister(n / 2)

        qc = QuantumCircuit(q, p)
        
        qc.h(q[0])
        qc.rx(np.pi/4, q[1])
        qc.cx(q[0], p[0])
        qc.cx(q[1], p[1])

        rho = DensityMatrix.from_instruction(qc).data

        measured_qubits = q#[q[0], q[1], q[2]]
        circ = pairwise_state_tomography_circuits(qc, measured_qubits)
        job = execute(circ, Aer.get_backend("qasm_simulator"), shots=nshots)
        fitter = PairwiseStateTomographyFitter(job.result(), circ, measured_qubits)
        result = fitter.fit()
        result_exp = fitter.fit(output='expectation')

        # Compare the tomography matrices with the partial trace of
        # the original state using fidelity
        for (k, v) in result.items():
            trace_qubits = list(range(n))
            trace_qubits.remove(measured_qubits[k[0]].index)
            trace_qubits.remove(measured_qubits[k[1]].index)
            rhok = partial_trace(rho, trace_qubits)
            try:
                self.check_density_matrix(v, rhok)
            except:
                print("Problem with density matrix:", k)
                raise
            try:
                self.check_pauli_expectaion(result_exp[k], rhok)
            except:
                print("Problem with expectation values:", k)
                raise

if __name__ == '__main__':
    unittest.main()
