"""
Fichero de definición del diccionario de las métricas de circuitos cúanticos
"""

# Constantes para el análisis de métricas
single_qubit_gates = 'Single Qubit Gates'
""" Categoría que agrupa las métricas de puertas simples """
multiple_qubit_gates = 'Multiple Qubit Gates'
""" Categoría que agrupa las métricas de puertas múltiples """
all_gates = 'All Gates'
""" Categoría que agrupa las métricas de todas las puertas """
m_TNoP = 'm.TNo-P'
""" Número total de puertas Pauli-X """
m_NoH = 'm.NoH'
""" Número de puertas Hadamard """
m_NoOtherSG = 'm.NoOtherSG'
""" Número de otras puertas simples """
m_TNoSQG = 'm.TNoSQG'
""" Número total de puertas simples """
m_NoGates = 'm.NoGates'
""" Número total de puertas """

circuit_metrics = {
    'Circuit Size': {
        'm.Width': 0, # Number of qubits in the circuit
        'm.Depth': 0 # Return circuit depth (i.e., length of critical path)
    },
    'Circuit Density': {
        'm.MaxDens': 0, # Maximum number of operations applied to the circuit qubits in parallel
        'm.AvgDens': 0.0 # Average of the number of operations applied to the circuit qubits in parallel
    },
    'Single Qubit Gates': {
        'm.NoP-X': 0, # Number of Pauli-X (NOT) gates
        'm.NoP-Y': 0, # Number of Pauli-Y gates
        'm.NoP-Z': 0, # Number of Pauli-Z gates
        m_TNoP: 0, # Total number of Pauli gates in the circuit (calculated as the addition of the previous three)
        m_NoH: 0, # Number of Hadamard gates
        'm.%SpposQ': 0, # Ratio of qubits with a Hadamard gate as initial gate (qubits in superposition state)
        m_NoOtherSG: 0, # Number of other single-qubit gates in the circuit
        m_TNoSQG: 0, # Total number of single qubit gates
        'm.TNoCSQG': 0 # Total number of controlled single-qubit gates
    },
    'Multiple Qubit Gates': {
        'm.NoSWAP': 0, # Number of SWAP gates
        'm.NoCNOT': 0, # Number of Controlled NOT (CNOT) gates
        'm.%QInCNOT': 0.0, # Ratio of qubits affected by CNOT gates (both the controlled and the target qubit in a CNOT will be considered as affected for the calculation of this metric)
        'm.AvgCNOT': 0.0, # Average number of CNOT gates targeting any qubit of a circuit
        'm.MaxCNOT': 0, #	Maximum number of CNOT gates targeting any qubit of a circuit
        'm.NoToff': 0, # Number of Toffoli gates
        'm.%QInToff': 0.0, # Ratio of qubits affected by Toffoli gates (the controlled qubit and the target qubits will be taken into account as affected for the calculation)
        'm.AvgToff': 0.0, # Average number of Toffoli gates targeting any qubit of a circuit
        'm.MaxToff': 0 # Maximum number of Toffoli gates targeting any qubit of a circuit
    },
    'All Gates': {
        m_NoGates: 0, # Total number of gates in the circuit
        'm.NoCGates': 0, # Total number of controlled gates in the circuit
        'm.%SGates': 0.0 # Ratio single vs total gates
    },
    #'Oracles': { 'm.NoOr': 0 }
    'Measurement Gates': {
        'm.NoM': 0, # Number of measurement gates in the circuit
        'm.%QM': 0.0 # Ratio of qubits measured
    }
}
""" Diccionario que almacena las métricas de circuitos cúanticos """