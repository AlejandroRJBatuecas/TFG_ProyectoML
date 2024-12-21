"""
Fichero de definición del diccionario de las métricas de circuitos cúanticos
"""

# Constantes para el análisis de métricas
descriptive_name = 'Descriptive name'
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
        'm.Width': {
            descriptive_name: 'Circuit Width',
            'Description': 'Number of qubits in the circuit',
            'Value': 0
        },
        'm.Depth': {
            descriptive_name: 'Circuit Depth',
            'Description': 'Circuit depth (i.e., length of critical path)',
            'Value': 0
        }
    },
    'Circuit Density': {
        'm.MaxDens': {
            descriptive_name: 'Max. Density',
            'Description': 'Maximum number of operations applied to the circuit qubits in parallel',
            'Value': 0
        },
        'm.AvgDens': {
            descriptive_name: 'Avg. Density',
            'Description': 'Average of the number of operations applied to the circuit qubits in parallel',
            'Value': 0.0
        }
    },
    'Single Qubit Gates': {
        'm.NoP-X': {
            descriptive_name: '# X gates',
            'Description': 'Number of Pauli-X (NOT) gates',
            'Value': 0
        },
        'm.NoP-Y': {
            descriptive_name: '# Y gates',
            'Description': 'Number of Pauli-Y gates',
            'Value': 0
        },
        'm.NoP-Z': {
            descriptive_name: '# Z gates',
            'Description': 'Number of Pauli-Z gates',
            'Value': 0
        },
        m_TNoP: {
            descriptive_name: '# Pauli gates',
            'Description': 'Total number of Pauli gates in the circuit (calculated as the addition of the previous three)',
            'Value': 0
        },
        m_NoH: {
            descriptive_name: '# H gates',
            'Description': 'Number of Hadamard gates',
            'Value': 0
        }, 
        'm.%SpposQ': {
            descriptive_name: '% Qubits in Superposition',
            'Description': 'Ratio of qubits with a Hadamard gate as initial gate (qubits in superposition state)',
            'Value': 0
        },
        m_NoOtherSG: {
            descriptive_name: '# Other SQ gates',
            'Description': 'Number of other single-qubit gates in the circuit',
            'Value': 0
        },
        m_TNoSQG: {
            descriptive_name: '# SQ gates',
            'Description': 'Total number of single qubit gates',
            'Value': 0
        },
        'm.TNoCSQG': {
            descriptive_name: '# Controlled SQ gates',
            'Description': 'Total number of controlled single-qubit gates',
            'Value': 0
        }
    },
    'Multiple Qubit Gates': {
        'm.NoSWAP': {
            descriptive_name: '# SWAP gates',
            'Description': 'Number of SWAP gates',
            'Value': 0
        },
        'm.NoCNOT': {
            descriptive_name: '# CNOT gates',
            'Description': 'Number of Controlled NOT (CNOT) gates',
            'Value': 0
        },
        'm.%QInCNOT': {
            descriptive_name: '% Qubits in CNOT',
            'Description': 'Ratio of qubits affected by CNOT gates (both the controlled and the target qubit in a CNOT will be considered as affected for the calculation of this metric)',
            'Value': 0.0
        },
        'm.AvgCNOT': {
            descriptive_name: 'Avg. CNOT gates',
            'Description': 'Average number of CNOT gates targeting any qubit of a circuit',
            'Value': 0.0
        },
        'm.MaxCNOT': {
            descriptive_name: 'Max. CNOT gates',
            'Description': 'Maximum number of CNOT gates targeting any qubit of a circuit',
            'Value': 0
        },
        'm.NoToff': {
            descriptive_name: '# Toffoli gates',
            'Description': 'Number of Toffoli gates',
            'Value': 0
        }, 
        'm.%QInToff': {
            descriptive_name: '% Qubits in Toffoli',
            'Description': 'Ratio of qubits affected by Toffoli gates (the controlled qubit and the target qubits will be taken into account as affected for the calculation)',
            'Value': 0.0
        },
        'm.AvgToff': {
            descriptive_name: 'Avg. Toffoli gates',
            'Description': 'Average number of Toffoli gates targeting any qubit of a circuit',
            'Value': 0.0
        },
        'm.MaxToff': {
            descriptive_name: 'Max. Toffoli gates',
            'Description': 'Maximum number of Toffoli gates targeting any qubit of a circuit',
            'Value': 0
        }
    },
    'All Gates': {
        m_NoGates: {
            descriptive_name: '# gates',
            'Description': 'Total number of gates in the circuit',
            'Value': 0
        },
        'm.NoCGates': {
            descriptive_name: '# Controlled gates',
            'Description': 'Total number of controlled gates in the circuit',
            'Value': 0
        },
        'm.%SGates': {
            descriptive_name: '% Single gates',
            'Description': 'Ratio single vs total gates',
            'Value': 0.0
        }
    },
    #'Oracles': { 'm.NoOr': 0 }
    'Measurement Gates': {
        'm.NoM': {
            descriptive_name: '# Measurement gates',
            'Description': 'Number of measurement gates in the circuit',
            'Value': 0
        },
        'm.%QM': {
            descriptive_name: '% Qubits measured',
            'Description': 'Ratio of qubits measured',
            'Value': 0.0
        }
    }
}
""" Diccionario que almacena las métricas de circuitos cúanticos """