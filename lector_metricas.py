from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import json

def analyze_circuit(circuit):
    metrics = {
        'Circuit Size': {
            'Width': circuit.width(), # Number of qubits and clbits (classical qubits)
            'Qubits Used': circuit.num_qubits, # Number of qubits in the circuit
            'Depth': circuit.depth(), # Return circuit depth (i.e., length of critical path)
            'Max Operations in a Qubit': 0 # Maximum number of operations in a qubit
        },
        'Circuit Density': {
            'MaxDens': 0, # Maximum number of simultaneous operations
            'AvgDens': 0.0, # Average number of simultaneous operations
        },
        'Single Qubits Gates': {
            'NoP-X': 0, # Number of Pauli-X gates
            'NoP-Y': 0, # Number of Pauli-Y gates
            'NoP-Z': 0, # Number of Pauli-Z gates
            'TNo-P': 0, # Total number of Pauli gates
            'NoH': 0, # Number of Hadamard gates
            '%SpposQ': 0, # Percentage of qubits in initial superposition state
            'NoOtherSG': 0, # Number of other single qubit gates
            'TNoSQG': 0, # Total number of single qubit gates
            'TNoCSQG': 0, # Total number of controlled single qubit gates
        },
        'Controlled Gates Count': defaultdict(int),
        'Multiple Qubit Gates': {
            'NoSWAP': 0, # Total number of SWAP gates
            'NoCNOT': 0, # Total number of CNOT gates
            '%QInCNOT': 0.0, # Percentage of qubits affected by a CNOT
            'AvgCNOT': 0.0, # Average number of qubits affected by a CNOT
            'MaxCNOT': 0, # Maximum number of CNOTs targeting any qubit
            'NoToff': 0, # Total number of Toffoli gates
            'AvgToff': 0.0, # 
            '%QInToff': 0.0, # Percentage of qubits affected by Toffoli gates
            'MaxToff': 0 # Maximum number of Toffoli gates
        },
        'All Gates': {
            'NoGates': 0, # Total number gates
            'NoCGates': 0, # Total number of controlled gates
            '%SGates': 0.0 # Percentage of single qubit gates in the circuit
        },
        'Number of Oracles': 0,
        'Measurement Gates': {
            'NoM': 0, # Number of measurement gates in the circuit
            '%QM': 0.0 # Percentage of qubits measured in the circuit
        },
        'Entanglement Metrics': {
            'Entangling Gates Count': 0,
            'Qubit Pairs Entangled': Counter()
        },
        'Gate Count': defaultdict(int)
    }

    # Calculating circuit density
    max_dens = max(circuit.count_ops().values()) / circuit.num_qubits
    avg_dens = circuit.size() / (circuit.depth() * circuit.num_qubits)
    metrics['Circuit Density']['MaxDens'] = max_dens
    metrics['Circuit Density']['AvgDens'] = avg_dens

    # Conteo de instrucciones por qubit
    qubit_instructions = [0] * circuit.num_qubits
    qubits_in_cnot = set()
    qubits_in_toffoli = set()
    qubits_measured = set()

    # Analizar cada puerta en el circuito
    for instruction, qargs, _ in circuit.data:
        print("\n")
        print(instruction) # Instrucción
        # Conteo de instrucciones por qubit
        for qubit in qargs:
            qubit_instructions[qubit._index] += 1
        print(qargs) # Qubits
        print(_) # Classical bits

        # Obtener el nombre de la puerta
        gate_name = instruction.name

        # Añadir la puerta al conteo de puertas
        metrics['Gate Count'][gate_name] += 1

        # Contar las puertas específicas
        # Single Qubit Gates
        if gate_name == 'h':
            metrics['Single Qubits Gates']['NoH'] += 1
        elif gate_name == 'x':
            metrics['Single Qubits Gates']['NoP-X'] += 1
            metrics['Single Qubits Gates']['TNo-P'] += 1
        elif gate_name == 'y':
            metrics['Single Qubits Gates']['NoP-Y'] += 1
            metrics['Single Qubits Gates']['TNo-P'] += 1
        elif gate_name == 'z':
            metrics['Single Qubits Gates']['NoP-Z'] += 1
            metrics['Single Qubits Gates']['TNo-P'] += 1
        elif gate_name in ['s', 't']: # s y t son el resto de puertas simples
            metrics['Single Qubits Gates']['NoOtherSG'] += 1
        # Multiple Qubit Gates
        elif gate_name == 'swap':
            metrics['Multiple Qubit Gates']['NoSWAP'] += 1
        # Measure
        elif gate_name == 'measure':
            metrics['Measurement Gates']['NoM'] += 1
            for qubit in qargs:
                qubits_measured.add(qubit)

        # Contar las puertas controladas
        if 'c' in gate_name and gate_name != 'measure':
            controlled_gate = gate_name.lstrip('c')
            metrics['Controlled Gates Count'][controlled_gate] += 1
            metrics['All Gates']['NoCGates'] += 1

            if gate_name in ['x', 'y', 'z', 'h', 's', 't']:
                metrics['Single Qubits Gates']['TNoCSQG'] += 1

            if gate_name == 'x': # cx o CNOT
                metrics['Multiple Qubit Gates']['NoCNOT'] += 1
                for qubit in qargs:
                    qubits_in_cnot.add(qubit)
            elif gate_name == 'cx': # ccx o Toffoli
                metrics['Multiple Qubit Gates']['NoToff'] += 1
                for qubit in qargs:
                    qubits_in_toffoli.add(qubit)

        # Contar y almacenar las puertas entrelazadas
        if gate_name in ['cx', 'cz', 'cy', 'swap']:
            metrics['Entanglement Metrics']['Entangling Gates Count'] += 1
            qubit_pair = str(tuple(sorted([q._index for q in qargs])))  # Sort to avoid different orderings
            print(qubit_pair)
            metrics['Entanglement Metrics']['Qubit Pairs Entangled'][qubit_pair] += 1

    # Obtener el nº máximo de operaciones entre todos los qubits
    print(qubit_instructions)
    metrics['Circuit Size']['Max Operations in a Qubit'] = max(qubit_instructions)

    # Calcular %SpposQ
    metrics['Single Qubits Gates']['%SpposQ'] = metrics['Single Qubits Gates']['NoH'] / circuit.num_qubits

    # Calcular TNoSQG = TNo-P + NoH + NoOtherSG
    metrics['Single Qubits Gates']['TNoSQG'] = metrics['Single Qubits Gates']['TNo-P'] + metrics['Single Qubits Gates']['NoH'] + metrics['Single Qubits Gates']['NoOtherSG']

    # Calcular %QInCNOT = Nº qubits afectados / Total Qubits
    metrics['Multiple Qubit Gates']['%QInCNOT'] = len(qubits_in_cnot) / circuit.num_qubits

    # Calcular %QInToff = Nº qubits afectados / Total Qubits
    metrics['Multiple Qubit Gates']['%QInToff'] = len(qubits_in_toffoli) / circuit.num_qubits

    # Calcular %QM = Nº qubits medidos / Total Qubits
    metrics['Measurement Gates']['%QM'] = len(qubits_measured) / circuit.num_qubits

    # Estimar el número de oráculos en el circuito
    metrics['Number of Oracles'] = estimate_oracles(circuit)

    return metrics

def estimate_oracles(circuit):
    # TODO
    return 0  # Asumir que no hay oráculos por ahora

# Ejemplo de circuito
circuit = QuantumCircuit(3)
circuit.h(0)
circuit.x(0)
circuit.y(1)
circuit.z(2)
circuit.swap(0, 1)
circuit.cx(0, 1)
circuit.cx(0, 1)
circuit.cy(1, 2)
circuit.cz(0, 2)
circuit.measure_all()

print("Listado de Qubits:", circuit.qubits)
print("Conteo de operaciones por tipo: ", circuit.count_ops())

# Analizar el circuito
results = analyze_circuit(circuit)

# Exportar los resultados a JSON
results_json = json.dumps(results, indent=4)
print(results_json)

# Dibujar el circuito
# circuit_drawer(circuit=circuit, output="mpl")
# plt.show()