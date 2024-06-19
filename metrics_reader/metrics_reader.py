from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
from collections import defaultdict
import json

def draw_circuit(circuit):
    # Dibujar el circuito
    circuit_drawer(circuit=circuit, output="mpl", cregbundle=False)
    plt.show()

def set_maximum_time(qargs, current_qubit_timeline, circuit_timeline):
    # Añadir puerta al momento temporal del circuito y a las líneas temporales de los qubits
    max_timeline = -1
    # Para los qubits involucrados, obtenemos el momento temporal de cada uno y nos quedamos con el máximo
    for qubit in qargs:
        if current_qubit_timeline[qubit._index] > max_timeline:
            max_timeline = current_qubit_timeline[qubit._index]
    # Inicializar el momento temporal si no lo está
    if max_timeline not in circuit_timeline:
        circuit_timeline[max_timeline] = 0
    # Añadir puerta al momento temporal del circuito
    circuit_timeline[max_timeline] += 1
    # Calculamos el próximo momento
    max_timeline +=1
    # Asignar a cada qubit el máximo momento temporal
    for qubit in qargs:
        current_qubit_timeline[qubit._index] = max_timeline

def analyze_circuit(circuit):
    metrics = {
        'Circuit Size': {
            'Width': circuit.num_qubits, # Number of qubits in the circuit
            'Total bits': circuit.width(), # Total number of bits (qubits and clbits -classical bits-)
            'Depth': circuit.depth(), # Return circuit depth (i.e., length of critical path)
            'Max Operations in a Qubit': 0 # Maximum number of operations applied to a qubit in the circuit
        },
        'Circuit Density': {
            'MaxDens': 0, # Maximum number of operations applied to the circuit qubits in parallel
            'AvgDens': 0.0, # Average of the number of operations applied to the circuit qubits in parallel
        },
        'Single Qubit Gates': {
            'NoP-X': 0, # Number of Pauli-X (NOT) gates
            'NoP-Y': 0, # Number of Pauli-Y gates
            'NoP-Z': 0, # Number of Pauli-Z gates
            'TNo-P': 0, # Total number of Pauli gates in the circuit (calculated as the addition of the previous three)
            'NoH': 0, # Number of Hadamard gates
            '%SpposQ': 0, # Ratio of qubits with a Hadamard gate as initial gate (qubits in superposition state)
            'NoOtherSG': 0, # Number of other single-qubit gates in the circuit
            'TNoSQG': 0, # Total number of single qubit gates
            'TNoCSQG': 0, # Total number of controlled single-qubit gates
        },
        'Multiple Qubit Gates': {
            'NoSWAP': 0, # Number of SWAP gates
            'NoCNOT': 0, # Number of Controlled NOT (CNOT) gates
            '%QInCNOT': 0.0, # Ratio of qubits affected by CNOT gates (both the controlled and the target qubit in a CNOT will be considered as affected for the calculation of this metric)
            'AvgCNOT': 0.0, # Average number of CNOT gates targeting any qubit of a circuit
            'MaxCNOT': 0, #	Maximum number of CNOT gates targeting any qubit of a circuit
            'NoToff': 0, # Number of Toffoli gates
            'AvgToff': 0.0, # Average number of Toffoli gates targeting any qubit of a circuit
            '%QInToff': 0.0, # Ratio of qubits affected by Toffoli gates (the controlled qubit and the target qubits will be taken into account as affected for the calculation)
            'MaxToff': 0 # Maximum number of Toffoli gates targeting any qubit of a circuit
        },
        #'Controlled Gates Count': defaultdict(int),
        'Number of Oracles': 0,
        'All Gates': {
            'NoGates': 0, #	Total number of gates in the circuit
            'NoCGates': 0, # Total number of controlled gates in the circuit
            '%SGates': 0.0 # Ratio single vs total gates
        },
        #'Gate Count': defaultdict(int),
        'Measurement Gates': {
            'NoM': 0, # Number of measurement gates in the circuit
            '%QM': 0.0 # Ratio of qubits measured
        }
    }
    
    # Instrucciones de cada qubit por tiempo
    current_qubit_timeline = [0] * circuit.num_qubits
    circuit_timeline = {}
    
    # Conteo de instrucciones por qubit
    qubit_instructions = [0] * circuit.num_qubits
    qubit_in_superposition_state = [False] * circuit.num_qubits
    qubits_in_cnot_count = [0] * circuit.num_qubits
    qubits_in_cnot = set()
    qubits_in_toffoli_count = [0] * circuit.num_qubits
    qubits_in_toffoli = set()
    qubits_measured = set()

    # Analizar cada puerta en el circuito
    for instruction, qargs, cargs in circuit.data:
        print(f"\n{instruction}") # Instrucción
        print(qargs) # Qubits
        print(cargs) # Classical bits

        # Obtener el nombre de la puerta
        gate_name = instruction.name

        # Conteo de instrucciones por qubit
        for qubit in qargs:
            qubit_instructions[qubit._index] += 1
            if qubit_in_superposition_state[qubit._index] is False: # Si es la primera instrucción del qubit
                if gate_name == 'h': # Si la primera instrucción es una puerta h
                    qubit_in_superposition_state[qubit._index] = 1
                else:
                    qubit_in_superposition_state[qubit._index] = 0

        # Añadir la puerta al conteo de puertas
        metrics['All Gates']['NoGates'] += 1

        if 'Gate Count' in metrics:
            metrics['Gate Count'][gate_name] += 1

        # Single Qubit Gates
        if len(qargs) == 1:
            if gate_name == 'measure': # Medición
                # Para la puerta measure, se establece la máxima línea temporal a todos los qubits,
                # # ya que la medición se utiliza un momento temporal para cada qubit
                set_maximum_time(circuit.qubits, current_qubit_timeline, circuit_timeline)
                # Contabilizar puerta y qubits medidos
                metrics['Single Qubit Gates']['NoOtherSG'] += 1
                metrics['Measurement Gates']['NoM'] += 1
                for qubit in qargs:
                    qubits_measured.add(qubit._index)
            else:
                # Añadir puerta al momento temporal del circuito y a la línea temporal del qubit
                qubit = qargs[0]
                # Inicializar el momento temporal si no lo está
                if current_qubit_timeline[qubit._index] not in circuit_timeline:
                    circuit_timeline[current_qubit_timeline[qubit._index]] = 0
                # Contabilizar puerta
                circuit_timeline[current_qubit_timeline[qubit._index]] += 1
                current_qubit_timeline[qubit._index] += 1

                # Conteo de puertas específicas
                if gate_name == 'x': # Pauli-X
                    metrics['Single Qubit Gates']['NoP-X'] += 1
                    metrics['Single Qubit Gates']['TNo-P'] += 1
                elif gate_name == 'y': # Pauli-Y
                    metrics['Single Qubit Gates']['NoP-Y'] += 1
                    metrics['Single Qubit Gates']['TNo-P'] += 1
                elif gate_name == 'z': # Pauli-Z
                    metrics['Single Qubit Gates']['NoP-Z'] += 1
                    metrics['Single Qubit Gates']['TNo-P'] += 1
                elif gate_name == 'h': # Hadamard
                    metrics['Single Qubit Gates']['NoH'] += 1
                else: # Resto de puertas simples
                    metrics['Single Qubit Gates']['NoOtherSG'] += 1
        # Multiple Qubit Gates
        else:
            # Establecer la máxima línea temporal a los qubits implicados
            set_maximum_time(qargs, current_qubit_timeline, circuit_timeline)

            if 'c' in gate_name: # Puertas controladas
                controlled_gate = gate_name.lstrip('c')
                metrics['All Gates']['NoCGates'] += 1
                if 'Controlled Gates Count' in metrics:
                    metrics['Controlled Gates Count'][controlled_gate] += 1

                # Conteo de puertas específicas
                if gate_name == 'ccx': # Toffoli
                    metrics['Multiple Qubit Gates']['NoToff'] += 1
                    for qubit in qargs:
                        qubits_in_toffoli.add(qubit._index)
                        qubits_in_toffoli_count[qubit._index] += 1
                else: # Puertas simples controladas
                    metrics['Single Qubit Gates']['TNoCSQG'] += 1
                    if gate_name == 'cx': # CNOT
                        metrics['Multiple Qubit Gates']['NoCNOT'] += 1
                        for qubit in qargs:
                            qubits_in_cnot.add(qubit._index)
                            qubits_in_cnot_count[qubit._index] += 1
            else: # Resto de puertas múltiples
                if gate_name == 'swap': # SWAP
                    metrics['Multiple Qubit Gates']['NoSWAP'] += 1

    # Obtener el nº máximo de operaciones entre todos los qubits
    metrics['Circuit Size']['Max Operations in a Qubit'] = max(qubit_instructions)

    # Obtener las densidades del circuito
    metrics['Circuit Density']['MaxDens'] = max(circuit_timeline.values())
    metrics['Circuit Density']['AvgDens'] = sum(circuit_timeline.values()) / len(circuit_timeline.keys())

    # Calcular %SpposQ = No Qubits con la primera puerta H / Total Qubits
    # # Ponemos los qubits a 0 si hay alguno en False (es decir, que no se le ha aplicado ninguna instrucción)
    qubit_in_superposition_state[:] = [0 if qubit is False else qubit for qubit in qubit_in_superposition_state]
    metrics['Single Qubit Gates']['%SpposQ'] = sum(qubit_in_superposition_state) / circuit.num_qubits

    # Calcular TNoSQG = TNo-P + NoH + NoOtherSG
    metrics['Single Qubit Gates']['TNoSQG'] = metrics['Single Qubit Gates']['TNo-P'] + metrics['Single Qubit Gates']['NoH'] + metrics['Single Qubit Gates']['NoOtherSG']

    # Calcular %QInCNOT = Nº qubits afectados / Total Qubits
    metrics['Multiple Qubit Gates']['%QInCNOT'] = len(qubits_in_cnot) / circuit.num_qubits

    # Calcular AvgCNOT = Sum(CNOT en cada qubit) / Total Qubits
    metrics['Multiple Qubit Gates']['AvgCNOT'] = sum(qubits_in_cnot_count) / circuit.num_qubits

    # Calcular MaxCNOT
    metrics['Multiple Qubit Gates']['MaxCNOT'] = max(qubits_in_cnot_count)

    # Calcular %QInToff = Nº qubits afectados / Total Qubits
    metrics['Multiple Qubit Gates']['%QInToff'] = len(qubits_in_toffoli) / circuit.num_qubits

    # Calcular AvgToff = Sum(Toffoli en cada qubit) / Total Qubits
    metrics['Multiple Qubit Gates']['AvgToff'] = sum(qubits_in_toffoli_count) / circuit.num_qubits

    # Calcular MaxToff
    metrics['Multiple Qubit Gates']['MaxToff'] = max(qubits_in_toffoli_count)

    # Calcular %SGates = TNoSQG / NoGates
    metrics['All Gates']['%SGates'] = metrics['Single Qubit Gates']['TNoSQG'] / metrics['All Gates']['NoGates']

    # Calcular %QM = Nº qubits medidos / Total Qubits
    metrics['Measurement Gates']['%QM'] = len(qubits_measured) / circuit.num_qubits

    # Estimar el número de oráculos en el circuito
    metrics['Number of Oracles'] = estimate_oracles(circuit)

    return metrics

def estimate_oracles(circuit):
    # TODO
    return 0  # Asumir que no hay oráculos por ahora

# Código de prueba
if __name__ == "__main__":
    # Ejemplo de circuito
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.x(0)
    circuit.y(1)
    circuit.z(2)
    circuit.swap(0, 1)
    circuit.x(2)
    circuit.cx(0, 1)
    circuit.cx(0, 1)
    circuit.cy(1, 2)
    circuit.cz(0, 2)
    circuit.ccx(0, 1, 2)
    circuit.x(0)
    circuit.y(0)
    circuit.z(1)
    circuit.measure_all()

    # Analizar el circuito
    results = analyze_circuit(circuit)

    # Exportar los resultados a JSON
    results_json = json.dumps(results, indent=4)
    print(results_json)

    # Dibujar el circuito
    draw_circuit(circuit)