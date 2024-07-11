from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
from collections import defaultdict
import json, csv

# Constantes para el análisis de métricas
m_TNoP = 'm.TNo-P'
m_NoH = 'm.NoH'
m_NoOtherSG = 'm.NoOtherSG'
m_TNoSQG = 'm.TNoSQG'
m_NoGates = 'm.NoGates'

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

# Definición del diccionario de métricas
def define_metrics(circuit):
    metrics = {
        # Circuit Size
        'm.Width': circuit.num_qubits, # Number of qubits in the circuit
        'm.Depth': circuit.depth(), # Return circuit depth (i.e., length of critical path)
        # Circuit Density
        'm.MaxDens': 0, # Maximum number of operations applied to the circuit qubits in parallel
        'm.AvgDens': 0.0, # Average of the number of operations applied to the circuit qubits in parallel
        # Single Qubit Gates
        'm.NoP-X': 0, # Number of Pauli-X (NOT) gates
        'm.NoP-Y': 0, # Number of Pauli-Y gates
        'm.NoP-Z': 0, # Number of Pauli-Z gates
        m_TNoP: 0, # Total number of Pauli gates in the circuit (calculated as the addition of the previous three)
        m_NoH: 0, # Number of Hadamard gates
        'm.%SpposQ': 0, # Ratio of qubits with a Hadamard gate as initial gate (qubits in superposition state)
        m_NoOtherSG: 0, # Number of other single-qubit gates in the circuit
        m_TNoSQG: 0, # Total number of single qubit gates
        'm.TNoCSQG': 0, # Total number of controlled single-qubit gates
        # Multiple Qubit Gates
        'm.NoSWAP': 0, # Number of SWAP gates
        'm.NoCNOT': 0, # Number of Controlled NOT (CNOT) gates
        'm.%QInCNOT': 0.0, # Ratio of qubits affected by CNOT gates (both the controlled and the target qubit in a CNOT will be considered as affected for the calculation of this metric)
        'm.AvgCNOT': 0.0, # Average number of CNOT gates targeting any qubit of a circuit
        'm.MaxCNOT': 0, #	Maximum number of CNOT gates targeting any qubit of a circuit
        'm.NoToff': 0, # Number of Toffoli gates
        'm.%QInToff': 0.0, # Ratio of qubits affected by Toffoli gates (the controlled qubit and the target qubits will be taken into account as affected for the calculation)
        'm.AvgToff': 0.0, # Average number of Toffoli gates targeting any qubit of a circuit
        'm.MaxToff': 0, # Maximum number of Toffoli gates targeting any qubit of a circuit
        # All Gates
        m_NoGates: 0, # Total number of gates in the circuit
        'm.NoCGates': 0, # Total number of controlled gates in the circuit
        'm.%SGates': 0.0, # Ratio single vs total gates
        # Oracles
        #'m.NoOr': 0,
        # Measurement Gates
        'm.NoM': 0, # Number of measurement gates in the circuit
        'm.%QM': 0.0 # Ratio of qubits measured
    }

    return metrics

# Conteo de instrucciones por qubit
def get_qubit_instruction_count(qargs, qubit_instructions, qubit_in_superposition_state, gate_name):
    for qubit in qargs:
        qubit_instructions[qubit._index] += 1
        if qubit_in_superposition_state[qubit._index] is False: # Si es la primera instrucción del qubit
            if gate_name == 'h': # Si la primera instrucción es una puerta h
                qubit_in_superposition_state[qubit._index] = 1
            else:
                qubit_in_superposition_state[qubit._index] = 0

# Conteo de puertas simples específicas
def get_sqg_instructions_count(gate_name, metrics):
    # Conteo de puertas específicas
    if gate_name == 'x': # Pauli-X
        metrics['m.NoP-X'] += 1
        metrics[m_TNoP] += 1
    elif gate_name == 'y': # Pauli-Y
        metrics['m.NoP-Y'] += 1
        metrics[m_TNoP] += 1
    elif gate_name == 'z': # Pauli-Z
        metrics['m.NoP-Z'] += 1
        metrics[m_TNoP] += 1
    elif gate_name == 'h': # Hadamard
        metrics[m_NoH] += 1
    else: # Resto de puertas simples
        metrics[m_NoOtherSG] += 1

# Single Qubit Gates
def get_sqg_instructions(gate_name, current_qubit_timeline, circuit_timeline, metrics, qargs, qubits_measured):
    if gate_name == 'measure': # Medición
        # Para la puerta measure, se establece la máxima línea temporal a todos los qubits,
        # # ya que la medición se utiliza un momento temporal para cada qubit
        set_maximum_time(circuit.qubits, current_qubit_timeline, circuit_timeline)
        # Contabilizar puerta y qubits medidos
        metrics[m_NoOtherSG] += 1
        metrics['m.NoM'] += 1
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
        get_sqg_instructions_count(gate_name, metrics)

# Conteo de puertas simples específicas
def get_sqg_instructions_count(gate_name, metrics, qargs, qubits_in_toffoli, qubits_in_toffoli_count, qubits_in_cnot, qubits_in_cnot_count):
    if gate_name == 'ccx': # Toffoli
        metrics['m.NoToff'] += 1
        for qubit in qargs:
            qubits_in_toffoli.add(qubit._index)
            qubits_in_toffoli_count[qubit._index] += 1
    else: # Puertas simples controladas
        metrics['m.TNoCSQG'] += 1
        if gate_name == 'cx': # CNOT
            metrics['m.NoCNOT'] += 1
            for qubit in qargs:
                qubits_in_cnot.add(qubit._index)
                qubits_in_cnot_count[qubit._index] += 1

# Multiple Qubits Gates
def get_mqg_instructions(gate_name, metrics, qargs, qubits_in_toffoli, qubits_in_toffoli_count, qubits_in_cnot, qubits_in_cnot_count):
    if 'c' in gate_name: # Puertas controladas
        controlled_gate = gate_name.lstrip('c')
        metrics['m.NoCGates'] += 1

        # Si se quiere almacenar un conteo individual de las puertas controladas
        if 'Controlled Gates Count' in metrics:
            metrics['Controlled Gates Count'][controlled_gate] += 1

        # Conteo de puertas específicas
        get_sqg_instructions_count(gate_name, metrics, qargs, qubits_in_toffoli, qubits_in_toffoli_count, qubits_in_cnot, qubits_in_cnot_count)
    else: # Resto de puertas múltiples
        if gate_name == 'swap': # SWAP
            metrics['m.NoSWAP'] += 1

# Cálculo de las métricas de densidad y superposición
def calculate_density_and_superposition_metrics(metrics, circuit_timeline, qubit_in_superposition_state):
    # Obtener las densidades del circuito
    metrics['m.MaxDens'] = max(circuit_timeline.values())
    metrics['m.AvgDens'] = round(sum(circuit_timeline.values()) / len(circuit_timeline.keys()), 3)

    # Calcular %SpposQ = No Qubits con la primera puerta H / Total Qubits
    # # Ponemos los qubits a 0 si hay alguno en False (es decir, que no se le ha aplicado ninguna instrucción)
    qubit_in_superposition_state[:] = [0 if qubit is False else qubit for qubit in qubit_in_superposition_state]
    metrics['m.%SpposQ'] = round(sum(qubit_in_superposition_state) / circuit.num_qubits, 3)

# Cálculo del resto de las métricas
def calculate_the_rest_metrics(metrics, qubits_in_cnot, qubits_in_cnot_count, qubits_in_toffoli, qubits_in_toffoli_count, qubits_measured):
    # Calcular TNoSQG = TNo-P + NoH + NoOtherSG
    metrics[m_TNoSQG] = metrics[m_TNoP] + metrics[m_NoH] + metrics[m_NoOtherSG]

    # Calcular %QInCNOT = Nº qubits afectados / Total Qubits
    metrics['m.%QInCNOT'] = round(len(qubits_in_cnot) / circuit.num_qubits, 3)

    # Calcular AvgCNOT = Sum(CNOT en cada qubit) / Total Qubits
    metrics['m.AvgCNOT'] = round(sum(qubits_in_cnot_count) / circuit.num_qubits, 3)

    # Calcular MaxCNOT
    metrics['m.MaxCNOT'] = max(qubits_in_cnot_count)

    # Calcular %QInToff = Nº qubits afectados / Total Qubits
    metrics['m.%QInToff'] = round(len(qubits_in_toffoli) / circuit.num_qubits, 3)

    # Calcular AvgToff = Sum(Toffoli en cada qubit) / Total Qubits
    metrics['m.AvgToff'] = round(sum(qubits_in_toffoli_count) / circuit.num_qubits, 3)

    # Calcular MaxToff
    metrics['m.MaxToff'] = max(qubits_in_toffoli_count)

    # Calcular %SGates = Total de puertas simples (TNoSQG) / Total de puertas (NoGates)
    metrics['m.%SGates'] = round(metrics[m_TNoSQG] / metrics[m_NoGates], 3)

    # Calcular %QM = Nº qubits medidos / Total Qubits
    metrics['m.%QM'] = round(len(qubits_measured) / circuit.num_qubits, 3)

def analyze_circuit(circuit):
    # Definición del diccionario de métricas
    metrics = define_metrics(circuit)
    
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
        get_qubit_instruction_count(qargs, qubit_instructions, qubit_in_superposition_state, gate_name)

        # Añadir la puerta al conteo de puertas
        metrics[m_NoGates] += 1

        # Si se quiere almacenar un conteo específico de las puertas
        if 'Gate Count' in metrics:
            metrics['Gate Count'][gate_name] += 1

        # Single Qubit Gates
        if len(qargs) == 1:
            # Realizar el conteo de las instrucciones con puertas simples
            get_sqg_instructions(gate_name, current_qubit_timeline, circuit_timeline, metrics, qargs, qubits_measured)
        # Multiple Qubit Gates
        else:
            # Establecer la máxima línea temporal a los qubits implicados
            set_maximum_time(qargs, current_qubit_timeline, circuit_timeline)
            # Realizar el conteo de las instrucciones con puertas múltiples
            get_mqg_instructions(gate_name, metrics, qargs, qubits_in_toffoli, qubits_in_toffoli_count, qubits_in_cnot, qubits_in_cnot_count)

    # Cálculo de las métricas de densidad y superposición
    calculate_density_and_superposition_metrics(metrics, circuit_timeline, qubit_in_superposition_state)

    # Cálculo del resto de las métricas
    calculate_the_rest_metrics(metrics, qubits_in_cnot, qubits_in_cnot_count, qubits_in_toffoli, qubits_in_toffoli_count, qubits_measured)

    return metrics

def get_metrics_csv(results, filename):
    headers = results[0].keys()

    with open(filename, mode='w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=headers, delimiter=";")
        csv_writer.writeheader()
        csv_writer.writerows(results)

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
    results = []
    circuit_results = analyze_circuit(circuit)
    results.append(circuit_results)

    # Exportar los resultados a JSON
    results_json = json.dumps(results, indent=4)
    print(f"\n{results_json}")

    # Obtener el csv con las métricas
    csv_filename = "../datasets/file_metrics.csv"
    get_metrics_csv(results, csv_filename)

    # Dibujar el circuito
    draw_circuit(circuit)