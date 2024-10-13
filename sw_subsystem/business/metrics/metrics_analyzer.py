import base64

from io import BytesIO
from config import metrics_definition
from qiskit.visualization import circuit_drawer
from collections import defaultdict

class MetricsAnalyzer:
    def __init__(self, circuit):
        # Establecimiento del circuito y del conjunto de métricas
        self.circuit = circuit
        self.metrics = self._define_metrics()
        # Conteo de puertas específico
        self.gate_count_dict = defaultdict(int)
        # Instrucciones de cada qubit por tiempo
        self.current_qubit_timeline = [0] * self.circuit.num_qubits
        self.circuit_timeline = {}
        # Conteo de instrucciones por qubit
        self.qubit_instructions = [0] * self.circuit.num_qubits
        self.qubit_in_superposition_state = [False] * self.circuit.num_qubits
        self.qubits_in_cnot_count = [0] * self.circuit.num_qubits
        self.qubits_in_cnot = set()
        self.qubits_in_toffoli_count = [0] * self.circuit.num_qubits
        self.qubits_in_toffoli = set()
        self.qubits_measured = set()

        # Analizar el circuito
        self._analyze_circuit()

        # Obtener el circuito simplificado (sin categorías)
        self.simplified_circuit = self._get_simplified_circuit()

        # Obtener el circuito
        self.circuit_draw = self._get_circuit_draw()

    # Obtención del diccionario de métricas
    def _define_metrics(self):
        return metrics_definition.circuit_metrics

    # Dibujar el circuito
    def _get_circuit_draw(self):
        # Dibujar el circuito con Matplotlib
        circuit_draw = circuit_drawer(circuit=self.circuit, output="mpl", cregbundle=False)

        # Guardar la imagen en un objeto BytesIO
        image_stream = BytesIO()
        circuit_draw.savefig(image_stream, format='png')
        image_stream.seek(0)  # Volver al inicio del buffer

        # Codificar la imagen en base64
        image_base64 = base64.b64encode(image_stream.getvalue()).decode('utf-8')

        return image_base64

    def _set_maximum_time(self, qargs):
        # Añadir puerta al momento temporal del circuito y a las líneas temporales de los qubits
        max_timeline = -1
        # Para los qubits involucrados, obtenemos el momento temporal de cada uno y nos quedamos con el máximo
        for qubit in qargs:
            if self.current_qubit_timeline[qubit._index] > max_timeline:
                max_timeline = self.current_qubit_timeline[qubit._index]
        # Inicializar el momento temporal si no lo está
        if max_timeline not in self.circuit_timeline:
            self.circuit_timeline[max_timeline] = 0
        # Añadir puerta al momento temporal del circuito
        self.circuit_timeline[max_timeline] += 1
        # Calculamos el próximo momento
        max_timeline +=1
        # Asignar a cada qubit el máximo momento temporal
        for qubit in qargs:
            self.current_qubit_timeline[qubit._index] = max_timeline

    # Conteo de instrucciones por qubit
    def _get_qubit_instruction_count(self, qargs, gate_name):
        for qubit in qargs:
            self.qubit_instructions[qubit._index] += 1
            if self.qubit_in_superposition_state[qubit._index] is False: # Si es la primera instrucción del qubit
                if gate_name == 'h': # Si la primera instrucción es una puerta h
                    self.qubit_in_superposition_state[qubit._index] = 1
                else:
                    self.qubit_in_superposition_state[qubit._index] = 0

    # Conteo de puertas simples específicas
    def _get_sqg_instructions_count(self, gate_name):
        # Conteo de puertas específicas
        if gate_name == 'x': # Pauli-X
            self.metrics[metrics_definition.single_qubit_gates]['m.NoP-X']["Value"] += 1
            self.metrics[metrics_definition.single_qubit_gates][metrics_definition.m_TNoP]["Value"] += 1
        elif gate_name == 'y': # Pauli-Y
            self.metrics[metrics_definition.single_qubit_gates]['m.NoP-Y']["Value"] += 1
            self.metrics[metrics_definition.single_qubit_gates][metrics_definition.m_TNoP]["Value"] += 1
        elif gate_name == 'z': # Pauli-Z
            self.metrics[metrics_definition.single_qubit_gates]['m.NoP-Z']["Value"] += 1
            self.metrics[metrics_definition.single_qubit_gates][metrics_definition.m_TNoP]["Value"] += 1
        elif gate_name == 'h': # Hadamard
            self.metrics[metrics_definition.single_qubit_gates][metrics_definition.m_NoH]["Value"] += 1
        else: # Resto de puertas simples
            self.metrics[metrics_definition.single_qubit_gates][metrics_definition.m_NoOtherSG]["Value"] += 1

    # Single Qubit Gates
    def _get_sqg_instructions(self, qargs, gate_name):
        if gate_name == 'measure': # Medición
            # Para la puerta measure, se establece la máxima línea temporal a todos los qubits,
            # # ya que la medición se utiliza un momento temporal para cada qubit
            self._set_maximum_time(self.circuit.qubits)
            # Contabilizar puerta y qubits medidos
            self.metrics[metrics_definition.single_qubit_gates][metrics_definition.m_NoOtherSG]["Value"] += 1
            self.metrics['Measurement Gates']['m.NoM']["Value"] += 1
            for qubit in qargs:
                self.qubits_measured.add(qubit._index)
        else:
            # Añadir puerta al momento temporal del circuito y a la línea temporal del qubit
            qubit = qargs[0]
            # Inicializar el momento temporal si no lo está
            if self.current_qubit_timeline[qubit._index] not in self.circuit_timeline:
                self.circuit_timeline[self.current_qubit_timeline[qubit._index]] = 0
            # Contabilizar puerta
            self.circuit_timeline[self.current_qubit_timeline[qubit._index]] += 1
            self.current_qubit_timeline[qubit._index] += 1

            # Conteo de puertas específicas
            self._get_sqg_instructions_count(gate_name)

    # Conteo de puertas simples específicas
    def _get_mqg_instructions_count(self, qargs, gate_name):
        if gate_name == 'ccx': # Toffoli
            self.metrics[metrics_definition.multiple_qubit_gates]['m.NoToff']["Value"] += 1
            for qubit in qargs:
                self.qubits_in_toffoli.add(qubit._index)
                self.qubits_in_toffoli_count[qubit._index] += 1
        else: # Puertas simples controladas
            self.metrics[metrics_definition.single_qubit_gates]['m.TNoCSQG']["Value"] += 1
            if gate_name == 'cx': # CNOT
                self.metrics[metrics_definition.multiple_qubit_gates]['m.NoCNOT']["Value"] += 1
                for qubit in qargs:
                    self.qubits_in_cnot.add(qubit._index)
                    self.qubits_in_cnot_count[qubit._index] += 1

    # Multiple Qubits Gates
    def _get_mqg_instructions(self, qargs, gate_name):
        if 'c' in gate_name: # Puertas controladas
            controlled_gate = gate_name.lstrip('c')
            self.metrics[metrics_definition.all_gates]['m.NoCGates']["Value"] += 1

            # Si se quiere almacenar un conteo individual de las puertas controladas
            if 'Controlled Gates Count' in self.metrics:
                self.metrics['Controlled Gates Count'][controlled_gate]["Value"] += 1

            # Conteo de puertas específicas
            self._get_mqg_instructions_count(qargs, gate_name)
        else: # Resto de puertas múltiples
            if gate_name == 'swap': # SWAP
                self.metrics[metrics_definition.multiple_qubit_gates]['m.NoSWAP']["Value"] += 1

    # Cálculo de las métricas de densidad y superposición
    def _calculate_density_and_superposition_metrics(self):
        # Obtener las densidades del circuito
        self.metrics['Circuit Density']['m.MaxDens']["Value"] = max(self.circuit_timeline.values())
        self.metrics['Circuit Density']['m.AvgDens']["Value"] = round(sum(self.circuit_timeline.values()) / len(self.circuit_timeline.keys()), 3)

        # Calcular %SpposQ = No Qubits con la primera puerta H / Total Qubits
        # # Ponemos los qubits a 0 si hay alguno en False (es decir, que no se le ha aplicado ninguna instrucción)
        self.qubit_in_superposition_state[:] = [0 if qubit is False else qubit for qubit in self.qubit_in_superposition_state]
        self.metrics[metrics_definition.single_qubit_gates]['m.%SpposQ']["Value"] = round(sum(self.qubit_in_superposition_state) / self.circuit.num_qubits, 3)

    # Cálculo del resto de las métricas
    def _calculate_the_rest_metrics(self):
        # Calcular TNoSQG = TNo-P + NoH + NoOtherSG
        self.metrics[metrics_definition.single_qubit_gates][metrics_definition.m_TNoSQG]["Value"] = self.metrics[metrics_definition.single_qubit_gates][metrics_definition.m_TNoP]["Value"] + self.metrics[metrics_definition.single_qubit_gates][metrics_definition.m_NoH]["Value"] + self.metrics[metrics_definition.single_qubit_gates][metrics_definition.m_NoOtherSG]["Value"]

        # Calcular %QInCNOT = Nº qubits afectados / Total Qubits
        self.metrics[metrics_definition.multiple_qubit_gates]['m.%QInCNOT']["Value"] = round(len(self.qubits_in_cnot) / self.circuit.num_qubits, 3)

        # Calcular AvgCNOT = Sum(CNOT en cada qubit) / Total Qubits
        self.metrics[metrics_definition.multiple_qubit_gates]['m.AvgCNOT']["Value"] = round(sum(self.qubits_in_cnot_count) / self.circuit.num_qubits, 3)

        # Calcular MaxCNOT
        self.metrics[metrics_definition.multiple_qubit_gates]['m.MaxCNOT']["Value"] = max(self.qubits_in_cnot_count)

        # Calcular %QInToff = Nº qubits afectados / Total Qubits
        self.metrics[metrics_definition.multiple_qubit_gates]['m.%QInToff']["Value"] = round(len(self.qubits_in_toffoli) / self.circuit.num_qubits, 3)

        # Calcular AvgToff = Sum(Toffoli en cada qubit) / Total Qubits
        self.metrics[metrics_definition.multiple_qubit_gates]['m.AvgToff']["Value"] = round(sum(self.qubits_in_toffoli_count) / self.circuit.num_qubits, 3)

        # Calcular MaxToff
        self.metrics[metrics_definition.multiple_qubit_gates]['m.MaxToff']["Value"] = max(self.qubits_in_toffoli_count)

        # Calcular %SGates = Total de puertas simples (TNoSQG) / Total de puertas (NoGates)
        self.metrics[metrics_definition.all_gates]['m.%SGates']["Value"] = round(self.metrics[metrics_definition.single_qubit_gates][metrics_definition.m_TNoSQG]["Value"] / self.metrics[metrics_definition.all_gates][metrics_definition.m_NoGates]["Value"], 3)

        # Calcular %QM = Nº qubits medidos / Total Qubits
        self.metrics['Measurement Gates']['m.%QM']["Value"] = round(len(self.qubits_measured) / self.circuit.num_qubits, 3)

    def _analyze_circuit(self):
        # Cálculo de las métricas 'Circuit Size'
        self.metrics['Circuit Size']['m.Width']["Value"] = self.circuit.num_qubits
        self.metrics['Circuit Size']['m.Depth']["Value"] = self.circuit.depth()

        # Analizar cada puerta en el circuito
        for instruction, qargs, cargs in self.circuit.data:
            print(f"\n{instruction}") # Instrucción
            print(qargs) # Qubits
            print(cargs) # Classical bits

            # Obtener el nombre de la puerta
            gate_name = instruction.name

            # Conteo de instrucciones por qubit
            self._get_qubit_instruction_count(qargs, gate_name)

            # Añadir la puerta al conteo de puertas y al conteo específico de puertas
            self.metrics[metrics_definition.all_gates][metrics_definition.m_NoGates]["Value"] += 1
            self.gate_count_dict[gate_name] += 1

            # Single Qubit Gates
            if len(qargs) == 1:
                # Realizar el conteo de las instrucciones con puertas simples
                self._get_sqg_instructions(qargs, gate_name)
            # Multiple Qubit Gates
            else:
                # Establecer la máxima línea temporal a los qubits implicados
                self._set_maximum_time(qargs)
                # Realizar el conteo de las instrucciones con puertas múltiples
                self._get_mqg_instructions(qargs, gate_name)

        # Cálculo de las métricas de densidad y superposición
        self._calculate_density_and_superposition_metrics()

        # Cálculo del resto de las métricas
        self._calculate_the_rest_metrics()

    def _get_simplified_circuit(self):
        # Nuevo diccionario para almacenar solo los valores
        simplified_circuit_metrics = {}

        for category, metrics in metrics_definition.circuit_metrics.items():
            for metric, metric_details in metrics.items():
                simplified_circuit_metrics[metric] = metric_details["Value"]
            
        return simplified_circuit_metrics