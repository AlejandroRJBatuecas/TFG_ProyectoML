import ast
import json

from .metrics_analyzer import MetricsAnalyzer
from config import ml_parameters
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit, Instruction

class QiskitAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.qiskit_imports = []
        self.circuits = {}
        self.registers = {}
        self.code_vars = {}

    def visit_Import(self, node):
        # Detectar importaciones de Qiskit
        for alias in node.names:
            if alias.name.startswith('qiskit'):
                self.qiskit_imports.append(alias.name)
                
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        # Detectar importaciones de Qiskit desde módulos específicos
        if node.module and node.module.startswith('qiskit'):
            self.qiskit_imports.append(node.module)

        self.generic_visit(node)

    # Establecer el circuito cuántico
    def _set_quantum_circuit(self, var_name, node):
        # Si el circuito no ha sido creado anteriormente
        if var_name not in self.circuits:
            circuit = QuantumCircuit()
            # Añadir cada registro asignado al circuito
            for arg in node.value.args:
                if arg.id in self.registers:
                    circuit.add_register(self.registers[arg.id])
            self.circuits[var_name] = circuit

    # Obtener los parámetros del registro
    def _get_register_params(self, node):
        # Definir los parámetros del registro
        register_params = {
            'size': 0,
            'name': None,
            'bits': None
        }
        # Argumentos sin etiqueta
        for i, arg in enumerate(node.value.args):
            if i == 0:
                register_params['size'] = arg.value
            elif i == 1:
                register_params['name'] = arg.value
            elif i == 2:
                register_params['bits'] = arg.value
        # Argumentos con etiqueta
        for keyword in node.value.keywords:
            register_params[keyword.arg] = keyword.value.value

        return register_params

    # Establecer el registro (cuántico o clásico)
    def _set_register(self, var_name, node):
        # Si el registro no ha sido creado anteriormente
        if var_name not in self.registers:
            # Obtener los parámetros del registro
            register_params = self._get_register_params(node)
            # Crear y almacenar los registros
            if node.value.func.id == 'QuantumRegister':
                self.registers[var_name] = QuantumRegister(
                    size=register_params['size'], 
                    name=register_params['name'], 
                    bits=register_params['bits'])
            else:
                self.registers[var_name] = ClassicalRegister(
                    size=register_params['size'], 
                    name=register_params['name'], 
                    bits=register_params['bits'])

    def visit_Assign(self, node):
        # Asignaciones de variables con funciones o clases
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            var_name = node.targets[0].id
            # Verifica si es una asignación de QuantumCircuit, QuantumRegister, ClassicalRegister u otro
            if node.value.func.id == 'QuantumCircuit':
                # Establecer el circuito cuántico
                self._set_quantum_circuit(var_name, node)
            elif node.value.func.id in ['QuantumRegister', 'ClassicalRegister']:
                # Establecer el registro (cuántico o clásico)
                self._set_register(var_name, node)
            else:
                if var_name not in self.code_vars:
                    self.code_vars[var_name] = {}
        # Asignaciones de variables con valores determinados
        elif isinstance(node.value, ast.Constant):
            var_name = node.targets[0].id
            if var_name not in self.code_vars:
                self.code_vars[var_name] = node.value.value

        self.generic_visit(node)

    # Obtener los argumentos de la instrucción
    def _get_instruction_args(self, node, circuit_instruction_name):
        # Definir los argumentos de la instrucción
        instruction_args = {
            'name': circuit_instruction_name,
            'num_qubits': 0,
            'num_clbits': 0,
            'params': []
        }
        qargs = []
        cargs = []

        # Obtener los argumentos de la instrucción
        for arg in node.value.args:
            # Si se referencia un QuantumRegister
            if isinstance(arg, ast.Subscript):
                print(arg)
                # Obtener la variable y la posicion del registro
                register_var = arg.value.id
                register_pos = arg.slice.value
                print(f"{register_var}[{register_pos}]")
                # Obtener la variable y el index
                register = self.registers[register_var][register_pos]
                print(register)
                # Almacenar el registro en los argumentos de la instrucción
                if isinstance(register, Qubit):
                    instruction_args['num_qubits'] += 1
                    qargs.append(register)
                elif isinstance(register, Clbit):
                    instruction_args['num_clbits'] += 1
                    cargs.append(register)

        return instruction_args, qargs, cargs

    def visit_Expr(self, node):
        # Verificar si es una instrucción aplicada a un circuito cuántico
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            circuit_name = node.value.func.value.id
            if circuit_name in self.circuits:
                circuit_instruction_name = node.value.func.attr
                print("\nCircuito", circuit_name, ": Instrucción", circuit_instruction_name)
                # Obtener los argumentos de la instrucción
                instruction_args, qargs, cargs = self._get_instruction_args(node, circuit_instruction_name)
                # Aplicar instrucción al circuito
                self.circuits[circuit_name].append(Instruction(**instruction_args), qargs, cargs)

        self.generic_visit(node)

class ASTFileReader:
    def __init__(self, filename: str):
        self.filename = filename

        # Leer el fichero para obtener el código
        self.code = self._read_file()

        # Obtener el analizador de qiskit, para obtener todas las variables
        self.qiskit_analyzer = self._analyze_code()

        # Obtener las circuitos y sus métricas
        self.circuits_list = self.get_metrics()

        # Obtener el json con las métricas
        self._get_metrics_json()

    def _read_file(self):
        with open(self.filename, 'r') as file:
            content = file.read()

        return content

    def _analyze_code(self):
        tree = ast.parse(self.code)
        print(ast.dump(tree, indent=4)) # Muestra el código en forma de árbol para debug
        visitor = QiskitAnalyzer()
        visitor.visit(tree)

        return visitor

    def _get_metrics_json(self):
        # Guardar los circuitos en un archivo JSON
        with open(ml_parameters.test_data_filename, 'w') as json_file:
            json.dump(self.circuits_list, json_file, indent=4)

    def get_metrics(self):
        circuits_list = []

        if self.qiskit_analyzer.qiskit_imports:
            print("\n--- MÉTRICAS ---")
            print("Importaciones:", self.qiskit_analyzer.qiskit_imports)
            print("Circuitos:", self.qiskit_analyzer.circuits)
            print("Registros:", self.qiskit_analyzer.registers)
            print("Variables:", self.qiskit_analyzer.code_vars)
            
            for circuit_name, circuit in self.qiskit_analyzer.circuits.items():
                print("\n--- CIRCUITO", circuit_name, "---")
                print(circuit.data)
                
                # Analizar el circuito y añadirlo a la lista
                metrics_analyzer = MetricsAnalyzer(circuit, draw_circuit=True)
                circuits_list.append(metrics_analyzer.simplified_circuit)
        else:
            print("No hay importaciones de qiskit")

        return circuits_list