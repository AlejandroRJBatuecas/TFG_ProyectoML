import ast
import json
from .metrics_reader import analyze_circuit, draw_circuit, get_metrics_csv
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit, Instruction

def read_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
    return content

class QiskitAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.qiskit_imports = []
        self.circuits = {}
        self.registers = {}
        self.vars = {}

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

    def visit_Assign(self, node):
        # Asignaciones de variables con funciones o clases
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            var_name = node.targets[0].id
            # Verifica si es una asignación de QuantumCircuit, QuantumRegister, ClassicalRegister u otro
            if node.value.func.id == 'QuantumCircuit':
                # Si el circuito no ha sido creado anteriormente
                if var_name not in self.circuits:
                    circuit = QuantumCircuit()
                    # Añadir cada registro asignado al circuito
                    for arg in node.value.args:
                        if arg.id in self.registers:
                            circuit.add_register(self.registers[arg.id])
                    self.circuits[var_name] = circuit
            elif node.value.func.id in ['QuantumRegister', 'ClassicalRegister']:
                # Si el registro no ha sido creado anteriormente
                if var_name not in self.registers:
                    # Definir los parámetros del registro
                    register_params = {
                        'size': 0,
                        'name': None,
                        'bits': None
                    }
                    # Obtener los parámetros del registro
                    # # Argumentos sin etiqueta
                    for i, arg in enumerate(node.value.args):
                        if i == 0:
                            register_params['size'] = arg.value
                        elif i == 1:
                            register_params['name'] = arg.value
                        elif i == 2:
                            register_params['bits'] = arg.value
                    # # Argumentos con etiqueta
                    for keyword in node.value.keywords:
                        register_params[keyword.arg] = keyword.value.value
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
            else:
                if var_name not in self.vars:
                    self.vars[var_name] = {}
        # Asignaciones de variables con valores determinados
        elif isinstance(node.value, ast.Constant):
            var_name = node.targets[0].id
            if var_name not in self.vars:
                self.vars[var_name] = node.value.value

        self.generic_visit(node)

    def visit_Expr(self, node):
        # Verificar si es una instrucción aplicada a un circuito cuántico
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            #print(node.value.func.value.id)
            #print(node.value.func.attr)
            circuit_name = node.value.func.value.id
            if circuit_name in self.circuits:
                circuit_instruction_name = node.value.func.attr
                print("Circuito", circuit_name, ": Instrucción", circuit_instruction_name)
                instruction_args = {
                    'name': circuit_instruction_name,
                    'num_qubits': 0,
                    'num_clbits': 0,
                    'params': []
                }
                qargs = []
                cargs = []
                for arg in node.value.args:
                    if isinstance(arg, ast.Subscript): # Referencia a un QuantumRegister
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
                # Aplicar instrucción al circuito
                self.circuits[circuit_name].append(Instruction(**instruction_args), qargs, cargs)

        self.generic_visit(node)

def analyze_code(code):
    tree = ast.parse(code)
    print(ast.dump(tree, indent=4)) # Muestra el código en forma de árbol para debug
    visitor = QiskitAnalyzer()
    visitor.visit(tree)
    return visitor.qiskit_imports, visitor.circuits, visitor.registers, visitor.vars

def count_instructions(instructions):
    counting = {}
    for circuit, instruction in instructions:
        if circuit not in counting:
            counting[circuit] = {}
        if instruction not in counting[circuit]:
            counting[circuit][instruction] = 0
        counting[circuit][instruction] += 1
    return counting

def get_metrics(filename, test_data_filename):
    code = read_file(filename)
    qiskit_imports, circuits, registers, vars = analyze_code(code)
    print("\n--- MÉTRICAS ---")
    if qiskit_imports:
        print("Importaciones:", qiskit_imports)
        print("Circuitos:", circuits)
        print("Registros:", registers)
        print("Variables:", vars)
        results = []
        for circuit_name, circuit in circuits.items():
            print("\n--- CIRCUITO", circuit_name, "---")
            print(circuit.data)
            
            # Analizar el circuito
            circuit_results = analyze_circuit(circuit)
            results.append(circuit_results)

            # Exportar los resultados a JSON
            results_json = json.dumps(results, indent=4)
            print(f"\n{results_json}")

            # Dibujar el circuito
            draw_circuit(circuit)

        # Obtener el csv con las métricas
        get_metrics_csv(results, test_data_filename)
    else:
        print("No hay importaciones de qiskit")

# Ejemplo de uso
if __name__ == "__main__":
    filename = "../test_code_files/grover.py"
    test_data_filename = "../datasets/file_metrics.csv"
    get_metrics(filename, test_data_filename)