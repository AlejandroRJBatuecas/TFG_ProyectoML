import ast
import os

def leer_archivo(nombre_archivo):
    with open(nombre_archivo, 'r') as archivo:
        contenido = archivo.read()
    return contenido

class QiskitCircuitVisitor(ast.NodeVisitor):
    def __init__(self):
        self.circuitos = []
        self.instrucciones = []

    def visit_Assign(self, node):
        # Verifica si es una asignación de QuantumCircuit
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if node.value.func.attr == 'construct_circuit':
                print(node)
                nombre_circuito = node.targets[0].id
                self.circuitos.append(nombre_circuito)
        self.generic_visit(node)

    def visit_Expr(self, node):
        # Verifica si es una instrucción aplicada a un circuito cuántico
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            print(node.value.func.value.id)
            print(node.value.func.attr)
            nombre_circuito = node.value.func.value.id
            if nombre_circuito in self.circuitos:
                self.instrucciones.append((nombre_circuito, node.value.func.attr))
        self.generic_visit(node)

def analizar_codigo(codigo):
    tree = ast.parse(codigo)
    print(ast.dump(tree, indent=4))
    visitor = QiskitCircuitVisitor()
    visitor.visit(tree)
    return visitor.circuitos, visitor.instrucciones

def contar_instrucciones(instrucciones):
    conteo = {}
    for circuito, instruccion in instrucciones:
        if circuito not in conteo:
            conteo[circuito] = {}
        if instruccion not in conteo[circuito]:
            conteo[circuito][instruccion] = 0
        conteo[circuito][instruccion] += 1
    return conteo

def obtener_metricas(nombre_archivo):
    codigo = leer_archivo(nombre_archivo)
    circuitos, instrucciones = analizar_codigo(codigo)
    print(circuitos)
    print(instrucciones)
    metricas = contar_instrucciones(instrucciones)
    return metricas

# Ejemplo de uso
# Obtiene la ruta del directorio actual
current_directory = os.path.dirname(os.path.abspath(__file__))
# Construye la ruta al archivo en una carpeta anterior y luego entra en la carpeta 'projects'
filename = os.path.join(current_directory, '..', 'projects', 'qiskit_algorithms', 'phase_estimators', 'ipe.py')
metricas = obtener_metricas(filename)
print(metricas)