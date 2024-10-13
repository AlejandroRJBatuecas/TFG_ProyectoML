import json, joblib

from flask import Flask, jsonify, render_template, request
from pathlib import Path
from config import metrics_definition, ml_trained_model_paths, ml_parameters
from ml_subsystem.ml_models import KNNClassifierModel, KNNOvsRClassifierModel, RandomForestClassifierModel, store_model
from sw_subsystem.business.metrics import ASTFileReader

app = Flask(__name__)
app.json.sort_keys = False

# Nombre de la aplicación
APP_NAME = "QPP-ML"

# Constantes
PATTERN_ANALYSIS_HTML_FILE = '/pattern_analysis/pattern_analysis.html'

def get_or_create_model(trained_model):
    # Obtener la ruta de almacenamiento del modelo
    trained_model_path = ml_trained_model_paths.trained_model_base_path+trained_model+ml_trained_model_paths.trained_model_extension

    # Si existe el modelo entrenado, lo recuperamos. Si no, lo creamos
    if Path(trained_model_path).is_file():
        model = joblib.load(trained_model_path)
    else:
        if trained_model == "knn_classifier":
            model = KNNClassifierModel()
        elif trained_model == "knn_one_vs_rest_classifier":
            model = KNNOvsRClassifierModel()
        elif trained_model == "random_forest_classifier":
            model = RandomForestClassifierModel()

        store_model(model, trained_model_path)

    return model

@app.context_processor
def inject_global_vars():
    return dict(app_name=APP_NAME)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_metrics')
def obtain_metrics():
    return jsonify(metrics_definition.circuit_metrics)

@app.route('/pattern_analysis')
def pattern_analysis():
    return render_template(PATTERN_ANALYSIS_HTML_FILE, trained_models=ml_trained_model_paths.trained_models)

def get_simplified_circuit(metric_detail):
    # Nuevo diccionario para almacenar solo los valores
    simplified_circuit_metrics = {}

    for category, metrics in metrics_definition.circuit_metrics.items():
        for metric, metric_details in metrics.items():
            simplified_circuit_metrics[metric] = metric_details[metric_detail]
        
    return simplified_circuit_metrics

@app.route('/predict', methods=['POST'])
def predict():
    circuits_list = [] # Lista de las métricas de los circuitos
    selected_models = [] # Lista de modelos seleccionados

    # Obtener las claves y valores del formulario
    form_keys = list(request.form.keys())
    form_values = list(request.form.values())
    
    # Obtener el número de métricas de un circuito y el número de circuitos en el formulario
    circuit_metrics_total_values = sum(len(sub_dict) for sub_dict in metrics_definition.circuit_metrics.values())
    circuits_number = int(len(request.form) / circuit_metrics_total_values)

    selected_models_num = len(request.form)-circuit_metrics_total_values*circuits_number

    if selected_models_num == 0:
        circuits_list = "No model selected"
        return render_template(PATTERN_ANALYSIS_HTML_FILE, trained_models=ml_trained_model_paths.trained_models, circuits_list=circuits_list)

    # Crear los diccionarios de métricas de cada circuito y añadirlos a la lista
    list_index = selected_models_num
    for _ in range(0, circuits_number): # Para cada circuito
        circuit_dict = {} # Crear el diccionario de métricas
        for _ in range(0, circuit_metrics_total_values): # Para cada métrica
            # Obtener el nombre de la métrica y añadirla con su valor al diccionario
            metric_name = form_keys[list_index].split("_")[0]

            try:
                circuit_dict[metric_name] = float(form_values[list_index])
            except (ValueError, TypeError):
                circuits_list = "Metric values must be numeric"
                return render_template(PATTERN_ANALYSIS_HTML_FILE, trained_models=ml_trained_model_paths.trained_models, circuits_list=circuits_list)
            list_index+=1

        # Añadir el diccionario a la lista
        circuits_list.append(circuit_dict)

    # Guardar los circuitos en un archivo JSON
    with open(ml_parameters.test_data_filename, 'w') as json_file:
        json.dump(circuits_list, json_file, indent=4)

    # Obtener los modelos seleccionados
    for i in range(0, selected_models_num):
        selected_models.append(form_values[i])

    circuits_predictions = [{} for _ in range(circuits_number)] # Lista que contiene las predicciones de los circuitos
    
    for selected_model in selected_models:
        model = get_or_create_model(selected_model)
        model_predictions_list = model.get_prediction()
        for i, value in enumerate(model_predictions_list):
            circuits_predictions[i][selected_model] = value

    circuits_metrics = get_simplified_circuit("Descriptive name")
    return render_template('/pattern_analysis/prediction_results.html', circuits_predictions=circuits_predictions, patterns_list=ml_parameters.patterns_list, circuits_metrics=circuits_metrics, circuits_list=circuits_list)

def allowed_file(filename, file_extension):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() == file_extension

def get_json_file_metrics(file_data):
    circuit_metrics = []
    # Obtener el circuito base simplificado
    simplified_circuit = get_simplified_circuit("Value")

    # Recorrer cada elemento de la lista
    for circuit in file_data:
        # Comprobar si es un diccionario
        if isinstance(circuit, dict):
            circuit_dict = {}
            # Recorrer las métricas del diccionario original buscando los elementos en el diccionario del json
            for metric in simplified_circuit.keys():
                if metric in circuit: # Si encuentra la clave, se añade el valor al diccionario
                    circuit_dict[metric] = circuit[metric]
                else:
                    circuit_dict[metric] = "Undefined"

            # Añadir el circuito a la lista de métricas
            circuit_metrics.append(circuit_dict)

    return circuit_metrics

def read_json_file(file):
    file_content_string = None
    circuit_metrics = None

    # Leer el contenido del archivo
    try:
        file_content = json.load(file)
    except json.JSONDecodeError:
        return file_content_string, circuit_metrics

    # Comprobar si el fichero contiene una lista 
    if isinstance(file_content, list):
        # Obtener las métricas del fichero
        circuit_metrics = get_json_file_metrics(file_content)
    elif isinstance(file_content, dict):
        # Añadir el diccionario a una lista
        file_content = [file_content]
        # Obtener las métricas del fichero
        circuit_metrics = get_json_file_metrics(file_content)

    # Devolver el contenido del archivo para mostrarlo
    file_content_string = json.dumps(file_content, indent=4)

    return file_content_string, circuit_metrics

def read_python_file(file):
    # Obtener el nombre del archivo y su contenido
    file_name = file.filename
    file_content = file.read().decode('utf-8')

    # Obtener las métricas de los circuitos y sus imágenes
    circuits_list = ASTFileReader(file_name, file_content).circuits_list

    return file_content, circuits_list

@app.route('/import_file', methods=['POST'])
def import_file():
    print(request.files)

    error_message = "Error importing file"

    # Verificar si se ha enviado un archivo
    if 'file' not in request.files:
        return render_template(PATTERN_ANALYSIS_HTML_FILE, trained_models=ml_trained_model_paths.trained_models, error_message=error_message)

    # Obtener el archivo del formulario
    file = request.files['file']

    # Verificar si se ha seleccionado un archivo
    if file.filename == '':
        error_message = "No file selected" 
        return render_template(PATTERN_ANALYSIS_HTML_FILE, trained_models=ml_trained_model_paths.trained_models, error_message=error_message)

    # Obtener la extensión del archivo requerido
    file_extension = request.form['file_extension']

    # Verificar la extensión del archivo
    if not allowed_file(file.filename, file_extension.replace(".", "")):
        error_message = "The file does not have the specified extension"
        return render_template(PATTERN_ANALYSIS_HTML_FILE, trained_models=ml_trained_model_paths.trained_models, error_message=error_message)

    # Lectura del archivo dependiendo de su extensión
    if file_extension == ".json":
        file_content, circuit_metrics = read_json_file(file)
        # Si se han obtenido las métricas correctamente, devolver el archivo y las métricas
        if circuit_metrics:
            return render_template(PATTERN_ANALYSIS_HTML_FILE, trained_models=ml_trained_model_paths.trained_models, file_name=file.filename, file_content=file_content, file_extension=file_extension.replace(".", ""), circuit_metrics=circuit_metrics)
        else:
            error_message = "The file does not have the specified format"
            return render_template(PATTERN_ANALYSIS_HTML_FILE, trained_models=ml_trained_model_paths.trained_models, error_message=error_message)
    elif file_extension == ".py":
        file_content, circuits_list = read_python_file(file)
        # Si se han obtenido las métricas correctamente, devolver el archivo y las métricas
        if circuits_list:
            return render_template(PATTERN_ANALYSIS_HTML_FILE, trained_models=ml_trained_model_paths.trained_models, file_name=file.filename, file_content=file_content, file_extension=file_extension.replace(".", ""), circuits_list=circuits_list)
        else:
            error_message = "The file does not have the specified format"
            return render_template(PATTERN_ANALYSIS_HTML_FILE, trained_models=ml_trained_model_paths.trained_models, error_message=error_message)
    else:
        error_message= "The file does not have the specified extension"
        return render_template(PATTERN_ANALYSIS_HTML_FILE, trained_models=ml_trained_model_paths.trained_models, error_message=error_message)

@app.route('/ml_models')
def ml_models():
    return render_template('/ml_models/ml_models.html')

if __name__ == "__main__":
    app.run(debug=True)