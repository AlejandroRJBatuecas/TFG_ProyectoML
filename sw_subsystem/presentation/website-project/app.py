import json, joblib

from flask import Flask, jsonify, render_template, request
from pathlib import Path
from config import metrics_definition, ml_trained_model_paths, ml_parameters
from ml_subsystem.ml_models import KNNClassifierModel, KNNOvsRClassifierModel, RandomForestClassifierModel, store_model

app = Flask(__name__)
app.json.sort_keys = False

app_name = "QuantumPatternsML"

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
    return dict(app_name=app_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/obtener_metricas')
def obtain_metrics():
    return jsonify(metrics_definition.circuit_metrics)

@app.route('/analisis_patrones')
def pattern_analysis():
    return render_template('/pattern_analysis/pattern_analysis.html', trained_models=ml_trained_model_paths.trained_models)

@app.route('/predecir', methods=['POST'])
def predict():
    circuits_list = [] # Lista de las métricas de los circuitos
    selected_models = [] # Lista de modelos seleccionados

    print(request.form)

    # Obtener las claves y valores del formulario
    form_keys = list(request.form.keys())
    form_values = list(request.form.values())
    
    # Obtener el número de métricas de un circuito y el número de circuitos en el formulario
    circuit_metrics_total_values = sum(len(sub_dict) for sub_dict in metrics_definition.circuit_metrics.values())
    circuits_number = int(len(request.form) / circuit_metrics_total_values)

    print(circuit_metrics_total_values, " x ", circuits_number, " -- ", len(request.form))

    selected_models_num = len(request.form)-circuit_metrics_total_values*circuits_number
    print(selected_models_num)

    if selected_models_num == 0:
        circuits_list = "No se ha seleccionado ningún modelo"
        return render_template('/pattern_analysis/pattern_analysis.html', trained_models=ml_trained_model_paths.trained_models, circuits_list=circuits_list)

    # Crear los diccionarios de métricas de cada circuito y añadirlos a la lista
    list_index = 0
    for _ in range(0, circuits_number): # Para cada circuito
        circuit_dict = {} # Crear el diccionario de métricas
        for _ in range(0, circuit_metrics_total_values): # Para cada métrica
            # Obtener el nombre de la métrica y añadirla con su valor al diccionario
            metric_name = form_keys[list_index].split("_")[0]
            try:
                circuit_dict[metric_name] = float(form_values[list_index])
            except:
                circuits_list = "Los valores de las métricas deben ser numéricos"
                return render_template('/pattern_analysis/pattern_analysis.html', trained_models=ml_trained_model_paths.trained_models, circuits_list=circuits_list)
            list_index+=1

        # Añadir el diccionario a la lista
        circuits_list.append(circuit_dict)

    # Guardar los circuitos en un archivo JSON
    with open(ml_parameters.test_data_filename, 'w') as json_file:
        json.dump(circuits_list, json_file, indent=4)

    print(circuits_list)

    # Obtener los circuitos seleccionados
    for i in range(0, len(request.form)-list_index):
        selected_models.append(form_values[list_index+i])

    print(selected_models)
    
    for selected_model in selected_models:
        model = get_or_create_model(selected_model)
        model.get_prediction()

    return render_template('/pattern_analysis/prediction_results.html', circuits_list=circuits_list)

@app.route('/modelos_ml')
def ml_models():
    return render_template('/ml_models/ml_models.html')

if __name__ == "__main__":
    app.run(debug=True)