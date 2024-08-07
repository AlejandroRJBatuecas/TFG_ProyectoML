from flask import Flask, jsonify, render_template
from config import metrics_definition, ml_trained_model_paths

app = Flask(__name__)
app.json.sort_keys = False

app_name = "QuantumPatternsML"

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
    circuit_metrics = metrics_definition.circuit_metrics
    trained_models = ml_trained_model_paths.trained_models
    return render_template('/pattern_analysis/pattern_analysis.html', circuit_metrics=circuit_metrics, trained_models=trained_models)

@app.route('/modelos_ml')
def ml_models():
    return render_template('/ml_models/ml_models.html')

if __name__ == "__main__":
    app.run(debug=True)