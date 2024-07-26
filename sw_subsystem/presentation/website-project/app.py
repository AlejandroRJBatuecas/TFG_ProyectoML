from flask import Flask, jsonify, render_template
from sw_subsystem.business.metrics import define_metrics

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
    metrics = define_metrics()
    return jsonify(metrics)

@app.route('/analisis_patrones')
def pattern_analysis():
    metrics = define_metrics()
    return render_template('/pattern_analysis/pattern_analysis.html', metrics=metrics)

@app.route('/modelos_ml')
def ml_models():
    return render_template('/ml_models/ml_models.html')

if __name__ == "__main__":
    app.run(debug=True)