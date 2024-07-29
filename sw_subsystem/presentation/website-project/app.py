from flask import Flask, jsonify, render_template
from config import metrics_definition

app = Flask(__name__)
app.json.sort_keys = False

app_name = "QuantumPatternsML"

@app.context_processor
def inject_global_vars():
    return dict(app_name=app_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analisis_patrones')
def pattern_analysis():
    metrics = metrics_definition
    return render_template('/pattern_analysis/pattern_analysis.html', metrics=metrics)

@app.route('/modelos_ml')
def ml_models():
    return render_template('/ml_models/ml_models.html')

if __name__ == "__main__":
    app.run(debug=True)