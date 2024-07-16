from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analisis_patrones')
def pattern_analysis():
    return render_template('/pattern_analysis/pattern_analysis.html')

@app.route('/modelos_ml')
def ml_models():
    return render_template('/ml_models/ml_models.html')

if __name__ == "__main__":
    app.run(debug=True)