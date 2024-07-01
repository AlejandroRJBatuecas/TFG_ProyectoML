import joblib

from config import ml_parameters
from pathlib import Path
from sw_subsystem.business.metrics import get_metrics
from ml_subsystem.ml_models import KNNClassifierModel, store_model

filename = "sw_subsystem/business/metrics/test_code_files/grover_wo_reset.py" # Fichero a analizar
trained_model_path = ml_parameters.knn_classifier_trained_model_path # Modelo

def main():    
    get_metrics(filename, ml_parameters.test_data_filename)

    # Si existe el modelo entrenado, lo recuperamos. Si no, lo creamos
    if Path(trained_model_path).is_file():
        model = joblib.load(trained_model_path)
    else:
       model = KNNClassifierModel()
       store_model(model, trained_model_path)
    
    model.get_prediction()

def show_model_evaluation():
    # Si existe el modelo entrenado, lo recuperamos. Si no, lo creamos
    if Path(trained_model_path).is_file():
        model = joblib.load(trained_model_path)
        model.show_model_evaluation()
    else:
       model = KNNClassifierModel()
       store_model(model, trained_model_path)

if __name__ == "__main__":
    main()