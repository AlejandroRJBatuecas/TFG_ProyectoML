import joblib

from config import ml_parameters, ml_trained_model_paths
from pathlib import Path
from sw_subsystem.business.metrics import get_metrics
from ml_subsystem.ml_models import KNNClassifierModel, KNNOvsRClassifierModel, RandomForestClassifierModel, store_model

filename = "sw_subsystem/business/metrics/test_code_files/grover.py" # Fichero a analizar
trained_model_path = ml_trained_model_paths.random_forest_classifier_trained_model_path # Modelo

def get_or_create_model():
    # Si existe el modelo entrenado, lo recuperamos. Si no, lo creamos
    if Path(trained_model_path).is_file():
        model = joblib.load(trained_model_path)
    else:
        if trained_model_path == ml_trained_model_paths.knn_classifier_trained_model_path:
            model = KNNClassifierModel()
        elif trained_model_path == ml_trained_model_paths.knn_one_vs_rest_classifier_trained_model_path:
            model = KNNOvsRClassifierModel()
        elif trained_model_path == ml_trained_model_paths.random_forest_classifier_trained_model_path:
            model = RandomForestClassifierModel()

        store_model(model, trained_model_path)

    return model

def prediction():    
    get_metrics(filename, ml_parameters.test_data_filename)
    model = get_or_create_model()
    model.get_prediction()

def model_evaluation():
    model = get_or_create_model()
    model.show_model_evaluation()

if __name__ == "__main__":
    prediction()