from metrics import get_metrics
from ml_models import get_prediction

filename = "test_code_files/grover.py"
data_filename = "datasets/dataset_openqasm_qiskit.csv"
test_data_filename = "datasets/file_metrics.csv"

def main():    
    get_metrics(filename, test_data_filename)
    get_prediction(data_filename, test_data_filename)

if __name__ == "__main__":
    main()