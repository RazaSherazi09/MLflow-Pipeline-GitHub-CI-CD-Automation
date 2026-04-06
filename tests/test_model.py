import os

def test_model_file_exists():
    assert os.path.exists("models/best_model.pkl"), "Model file not found!"