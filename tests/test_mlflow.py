import pytest
import mlflow
from src.mlflow_funcs import conectar_mlflow


def test_conectar_mlflow():
    # Configuração
    url_mlflow = "http://localhost:5000"  # URL do servidor do MLflow

    # Conexão com o MLflow
    try:
        conectar_mlflow(url_mlflow)
    except Exception as e:
        pytest.fail(f"Falha ao conectar ao MLflow: {e}")
