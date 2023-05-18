import mlflow


def delete_runs():
    mlflow.set_tracking_uri("http://localhost:5000")
    runs = mlflow.search_runs()

    for run in runs.iterrows():
        mlflow.delete_run(run[1].run_id)


def delete_gpu_trys():
    mlflow.set_tracking_uri("http://localhost:5000")
    runs = mlflow.search_runs()

    for run in runs.iterrows():
        if run[1]["params.model"] in ["svm", "rf", "knn"] and run[1][
            "params.stage"
        ] not in ["Smote_strategy", "Feature_extraction"]:
            mlflow.delete_run(run[1].run_id)
    print("Removidas tentativas com algortimos de GPU")


def delete_statistics_analysis():
    mlflow.set_tracking_uri("http://localhost:5000")
    runs = mlflow.search_runs()

    for run in runs.iterrows():
        if run[1]["params.stage"] == "statistics_analysis":
            mlflow.delete_run(run[1].run_id)
    print("Removidas statistics_analysis")


def conectar_mlflow(url_mlflow):
    mlflow.set_tracking_uri(url_mlflow)
