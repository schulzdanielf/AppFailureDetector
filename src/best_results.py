import mlflow


def ajust_columns(results):
    for c in results.columns:
        if c[:7] == "params.":
            results = results.rename(columns={c: c[7:]})

    results = results.rename(columns={"metrics.f1_val": "f1_val"})
    return results


def correct_parameters(best_results):
    for result in best_results:
        if result != "knn":
            try:
                del best_results[result]["n_neighbors"]
                del best_results[result]["metric"]
            except KeyError:
                pass
    for k in best_results:
        for p in best_results[k]:
            if p in ("n_estimators", "n_neighbors"):
                best_results[k][p] = int(best_results[k][p])
            if p in ("C", "var_smoothing", "learning_rate"):
                best_results[k][p] = float(best_results[k][p])

    return best_results


def get_best_parameters(split_strategy):
    mlflow.set_tracking_uri("http://localhost:5000")
    results = mlflow.search_runs()

    results = ajust_columns(results)
    query = f'model_selection == "{split_strategy}"'
    grouped = results.query(query).groupby("type")
    indices_max = grouped["f1_val"].idxmax()
    best_results = {}

    for modelo, indice in indices_max.items():
        parametros = results.loc[
            indice,
            [
                "preprocessing",
                "C",
                "kernel",
                "n_estimators",
                "n_neighbors",
                "criterion",
                "var_smoothing",
                "learning_rate",
                "metric"
            ],
        ]
        parametros = {
            chave: valor
            for chave, valor in parametros.to_dict().items()
            if type(valor) == str
        }
        best_results[modelo] = parametros

    return correct_parameters(best_results)
