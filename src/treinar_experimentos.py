import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
np.random.seed(SEED)

def salvar_relatorio(nome, y_true, y_pred, classes, pasta_saida):
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    texto = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        zero_division=0,
    )

    with open(
        pasta_saida / f"{nome}_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as arquivo:
        arquivo.write(str(texto))

    matriz = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=classes,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45, values_format="d")
    ax.set_title(f"Matriz de confusão - {nome}")

    plt.tight_layout()
    plt.savefig(pasta_saida / f"{nome}_matriz_confusao.png", dpi=150)
    plt.close(fig)

def experimento_svm(
    nome, params, x_train, y_train, x_val, y_val, x_test, y_test, classes, pasta_saida
):
    print("\n" + "=" * 80)
    print(f"Executando {nome}")
    print(f"Técnica: SVM | Parâmetros: {params}")

    inicio = time.time()

    modelo = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(**params)),
        ]
    )

    modelo.fit(x_train, y_train)

    y_val_pred = modelo.predict(x_val)
    metricas_val = calcular_metricas(y_val, y_val_pred)

    y_test_pred = modelo.predict(x_test)
    metricas_teste = calcular_metricas(y_test, y_test_pred)

    salvar_relatorio(nome, y_test, y_test_pred, classes, pasta_saida)

    tempo = time.time() - inicio

    return {
        "experimento": nome,
        "tecnica": "SVM",
        "parametros": str(params),
        "tempo_segundos": round(tempo, 2),
        **{f"val_{k}": v for k, v in metricas_val.items()},
        **{f"teste_{k}": v for k, v in metricas_teste.items()},
    }

def experimento_mlp(
    nome, params, x_train, y_train, x_val, y_val, x_test, y_test, classes, pasta_saida
):
    print("\n" + "=" * 80)
    print(f"Executando {nome}")
    print(f"Técnica: MLP Scikit-Learn | Parâmetros: {params}")

    inicio = time.time()

    modelo = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=params["hidden_layer_sizes"],
                    activation="relu",
                    solver="adam",
                    alpha=params["alpha"],
                    learning_rate_init=params["learning_rate_init"],
                    max_iter=params["max_iter"],
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=8,
                    random_state=SEED,
                    verbose=False,
                ),
            ),
        ]
    )

    modelo.fit(x_train, y_train)

    y_val_pred = modelo.predict(x_val)
    metricas_val = calcular_metricas(y_val, y_val_pred)

    y_test_pred = modelo.predict(x_test)
    metricas_teste = calcular_metricas(y_test, y_test_pred)

    salvar_relatorio(nome, y_test, y_test_pred, classes, pasta_saida)

    tempo = time.time() - inicio

    return {
        "experimento": nome,
        "tecnica": "MLP Scikit-Learn",
        "parametros": str(params),
        "tempo_segundos": round(tempo, 2),
        **{f"val_{k}": v for k, v in metricas_val.items()},
        **{f"teste_{k}": v for k, v in metricas_teste.items()},
    }
