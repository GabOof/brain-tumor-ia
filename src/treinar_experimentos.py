import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import carregar_dataset

SEED = 42
np.random.seed(SEED)

def calcular_metricas(y_true, y_pred):
    return {
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

def salvar_relatorio(nome, y_true, y_pred, classes, pasta_saida):
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    texto = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    with open(pasta_saida / f"{nome}_classification_report.txt", "w", encoding="utf-8") as arquivo:
        arquivo.write(str(texto))

    matriz = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=classes)
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

    mlp = modelo.named_steps["mlp"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mlp.loss_curve_)
    ax.set_title(f"Curva de perda - {nome}")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig(Path(pasta_saida) / f"{nome}_curva_loss.png", dpi=150)
    plt.close(fig)

    tempo = time.time() - inicio

    return {
        "experimento": nome,
        "tecnica": "MLP Scikit-Learn",
        "parametros": str(params),
        "tempo_segundos": round(tempo, 2),
        **{f"val_{k}": v for k, v in metricas_val.items()},
        **{f"teste_{k}": v for k, v in metricas_teste.items()},
    }

def main():
    parser = argparse.ArgumentParser(
        description="Executa 8 experimentos para classificação de tumores cerebrais."
    )
    parser.add_argument("--data-dir", default="data", help="Pasta do dataset.")
    parser.add_argument("--img-size", type=int, default=64, help="Tamanho da imagem quadrada.")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Limite de imagens por classe para testes rápidos.",
    )
    parser.add_argument("--saida", default="resultados", help="Pasta de saída dos resultados.")
    args = parser.parse_args()

    pasta_saida = Path(args.saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    x_train, x_val, x_test, y_train, y_val, y_test, classes = carregar_dataset(
        data_dir=args.data_dir,
        img_size=args.img_size,
        max_per_class=args.max_per_class,
        seed=SEED,
    )

    experimentos_svm = [
        ("E01_SVM_linear_C1", {"kernel": "linear", "C": 1.0}),
        ("E02_SVM_linear_C01", {"kernel": "linear", "C": 0.1}),
        ("E03_SVM_rbf_C1", {"kernel": "rbf", "C": 1.0, "gamma": "scale"}),
        ("E04_SVM_rbf_C10", {"kernel": "rbf", "C": 10.0, "gamma": "scale"}),
    ]

    experimentos_mlp = [
        (
            "E05_MLP_128_alpha0001",
            {
                "hidden_layer_sizes": (128,),
                "alpha": 0.0001,
                "learning_rate_init": 0.001,
                "max_iter": 120,
            },
        ),
        (
            "E06_MLP_256_alpha0001",
            {
                "hidden_layer_sizes": (256,),
                "alpha": 0.0001,
                "learning_rate_init": 0.001,
                "max_iter": 120,
            },
        ),
        (
            "E07_MLP_256_128_alpha001",
            {
                "hidden_layer_sizes": (256, 128),
                "alpha": 0.001,
                "learning_rate_init": 0.001,
                "max_iter": 150,
            },
        ),
        (
            "E08_MLP_512_256_alpha01",
            {
                "hidden_layer_sizes": (512, 256),
                "alpha": 0.01,
                "learning_rate_init": 0.0005,
                "max_iter": 150,
            },
        ),
    ]

    resultados = []

    for nome, params in experimentos_svm:
        resultado = experimento_svm(
            nome, params, x_train, y_train, x_val, y_val, x_test, y_test, classes, pasta_saida
        )
        resultados.append(resultado)

    for nome, params in experimentos_mlp:
        resultado = experimento_mlp(
            nome, params, x_train, y_train, x_val, y_val, x_test, y_test, classes, pasta_saida
        )
        resultados.append(resultado)

    df = pd.DataFrame(resultados)
    df = df.sort_values(by="teste_f1_macro", ascending=False)
    df.to_csv(pasta_saida / "resultados_experimentos.csv", index=False)

    print("\n" + "=" * 80)
    print("RESULTADOS ORDENADOS POR F1-SCORE MACRO NO TESTE")
    print(
        df[
            [
                "experimento",
                "tecnica",
                "teste_acuracia",
                "teste_precisao_macro",
                "teste_recall_macro",
                "teste_f1_macro",
                "tempo_segundos",
            ]
        ].to_string(index=False)
    )

    melhor = df.iloc[0]
    print("\nMelhor experimento:")
    print(f"{melhor['experimento']} - F1 macro: {melhor['teste_f1_macro']:.4f}")
    print(f"Resultados salvos em: {pasta_saida.resolve()}")

if __name__ == "__main__":
    main()
