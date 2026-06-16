import numpy as np

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
