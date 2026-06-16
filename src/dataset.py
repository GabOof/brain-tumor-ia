from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from features import carregar_imagem_cinza, vetorizar_imagem

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp"}

def carregar_pasta(
    pasta_base: Path,
    img_size: int = 64,
    max_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    imagens = []
    rotulos = []

    classes_encontradas = [p.name for p in pasta_base.iterdir() if p.is_dir()]
    classes_ordenadas = [c for c in CLASSES if c in classes_encontradas]

    if not classes_ordenadas:
        raise FileNotFoundError(
            f"Nenhuma classe esperada encontrada em {pasta_base}. Esperado: {CLASSES}"
        )

    print(f"Classes em {pasta_base.name}: {classes_ordenadas}")

    for indice_classe, nome_classe in enumerate(classes_ordenadas):
        pasta_classe = pasta_base / nome_classe
        caminhos = sorted(
            [p for p in pasta_classe.iterdir() if p.suffix.lower() in EXTENSOES_VALIDAS]
        )

        if max_per_class is not None:
            caminhos = caminhos[:max_per_class]

        print(f"Carregando {len(caminhos)} imagens da classe {nome_classe}...")

        for caminho in caminhos:
            try:
                imagem = carregar_imagem_cinza(caminho, img_size)
                imagens.append(vetorizar_imagem(imagem))
                rotulos.append(indice_classe)
            except Exception as erro:
                print(f"Aviso: erro ao carregar {caminho}: {erro}")

    x = np.array(imagens, dtype=np.float32)
    y = np.array(rotulos, dtype=np.int64)
    return x, y, classes_ordenadas

def carregar_dataset(
    data_dir: str = "data",
    img_size: int = 64,
    max_per_class: Optional[int] = None,
    validacao: float = 0.15,
    seed: int = 42,
):
    base = Path(data_dir)
    treino_dir = base / "Training"
    teste_dir = base / "Testing"

    if not treino_dir.exists() or not teste_dir.exists():
        raise FileNotFoundError(
            "As pastas data/Training e data/Testing não foram encontradas. "
            "Confira se o dataset foi baixado e extraído corretamente."
        )

    print("Carregando treino...")
    x_train_total, y_train_total, classes_treino = carregar_pasta(
        treino_dir, img_size=img_size, max_per_class=max_per_class
    )

    print("Carregando teste...")
    x_test, y_test, classes_teste = carregar_pasta(
        teste_dir, img_size=img_size, max_per_class=max_per_class
    )

    if classes_treino != classes_teste:
        print("Aviso: as classes de treino e teste não estão exatamente na mesma ordem.")

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_total,
        y_train_total,
        test_size=validacao,
        random_state=seed,
        stratify=y_train_total,
    )

    print(f"Treino: {x_train.shape} | Validação: {x_val.shape} | Teste: {x_test.shape}")

    return x_train, x_val, x_test, y_train, y_val, y_test, classes_treino
