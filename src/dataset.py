from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

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
