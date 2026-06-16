from pathlib import Path

import numpy as np
from PIL import Image

def carregar_imagem_cinza(caminho: Path, img_size: int) -> np.ndarray:
    imagem = Image.open(caminho).convert("L")
    imagem = imagem.resize((img_size, img_size))
    array = np.asarray(imagem, dtype=np.float32) / 255.0
    return array

def vetorizar_imagem():
    # TODO: Implementar função para vetorizar imagem
    pass
