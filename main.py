import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ============================================================
# Configurações principais
# ============================================================
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
EXTENSOES_VALIDAS = [".jpg", ".jpeg", ".png", ".bmp"]
IMG_SIZE = 64
DATA_DIR = "data"
PASTA_RESULTADOS = "resultados"
SEED = 42
np.random.seed(SEED)


# ============================================================
# Carregamento e pré-processamento das imagens
# ============================================================
def carregar_imagem(caminho):
    # Transforma a imagem em escala de cinza
    imagem = Image.open(caminho).convert("L")

    # Redimensiona a imagem para 64x64 pixels
    imagem = imagem.resize((IMG_SIZE, IMG_SIZE))

    # dtype=np.float32 reduz custo computacional
    # Divide por 255 para normalizar os pixels para a faixa entre 0 e 1
    array = np.array(imagem, dtype=np.float32) / 255.0

    # Transforma a imagem vira um vetor com 4096 atributos (64x64)
    vetor = array.reshape(-1)

    return vetor


def carregar_pasta(pasta_base, max_por_classe=None):
    imagens = []
    rotulos = []

    pasta_base = Path(pasta_base)

    # Percorre todas as classes definidas na lista CLASSES
    for indice_classe, nome_classe in enumerate(CLASSES):
        # Monta o caminho da pasta correspondente à classe atual
        pasta_classe = pasta_base / nome_classe

        # Verifica se a pasta da classe existe
        if not pasta_classe.exists():
            print(f"Aviso: pasta não encontrada: {pasta_classe}")
            continue

        # Lista temporária para armazenar os caminhos das imagens da classe
        caminhos = []

        # Percorre todos os arquivos dentro da pasta da classe
        for arquivo in pasta_classe.iterdir():
            # Verifica se a extensão do arquivo é uma imagem válida
            if arquivo.suffix.lower() in EXTENSOES_VALIDAS:
                caminhos.append(arquivo)

        # Ordena os caminhos para manter uma leitura padronizada
        caminhos = sorted(caminhos)

        # Mantém apenas as primeiras imagens até o limite definido
        if max_por_classe is not None:
            caminhos = caminhos[:max_por_classe]

        # Exibe no terminal quantas imagens serão carregadas da classe atual
        print(f"Carregando {len(caminhos)} imagens da classe {nome_classe}...")

        # Percorre cada imagem da classe atual
        for caminho in caminhos:
            try:
                # Carrega e pré-processa a imagem
                vetor = carregar_imagem(caminho)

                # Adiciona o vetor da imagem à lista de imagens
                imagens.append(vetor)

                # Adiciona o rótulo numérico correspondente à classe
                rotulos.append(indice_classe)

            except Exception as erro:
                print(f"Erro ao carregar {caminho}: {erro}")

    # Converte a lista de imagens em array
    x = np.array(imagens, dtype=np.float32)

    # Converte a lista de rótulos em array
    y = np.array(rotulos, dtype=np.int64)

    # Retorna os dados e seus respectivos rótulos
    return x, y


# ============================================================
# Métricas de avaliação
# ============================================================
def calcular_metricas(y_real, y_predito):
    acuracia = accuracy_score(y_real, y_predito)
    precisao = precision_score(y_real, y_predito, average="macro", zero_division=0)
    recall = recall_score(y_real, y_predito, average="macro", zero_division=0)
    f1 = f1_score(y_real, y_predito, average="macro", zero_division=0)
    return acuracia, precisao, recall, f1


# ============================================================
# Execução de cada experimento
# ============================================================
def executar_experimento(nome, tecnica, modelo, x_train, y_train, x_val, y_val, x_test, y_test):
    print("\n" + "=" * 70)
    print(f"Executando experimento: {nome}")
    print(f"Técnica utilizada: {tecnica}")

    inicio = time.time()

    # Treina o modelo usando os dados de treino
    modelo.fit(x_train, y_train)

    # Realiza predições no conjunto de validação
    pred_val = modelo.predict(x_val)

    # Realiza predições no conjunto de teste
    pred_test = modelo.predict(x_test)

    # Calcula as métricas no conjunto de validação
    val_acuracia, val_precisao, val_recall, val_f1 = calcular_metricas(y_val, pred_val)

    # Calcula as métricas no conjunto de teste
    teste_acuracia, teste_precisao, teste_recall, teste_f1 = calcular_metricas(y_test, pred_test)

    # Calcula o tempo total gasto no treinamento e avaliação
    tempo = time.time() - inicio

    # Exibe um relatório detalhado de classificação no conjunto de teste
    print("\nRelatório no conjunto de teste:")
    print(classification_report(y_test, pred_test, target_names=CLASSES, zero_division=0))

    # Cria um dicionário com os resultados do experimento
    resultado = {
        "experimento": nome,
        "tecnica": tecnica,
        "val_acuracia": val_acuracia,
        "val_precisao": val_precisao,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "teste_acuracia": teste_acuracia,
        "teste_precisao": teste_precisao,
        "teste_recall": teste_recall,
        "teste_f1": teste_f1,
        "tempo_segundos": round(tempo, 2),
    }

    return resultado


# ============================================================
# Programa principal
# ============================================================
def main():
    # Cria o parser de argumentos do terminal
    parser = argparse.ArgumentParser(
        description="Classificação de tumores cerebrais usando SVM e MLP."
    )

    # Adiciona o argumento opcional --max-per-class
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Limite de imagens por classe. Use para testes rápidos.",
    )

    # Lê os argumentos informados pelo usuário no terminal
    args = parser.parse_args()

    # Cria a pasta de resultados
    Path(PASTA_RESULTADOS).mkdir(exist_ok=True)

    # Define o caminho da pasta de treino
    treino_dir = Path(DATA_DIR) / "Training"

    # Define o caminho da pasta de teste
    teste_dir = Path(DATA_DIR) / "Testing"

    if not treino_dir.exists() or not teste_dir.exists():
        print("Erro: as pastas data/Training e data/Testing não foram encontradas.")
        return

    # Carrega as imagens de treino
    print("Carregando imagens de treino...")
    x_total, y_total = carregar_pasta(treino_dir, max_por_classe=args.max_per_class)

    # Carrega as imagens de teste
    print("\nCarregando imagens de teste...")
    x_test, y_test = carregar_pasta(teste_dir, max_por_classe=args.max_per_class)

    if len(x_total) == 0 or len(x_test) == 0:
        print("Erro: nenhuma imagem foi carregada.")
        return

    # Divide o conjunto de treino em treino e validação
    # test_size=0.15 reserva 15% para validação
    # random_state garante reprodutibilidade
    # stratify mantém a proporção das classes na divisão
    x_train, x_val, y_train, y_val = train_test_split(
        x_total, y_total, test_size=0.15, random_state=SEED, stratify=y_total
    )

    print("\nFormato dos dados:")
    print("Treino:", x_train.shape)
    print("Validação:", x_val.shape)
    print("Teste:", x_test.shape)

    # ========================================================
    # Lista de experimentos
    # ========================================================
    experimentos = []

    # --------------------------------------------------------
    # Experimentos com SVM
    # --------------------------------------------------------
    experimentos.append(
        (
            "E01_SVM_linear_C1",
            "SVM",
            Pipeline([("scaler", StandardScaler()), ("modelo", SVC(kernel="linear", C=1.0))]),
        )
    )

    experimentos.append(
        (
            "E02_SVM_linear_C01",
            "SVM",
            Pipeline([("scaler", StandardScaler()), ("modelo", SVC(kernel="linear", C=0.1))]),
        )
    )

    experimentos.append(
        (
            "E03_SVM_rbf_C1",
            "SVM",
            Pipeline(
                [("scaler", StandardScaler()), ("modelo", SVC(kernel="rbf", C=1.0, gamma="scale"))]
            ),
        )
    )

    experimentos.append(
        (
            "E04_SVM_rbf_C10",
            "SVM",
            Pipeline(
                [("scaler", StandardScaler()), ("modelo", SVC(kernel="rbf", C=10.0, gamma="scale"))]
            ),
        )
    )

    # --------------------------------------------------------
    # Experimentos com MLP
    # --------------------------------------------------------
    experimentos.append(
        (
            "E05_MLP_128",
            "MLP",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "modelo",
                        MLPClassifier(
                            hidden_layer_sizes=(128,),
                            activation="relu",
                            solver="adam",
                            alpha=0.0001,
                            learning_rate_init=0.001,
                            max_iter=120,
                            early_stopping=True,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
        )
    )

    experimentos.append(
        (
            "E06_MLP_256",
            "MLP",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "modelo",
                        MLPClassifier(
                            hidden_layer_sizes=(256,),
                            activation="relu",
                            solver="adam",
                            alpha=0.0001,
                            learning_rate_init=0.001,
                            max_iter=120,
                            early_stopping=True,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
        )
    )

    experimentos.append(
        (
            "E07_MLP_256_128",
            "MLP",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "modelo",
                        MLPClassifier(
                            hidden_layer_sizes=(256, 128),
                            activation="relu",
                            solver="adam",
                            alpha=0.001,
                            learning_rate_init=0.001,
                            max_iter=150,
                            early_stopping=True,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
        )
    )

    experimentos.append(
        (
            "E08_MLP_512_256",
            "MLP",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "modelo",
                        MLPClassifier(
                            hidden_layer_sizes=(512, 256),
                            activation="relu",
                            solver="adam",
                            alpha=0.01,
                            learning_rate_init=0.0005,
                            max_iter=150,
                            early_stopping=True,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
        )
    )

    # ========================================================
    # Executar todos os experimentos
    # ========================================================
    resultados = []

    # Percorre todos os experimentos configurados
    for nome, tecnica, modelo in experimentos:
        # Executa o experimento atual.
        resultado = executar_experimento(
            nome, tecnica, modelo, x_train, y_train, x_val, y_val, x_test, y_test
        )

        # Adiciona o resultado à lista geral
        resultados.append(resultado)

    # ========================================================
    # Salvar resultados em CSV
    # ========================================================

    # Converte a lista de resultados para um DataFrame
    df = pd.DataFrame(resultados)

    # Ordena os resultados pelo F1-score de teste em ordem decrescente
    df = df.sort_values(by="teste_f1", ascending=False)

    # Define o caminho do arquivo CSV
    caminho_csv = Path(PASTA_RESULTADOS) / "resultados_experimentos.csv"

    # Salva os resultados em um arquivo CSV
    df.to_csv(caminho_csv, index=False)

    # Exibe os resultados finais no terminal
    print("\n" + "=" * 70)
    print("RESULTADOS FINAIS ORDENADOS PELO F1-SCORE")
    print(df.to_string(index=False))

    # Seleciona o melhor experimento
    melhor = df.iloc[0]

    # Exibe as principais métricas do melhor experimento
    print("\nMelhor experimento encontrado:")
    print(f"Experimento: {melhor['experimento']}")
    print(f"Técnica: {melhor['tecnica']}")
    print(f"Acurácia: {melhor['teste_acuracia']:.4f}")
    print(f"Precisão: {melhor['teste_precisao']:.4f}")
    print(f"Recall: {melhor['teste_recall']:.4f}")
    print(f"F1-score: {melhor['teste_f1']:.4f}")

    print(f"\nTabela salva em: {caminho_csv}")


if __name__ == "__main__":
    main()
