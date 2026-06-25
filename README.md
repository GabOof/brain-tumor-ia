# Detecção de Tumores Cerebrais com Inteligência Artificial

Projeto prático desenvolvido para a disciplina de **Inteligência Artificial** do curso de **Bacharelado em Sistemas de Informação** — IFMG Campus Ouro Branco.

## Tema

O projeto aplica técnicas de Aprendizado de Máquina para classificar imagens de ressonância magnética cerebral em quatro classes:

- Glioma;
- Meningioma;
- Tumor hipofisário;
- Sem tumor.

O objetivo é comparar diferentes modelos de classificação e avaliar qual abordagem apresenta melhor desempenho na identificação das classes.

---

## Dataset utilizado

O dataset utilizado foi o **Brain Tumor MRI Dataset**, disponível no Kaggle.

A base contém imagens de ressonância magnética cerebral divididas em quatro categorias:

- `glioma`
- `meningioma`
- `pituitary`
- `notumor`

O dataset deve ser baixado manualmente no Kaggle e colocado na raiz do projeto dentro da pasta `data`.

A estrutura esperada é:

```text
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

---

## Técnicas utilizadas

Foram implementadas duas técnicas principais de Aprendizado de Máquina:

### 1. SVM

A técnica **SVM**, ou Support Vector Machine, foi utilizada com diferentes configurações de kernel e parâmetro `C`.

Foram testadas variações com:

- Kernel linear;
- Kernel RBF;
- Diferentes valores para o parâmetro `C`.

### 2. MLP

A técnica **MLP**, ou Multilayer Perceptron, foi utilizada como uma rede neural simples.

Foram testadas variações com:

- Diferentes quantidades de neurônios;
- Uma ou duas camadas ocultas;
- Regularização L2 por meio do parâmetro `alpha`;
- Early stopping para reduzir overfitting.

---

## Experimentos realizados

Foram realizados 8 experimentos no total.

| Experimento        | Técnica | Configuração                                 |
| ------------------ | ------- | -------------------------------------------- |
| E01_SVM_linear_C1  | SVM     | Kernel linear, C = 1.0                       |
| E02_SVM_linear_C01 | SVM     | Kernel linear, C = 0.1                       |
| E03_SVM_rbf_C1     | SVM     | Kernel RBF, C = 1.0                          |
| E04_SVM_rbf_C10    | SVM     | Kernel RBF, C = 10.0                         |
| E05_MLP_128        | MLP     | Uma camada oculta com 128 neurônios          |
| E06_MLP_256        | MLP     | Uma camada oculta com 256 neurônios          |
| E07_MLP_256_128    | MLP     | Duas camadas ocultas com 256 e 128 neurônios |
| E08_MLP_512_256    | MLP     | Duas camadas ocultas com 512 e 256 neurônios |

---

## Métricas avaliadas

Os modelos foram avaliados utilizando as seguintes métricas:

- Acurácia;
- Precisão;
- Recall;
- F1-score.

Como o problema possui quatro classes, foi utilizada a média macro para precisão, recall e F1-score.

O F1-score foi utilizado como principal critério para definir o melhor experimento, pois essa métrica equilibra precisão e recall.

---

## Resultados obtidos

Após a execução completa dos experimentos, o melhor resultado foi obtido pelo modelo:

```text
E04_SVM_rbf_C10
```

Esse experimento utilizou:

- Técnica: SVM;
- Kernel: RBF;
- Parâmetro C: 10.0.

Resultados no conjunto de teste:

| Métrica  |  Valor |
| -------- | -----: |
| Acurácia | 0,8725 |
| Precisão | 0,8746 |
| Recall   | 0,8725 |
| F1-score | 0,8703 |

O modelo SVM com kernel RBF apresentou melhor desempenho porque consegue criar fronteiras de decisão não lineares, o que é importante para problemas com imagens.

---

## Como executar o projeto

### 1. Criar o ambiente virtual

No Linux ou WSL:

```bash
python -m venv .venv
source .venv/bin/activate
```

No Windows PowerShell:

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

Caso não exista o arquivo `requirements.txt`, instale manualmente:

```bash
pip install numpy pandas pillow scikit-learn flask
```

---

### 3. Baixar o dataset

Baixe o dataset **Brain Tumor MRI Dataset** no Kaggle.

Depois, extraia os arquivos e coloque a pasta `data` na raiz do projeto.

A estrutura deve ficar assim:

```text
brain-tumor-ia/
├── data/
│   ├── Training/
│   └── Testing/
├── main.py
└── app.py
```

---

### 4. Rodar teste rápido

Para testar com poucas imagens por classe:

```bash
python main.py --max-per-class 10
```

Esse comando é útil para verificar se o projeto está funcionando sem precisar executar o dataset completo.

---

### 5. Rodar experimento completo

Para executar o projeto com todas as imagens:

```bash
python main.py
```

Os resultados serão salvos na pasta:

```text
resultados/
```

O principal arquivo gerado será:

```text
resultados/resultados_experimentos.csv
```

---

## Como executar a interface web

O projeto também possui uma interface web local feita com Flask.

Para iniciar a aplicação web, execute:

```bash
python app.py
```

Depois, abra no navegador:

```text
http://127.0.0.1:5000
```

Importante: o projeto deve ser aberto pelo endereço do Flask, na porta `5000`.

Não utilize o Live Server do VSCode, pois ele abre na porta `5500` e não executa as rotas da API Flask.
