# Detecção de Tumores Cerebrais com IA

Projeto prático da disciplina de Inteligência Artificial — IFMG Campus Ouro Branco.

## Tema
Classificação de imagens de ressonância magnética cerebral em quatro classes:

- glioma
- meningioma
- pituitary
- notumor

O dataset utilizado é o **Brain Tumor MRI Dataset**, disponível no Kaggle.

## Técnicas testadas
Foram implementados 8 experimentos, envolvendo duas técnicas principais:

1. SVM, com variações de kernel e parâmetro C.
2. MLP, com variações de quantidade de neurônios, regularização L2 e early stopping.

## Métricas avaliadas

- Acurácia
- Precisão macro
- Recall macro
- F1-score macro

## Estrutura do projeto

```text
brain_tumor_ia_projeto/
├── data/                         # dataset baixado do Kaggle, não sobe para o GitHub
├── resultados/                   # métricas e relatórios gerados
├── src/
│   ├── dataset.py                # carregamento e pré-processamento das imagens
│   ├── features.py               # funções auxiliares para imagens
│   └── treinar_experimentos.py   # execução dos 8 experimentos
├── relatorio/
│   └── relatorio_modelo.md       # modelo de relatório para preencher
├── video/
│   └── roteiro_video.md          # roteiro sugerido para gravação
├── requirements.txt
├── .gitignore
└── README.md
```

## Como rodar

### 1. Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/WSL
# .venv\Scripts\activate        # Windows PowerShell
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Baixar o dataset

Entre no Kaggle, crie um token de API em **Account > Create New API Token** e coloque o arquivo `kaggle.json` em:

Linux/WSL:

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Windows:

```text
C:\Users\SEU_USUARIO\.kaggle\kaggle.json
```

Depois, rode:

```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip
```

A pasta `data` deve ficar parecida com:

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

### 4. Teste rápido

Use poucas imagens para validar se tudo está funcionando:

```bash
python src/treinar_experimentos.py --max-per-class 10
```

### 5. Rodar experimento completo

```bash
python src/treinar_experimentos.py
```

Os arquivos serão salvos em `resultados/`:

- `resultados_experimentos.csv`
- relatórios individuais de classificação
- matrizes de confusão

## Observação

Os resultados podem variar conforme o computador, divisão dos dados, quantidade de épocas e versão das bibliotecas.

## Tela web local

O projeto também possui uma interface web simples feita com Flask.

### Instalar dependências

```bash
pip install -r requirements.txt
```

### Rodar a tela web

```bash
python app.py
```

Depois, abra no navegador:

```text
http://127.0.0.1:5000
```

Na tela, é possível:

- executar os experimentos pelo botão;
- usar `max_per_class` para teste rápido;
- atualizar e visualizar a tabela de resultados;
- consultar a saída do terminal;
- ver o melhor experimento, acurácia e F1-score macro.
