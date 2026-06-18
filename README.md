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

Entre no Kaggle e faça o download do dataset "Brain Tumor MRI Dataset". Extraia o conteúdo e coloque a pasta `data` na raiz do projeto.

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
