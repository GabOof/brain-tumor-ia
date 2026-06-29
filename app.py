import csv
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Define a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent

# Define o caminho dos arquivos necessários
SCRIPT_TREINO = BASE_DIR / "main.py"
CSV_RESULTADOS = BASE_DIR / "resultados" / "resultados_experimentos.csv"

#  Informa onde estão os templates e arquivos estáticos
app = Flask(
    __name__,
    template_folder=str(BASE_DIR),
    static_folder=str(BASE_DIR),
    static_url_path="",
)


# Define a rota principal do sistema
@app.route("/")
def pagina_inicial():
    return render_template("index.html")


# Define a rota dos resultados dos experimentos
@app.route("/api/resultados")
def buscar_resultados():
    # Verifica se o arquivo CSV existe
    if not CSV_RESULTADOS.exists():
        return jsonify({"ok": False, "mensagem": "Nenhum resultado encontrado.", "resultados": []})

    resultados = []

    # Abre o arquivo CSV em modo leitura
    with open(CSV_RESULTADOS, newline="", encoding="utf-8") as arquivo:
        # Lê cada linha do CSV como um dicionário
        leitor = csv.DictReader(arquivo)

        # Percorre cada linha do arquivo CSV
        for linha in leitor:
            # Adiciona a linha lida na lista de resultados
            resultados.append(linha)

    # Retorna os resultados em formato JSON
    return jsonify(
        {"ok": True, "mensagem": "Resultados carregados com sucesso.", "resultados": resultados}
    )


# Define a rota que executa os experimentos
@app.route("/api/executar", methods=["POST"])
def executar_experimentos():
    # Lê os dados enviados
    dados = request.get_json() or {}

    # Captura o valor de max_per_class
    limite = str(dados.get("max_per_class", "")).strip()

    # Monta o comando base para executar a main
    comando = [sys.executable, str(SCRIPT_TREINO)]

    # Verifica se existe limite de imagens
    if limite:
        # Valida se é inteiro
        if not limite.isdigit():
            return jsonify({"ok": False, "mensagem": "O limite deve ser um número inteiro."})

        # Adiciona o parâmetro --max-per-class ao comando
        comando.extend(["--max-per-class", limite])

    # Executa a main como um processo externo
    processo = subprocess.run(comando, cwd=BASE_DIR, capture_output=True, text=True)

    if processo.returncode == 0:
        return jsonify(
            {
                "ok": True,
                "mensagem": "Experimentos executados com sucesso.",
                # Retorna a saída para aparecer na tela
                "saida": processo.stdout,
            }
        )

    return jsonify(
        {
            "ok": False,
            "mensagem": "Erro ao executar os experimentos.",
            "saida": processo.stderr,
        }
    )


if __name__ == "__main__":
    app.run()
