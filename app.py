import csv
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
SCRIPT_TREINO = BASE_DIR / "main.py"
CSV_RESULTADOS = BASE_DIR / "resultados" / "resultados_experimentos.csv"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR),
    static_folder=str(BASE_DIR),
    static_url_path="",
)


@app.route("/")
def pagina_inicial():
    return render_template("index.html")


@app.route("/api/resultados")
def buscar_resultados():
    if not CSV_RESULTADOS.exists():
        return jsonify({"ok": False, "mensagem": "Nenhum resultado encontrado.", "resultados": []})

    resultados = []

    with open(CSV_RESULTADOS, newline="", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)

        for linha in leitor:
            resultados.append(linha)

    return jsonify(
        {"ok": True, "mensagem": "Resultados carregados com sucesso.", "resultados": resultados}
    )


@app.route("/api/executar", methods=["POST"])
def executar_experimentos():
    dados = request.get_json() or {}

    limite = str(dados.get("max_per_class", "")).strip()

    comando = [sys.executable, str(SCRIPT_TREINO)]

    if limite:
        if not limite.isdigit():
            return jsonify({"ok": False, "mensagem": "O limite deve ser um número inteiro."})

        comando.extend(["--max-per-class", limite])

    processo = subprocess.run(comando, cwd=BASE_DIR, capture_output=True, text=True)

    if processo.returncode == 0:
        return jsonify(
            {
                "ok": True,
                "mensagem": "Experimentos executados com sucesso.",
                "saida": processo.stdout,
            }
        )

    return jsonify(
        {"ok": False, "mensagem": "Erro ao executar os experimentos.", "saida": processo.stderr}
    )


if __name__ == "__main__":
    app.run(debug=True)
