import csv
import subprocess
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
SCRIPT_TREINO = BASE_DIR / "src" / "treinar_experimentos.py"
CSV_RESULTADOS = BASE_DIR / "resultados" / "resultados_experimentos.csv"

app = Flask(
    __name__,
    template_folder="docs",
    static_folder="docs",
    static_url_path="",
)

execucao_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/resultados")
def api_resultados():
    if not CSV_RESULTADOS.exists():
        return jsonify(
            {
                "existe": False,
                "mensagem": "Nenhum resultado encontrado. Execute os experimentos primeiro.",
                "resultados": [],
            }
        )

    with open(CSV_RESULTADOS, newline="", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        resultados = list(leitor)

    return jsonify(
        {
            "existe": True,
            "mensagem": "Resultados carregados com sucesso.",
            "resultados": resultados,
        }
    )


@app.route("/api/executar", methods=["POST"])
def api_executar():
    if not execucao_lock.acquire(blocking=False):
        return jsonify(
            {
                "ok": False,
                "mensagem": "Já existe uma execução em andamento. Aguarde finalizar.",
            }
        ), 409

    try:
        dados = request.get_json(silent=True) or {}
        max_per_class = str(dados.get("max_per_class", "")).strip()

        comando = [sys.executable, str(SCRIPT_TREINO)]

        if max_per_class:
            if not max_per_class.isdigit() or int(max_per_class) <= 0:
                return jsonify(
                    {
                        "ok": False,
                        "mensagem": "O campo max_per_class deve ser um número inteiro positivo.",
                    }
                ), 400

            comando.extend(["--max-per-class", max_per_class])

        processo = subprocess.run(
            comando,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
        )

        return jsonify(
            {
                "ok": processo.returncode == 0,
                "codigo_retorno": processo.returncode,
                "mensagem": "Execução finalizada."
                if processo.returncode == 0
                else "A execução terminou com erro.",
                "stdout": processo.stdout,
                "stderr": processo.stderr,
            }
        )

    finally:
        execucao_lock.release()


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
