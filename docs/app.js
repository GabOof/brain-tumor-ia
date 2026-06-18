const btnExecutar = document.getElementById("btnExecutar");
const btnAtualizar = document.getElementById("btnAtualizar");
const maxPerClass = document.getElementById("maxPerClass");
const tabelaResultados = document.getElementById("tabelaResultados");
const terminalOutput = document.getElementById("terminalOutput");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const statusDetail = document.getElementById("statusDetail");
const melhorExperimento = document.getElementById("melhorExperimento");
const melhorAcuracia = document.getElementById("melhorAcuracia");
const melhorF1 = document.getElementById("melhorF1");

function formatarNumero(valor) {
    const numero = Number(valor);
    if (Number.isNaN(numero)) return "-";
    return numero.toFixed(4);
}

function setStatus(tipo, texto, detalhe) {
    statusDot.className = "status-dot";
    if (tipo) statusDot.classList.add(tipo);
    statusText.textContent = texto;
    statusDetail.textContent = detalhe;
}

async function carregarResultados() {
    const resposta = await fetch("/api/resultados");
    const dados = await resposta.json();

    if (!dados.existe || !dados.resultados.length) {
        tabelaResultados.innerHTML = `
            <tr>
                <td colspan="7" class="empty">${dados.mensagem}</td>
            </tr>
        `;
        melhorExperimento.textContent = "-";
        melhorAcuracia.textContent = "-";
        melhorF1.textContent = "-";
        return;
    }

    const resultados = dados.resultados;
    const melhor = resultados[0];

    melhorExperimento.textContent = melhor.experimento;
    melhorAcuracia.textContent = formatarNumero(melhor.teste_acuracia);
    melhorF1.textContent = formatarNumero(melhor.teste_f1_macro);

    tabelaResultados.innerHTML = resultados
        .map(
            (linha, index) => `
            <tr class="${index === 0 ? "best" : ""}">
                <td>${linha.experimento}</td>
                <td>${linha.tecnica}</td>
                <td>${formatarNumero(linha.teste_acuracia)}</td>
                <td>${formatarNumero(linha.teste_precisao_macro)}</td>
                <td>${formatarNumero(linha.teste_recall_macro)}</td>
                <td>${formatarNumero(linha.teste_f1_macro)}</td>
                <td>${formatarNumero(linha.tempo_segundos)}</td>
            </tr>
        `
        )
        .join("");
}

async function executarExperimentos() {
    btnExecutar.disabled = true;
    btnAtualizar.disabled = true;
    terminalOutput.textContent =
        "Executando experimentos... Isso pode demorar na execução completa.\n";
    setStatus("running", "Executando", "O script Python está rodando os experimentos.");

    try {
        const resposta = await fetch("/api/executar", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ max_per_class: maxPerClass.value }),
        });

        const dados = await resposta.json();

        terminalOutput.textContent = [
            dados.mensagem || "Execução finalizada.",
            "",
            "===== STDOUT =====",
            dados.stdout || "Sem saída padrão.",
            "",
            "===== STDERR =====",
            dados.stderr || "Sem erros.",
        ].join("\n");

        if (dados.ok) {
            setStatus("ok", "Finalizado", "Resultados atualizados com sucesso.");
            await carregarResultados();
        } else {
            setStatus("error", "Erro", "Confira a saída do terminal na tela.");
        }
    } catch (error_) {
        terminalOutput.textContent = `Erro ao executar: ${error_.message}`;
        setStatus("error", "Erro", "Falha na comunicação com o servidor Flask.");
    } finally {
        btnExecutar.disabled = false;
        btnAtualizar.disabled = false;
    }
}

btnExecutar.addEventListener("click", executarExperimentos);
btnAtualizar.addEventListener("click", carregarResultados);

carregarResultados();
