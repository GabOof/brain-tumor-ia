const btnExecutar = document.getElementById("btnExecutar");
const btnAtualizar = document.getElementById("btnAtualizar");
const maxPerClass = document.getElementById("maxPerClass");
const tabelaResultados = document.getElementById("tabelaResultados");
const mensagem = document.getElementById("mensagem");
const saidaPrograma = document.getElementById("saidaPrograma");

function formatarNumero(valor) {
    const numero = Number(valor);

    if (Number.isNaN(numero)) {
        return "-";
    }

    return numero.toFixed(4);
}

function pegarValor(linha, campoNovo, campoAntigo) {
    return linha[campoNovo] || linha[campoAntigo] || "-";
}

async function carregarResultados() {
    const resposta = await fetch("/api/resultados");
    const dados = await resposta.json();

    if (!dados.ok || dados.resultados.length === 0) {
        tabelaResultados.innerHTML = `
            <tr>
                <td colspan="7">${dados.mensagem}</td>
            </tr>
        `;

        return;
    }

    tabelaResultados.innerHTML = "";

    dados.resultados.forEach((linha, indice) => {
        const testeAcuracia = pegarValor(linha, "teste_acuracia", "teste_acuracia");
        const testePrecisao = pegarValor(linha, "teste_precisao", "teste_precisao_macro");
        const testeRecall = pegarValor(linha, "teste_recall", "teste_recall_macro");
        const testeF1 = pegarValor(linha, "teste_f1", "teste_f1_macro");

        const tr = document.createElement("tr");

        if (indice === 0) {
            tr.classList.add("melhor");
        }

        tr.innerHTML = `
            <td>${linha.experimento}</td>
            <td>${linha.tecnica}</td>
            <td>${formatarNumero(testeAcuracia)}</td>
            <td>${formatarNumero(testePrecisao)}</td>
            <td>${formatarNumero(testeRecall)}</td>
            <td>${formatarNumero(testeF1)}</td>
            <td>${formatarNumero(linha.tempo_segundos)}</td>
        `;

        tabelaResultados.appendChild(tr);
    });

    mensagem.textContent = "Resultados carregados.";
}

async function executarExperimentos() {
    btnExecutar.disabled = true;
    btnAtualizar.disabled = true;

    mensagem.textContent = "Executando experimentos...";
    saidaPrograma.textContent = "Aguarde. O Python está treinando os modelos.";

    const dadosEnvio = {
        max_per_class: maxPerClass.value,
    };

    try {
        const resposta = await fetch("/api/executar", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(dadosEnvio),
        });

        const dados = await resposta.json();

        mensagem.textContent = dados.mensagem;
        saidaPrograma.textContent = dados.saida || "Sem saída do programa.";

        if (dados.ok) {
            await carregarResultados();
        }
    } catch (error_) {
        mensagem.textContent = "Erro ao comunicar com o servidor.";
        saidaPrograma.textContent = error_.message;
    }

    btnExecutar.disabled = false;
    btnAtualizar.disabled = false;
}

btnExecutar.addEventListener("click", executarExperimentos);
btnAtualizar.addEventListener("click", carregarResultados);

carregarResultados();
