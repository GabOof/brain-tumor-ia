// inicializa variáveis capturando os elementos HTML pelos IDs
const btnExecutar = document.getElementById("btnExecutar");
const btnAtualizar = document.getElementById("btnAtualizar");
const maxPerClass = document.getElementById("maxPerClass");
const tabelaResultados = document.getElementById("tabelaResultados");
const mensagem = document.getElementById("mensagem");
const saidaPrograma = document.getElementById("saidaPrograma");

// Formata valores numéricos para 4 casas decimais
function formatarNumero(valor) {
    const numero = Number(valor);

    // Verifica se é um número inválido
    if (Number.isNaN(numero)) {
        return "-";
    }

    return numero.toFixed(4);
}

// Recupera valores de uma linha de resultado
function pegarValor(linha, campoNovo, campoAntigo) {
    return linha[campoNovo] || linha[campoAntigo] || "-";
}

// Carrega os resultados dos experimentos na tabela HTML
async function carregarResultados() {
    // Busca os resultados salvos no servidor
    const resposta = await fetch("/api/resultados");

    // Converte a resposta para JSON
    const dados = await resposta.json();

    // Verifica se deu erro ou se não há resultados.
    if (!dados.ok || dados.resultados.length === 0) {
        // Cria linhas na tabela para os resultados
        tabelaResultados.innerHTML = `
            <tr>
                <td colspan="7">${dados.mensagem}</td>
            </tr>
        `;

        return;
    }

    // Limpa o conteúdo atigo da tabela
    tabelaResultados.innerHTML = "";

    // Percorre cada resultado retornado pela API
    dados.resultados.forEach((linha, indice) => {
        // Recupera as métricas do teste
        const testeAcuracia = pegarValor(linha, "teste_acuracia", "teste_acuracia");
        const testePrecisao = pegarValor(linha, "teste_precisao", "teste_precisao_macro");
        const testeRecall = pegarValor(linha, "teste_recall", "teste_recall_macro");
        const testeF1 = pegarValor(linha, "teste_f1", "teste_f1_macro");

        // Cria uma nova linha da tabela.
        const tr = document.createElement("tr");

        // Adiciona uma classe especial à primeira linha, que representa o melhor resultado
        if (indice === 0) {
            tr.classList.add("melhor");
        }

        // Preenche as linha da tabela com os dados do experimento
        tr.innerHTML = `
            <td>${linha.experimento}</td>
            <td>${linha.tecnica}</td>
            <td>${formatarNumero(testeAcuracia)}</td>
            <td>${formatarNumero(testePrecisao)}</td>
            <td>${formatarNumero(testeRecall)}</td>
            <td>${formatarNumero(testeF1)}</td>
            <td>${formatarNumero(linha.tempo_segundos)}</td>
        `;

        // Adiciona a linha criada na tabela
        tabelaResultados.appendChild(tr);
    });

    mensagem.textContent = "Resultados carregados.";
}

// Iniciar a execução dos experimentos
async function executarExperimentos() {
    // Desabilita os botões
    btnExecutar.disabled = true;
    btnAtualizar.disabled = true;

    // Atualiza as mensagens da interface
    mensagem.textContent = "Executando experimentos...";
    saidaPrograma.textContent = "Aguarde. O Python está treinando os modelos.";

    // Define o número máximo de imagens por classe usadas no experimento
    const dadosEnvio = {
        max_per_class: maxPerClass.value,
    };

    try {
        // Inicia o experimento
        const resposta = await fetch("/api/executar", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(dadosEnvio),
        });

        // Converte a resposta para JSON
        const dados = await resposta.json();

        // Exibe a mensagem
        mensagem.textContent = dados.mensagem;

        // Exibe a saída do terminal
        saidaPrograma.textContent = dados.saida || "Sem saída do programa.";

        if (dados.ok) {
            await carregarResultados();
        }
    } catch (error_) {
        mensagem.textContent = "Erro ao comunicar com o servidor.";
        saidaPrograma.textContent = error_.message;
    }

    // Reabilita os botões
    btnExecutar.disabled = false;
    btnAtualizar.disabled = false;
}

btnExecutar.addEventListener("click", executarExperimentos);
btnAtualizar.addEventListener("click", carregarResultados);

// Carrega automaticamente os resultados
await carregarResultados();
