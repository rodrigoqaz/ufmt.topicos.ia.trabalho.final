# Fine-Tuning de LLMs com QLoRA e Unsloth: Experimentos de Eficiência

Este repositório contém os códigos, notebooks e artefatos desenvolvidos como requisito para a avaliação da disciplina de **Tópicos Avançados em Inteligência Artificial** do Programa de Pós-Graduação em Computação (PPGComp) do Instituto de Computação da Universidade Federal de Mato Grosso (IC/UFMT).

## Autoria

Trabalho desenvolvido por:

- Matheus Cândido Teixeira (matheuscandido2009@gmail.com)
- Rodrigo de Souza Oliveira (rddsouzaoliveira@usp.br)


## Resumo do Projeto

O objetivo deste trabalho é explorar técnicas de ajuste fino eficiente de parâmetros (PEFT - Parameter-Efficient Fine-Tuning) em Grandes Modelos de Linguagem (LLMs). O foco principal reside na utilização da biblioteca **Unsloth** para otimização de memória e velocidade de treinamento, aplicando a técnica QLoRA (Quantized Low-Rank Adaptation).

O cenário de aplicação definido é a tarefa de geração de títulos acadêmicos a partir de resumos (abstracts), configurando um problema de geração de texto condicional (*instruction tuning*).

## Metodologia e Experimentos

O modelo base selecionado para os experimentos foi o **Qwen/Qwen2.5-0.5B-Instruct**, devido à sua eficiência e desempenho comprovado em benchmarks recentes para modelos de menor porte.

Foram conduzidos dois experimentos principais para avaliar o impacto da profundidade da adaptação no desempenho do modelo:

### 1. Baseline (Zero-Shot)
Neste cenário de controle, o modelo original (`Qwen/Qwen2.5-0.5B-Instruct`) foi utilizado diretamente para inferência, sem qualquer etapa de treinamento ou atualização de pesos.
* **Objetivo:** Estabelecer uma linha de base (*baseline*) para verificar a capacidade nativa do modelo em seguir as instruções de formatação e sumarização, servindo de referência para quantificar o ganho real obtido com o fine-tuning.

### 2. Fine-Tuning Convencional (Full LoRA)
Neste cenário, adaptadores LoRA foram injetados e treinados em todos os módulos lineares da arquitetura do Transformer (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Objetivo:** Avaliar o desempenho máximo possível utilizando a técnica QLoRA, permitindo que o modelo tenha alta plasticidade para se adaptar à tarefa específica e ao estilo dos dados.

### 3. Fine-Tuning da Última Camada (Last Layer Adaptation)
Neste cenário experimental, a aplicação dos adaptadores LoRA foi restrita exclusivamente à última camada do decodificador do Transformer.
* **Implementação:** Utilização do parâmetro `layers_to_transform` na configuração do PEFT.
* **Hipótese:** Investigar se a adaptação apenas das representações finais é suficiente para tarefas de formatação e sumarização, visando reduzir drasticamente o custo computacional e preservar o conhecimento generalista das camadas anteriores.

## Estrutura do Repositório

* **`ppgcomp_FT_QLoRA_Unsloth.ipynb`**: Notebook principal contendo o pipeline completo (carregamento do modelo, pré-processamento dos dados, configuração do Unsloth/PEFT, loop de treinamento e inferência).
* **`datasets/`**: Diretório contendo os arquivos de dados (formato JSON/JSONL) utilizados para treinamento supervisionado e validação.
* **`output_unsloth/`**: Diretório de saída para os checkpoints do modelo e adaptadores salvos (geralmente não versionado via Git devido ao tamanho).

## Stack Tecnológico

* **Linguagem:** Python 3.10+
* **Framework de Otimização:** Unsloth
* **Bibliotecas de ML/DL:**
    * PyTorch
    * Hugging Face Transformers & PEFT
    * TRL
    * BitsAndBytes
* **Métrica de Avaliação:** BERTScore (para medir similaridade semântica entre títulos gerados e títulos de referência).

## Instruções de Reprodução

Os experimentos foram configurados para execução em ambiente com GPU do Google Colab com T4.
Execute as células sequencialmente para realizar o fine-tuning e visualizar as métricas de avaliação.
