# MLOps e Engenharia de Dados para Manutenção Preditiva

Projeto desenvolvido para a Unidade Curricular de **Engenharia de Dados e MLOps** (UniSENAI), focado na aplicação de Machine Learning para detecção de falhas operacionais na Indústria 4.0.

A solução contempla um pipeline completo que sintetiza dados de sensores industriais, previne *Data Leakage*, treina um modelo de classificação probabilístico, avalia o desempenho estatístico através de múltiplas métricas e executa a otimização do threshold de decisão baseada nos custos operacionais do negócio.

---

## 1. Escopo da Solução: Componentes Implementados e Propostos

Para garantir transparência técnica e alinhar a documentação com o código-fonte existente no repositório, as funcionalidades são divididas em duas categorias:

* **Componentes Efetivamente Implementados:** Código em Python (`.py`) executável para geração sintética de dados, divisão estratificada, padronização sem vazamento, treinamento do modelo, cálculo estatístico de métricas, otimização financeira por threshold, geração de relatório executivo em PDF e interface interativa em Streamlit.
* **Arquitetura Proposta para Produção (Recomendações MLOps):** Conceitos de governança visualizados na aba de arquitetura do dashboard — tais como monitoramento contínuo de *Data Drift* via KS-Test, gatilhos automáticos de retreinamento (MLflow/Kubeflow), *Feature Store* centralizada e rotinas de *Safe Fallback* — representam diretrizes para uma futura implantação em escala, não estando executáveis neste repositório.

---

## 2. Contexto e Objetivo

Em ambientes industriais, eventos de falha grave são raros em comparação com o tempo de operação normal. Contudo, suas consequências geram paradas não planejadas e custos elevados.

O objetivo deste projeto é construir um pipeline preditivo reprodutível focado no impacto financeiro das decisões do modelo.

Em bases altamente desbalanceadas, a **Acurácia isolada é uma métrica enganosa**: um modelo ingênuo que classifique 100% das amostras como "Operação Normal" obteria ~99,5% de acurácia, mas falharia em identificar todas as falhas reais. Por isso, a avaliação utiliza métricas focadas em sensibilidade, precisão e custos reais de classificação.

---

## 3. Telemetria e Amostragem de Dados

Os dados são sintetizados via `make_classification` (`scikit-learn`) com semente fixa (`random_state=42`), garantindo reprodutibilidade.

* **Proporção das Classes:** Fixada intencionalmente em **99,5% de Operação Normal (`target=0`)** e **0,5% de Eventos de Falha (`target=1`)** (`weights=[0.995, 0.005]`, `flip_y=0`).
* **Variação do Volume de Amostras:** O número total de registros pode ser alterado na aplicação (5.000, 10.000, 20.000 ou 50.000). A proporção relativa de falhas permanece em ~0,5%, mas a **quantidade absoluta** de falhas varia conforme o tamanho da base:
* 5.000 amostras $\approx$ 25 eventos de falha
* 20.000 amostras $\approx$ 100 eventos de falha
* 50.000 amostras $\approx$ 250 eventos de falha


* **Variáveis de Entrada (10 atributos contínuos):**
* `sensor_vibracao_eixo_x`, `sensor_vibracao_eixo_y`, `sensor_vibracao_eixo_z`
* `temp_motor_principal`, `pressao_hidraulica`, `torque_braco_robotico`
* `rpm_esteira`, `consumo_corrente_motor`, `ruido_acustico_db`, `fluxo_lubrificante`



---

## 4. Pipeline de Dados e Prevenção de Data Leakage

A prevenção de *Data Leakage* (vazamento de dados) foi estruturada no arquivo `preprocessing.py`.

```text
               [ Dataset Sintético Completo ]
                             │
                             ▼
              [ Train/Test Split (80/20) ]
              (Estratificado por Classe)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [ Conjunto de Treino ]           [ Conjunto de Teste ]
   (80% das Amostras)               (20% das Amostras)
            │                                 │
            ▼                                 │
[ Fit + Transform: StandardScaler ]           │
            │                                 │
            ▼                                 ▼
 [ Treinamento do Modelo ]        [ Transform: StandardScaler ]
 (RandomForestClassifier)         (Apenas parâmetros do Treino)
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
                 [ Avaliação Final de Teste ]

```

### Fluxo de Execução

1. **Divisão Estratificada Prévia:** Os dados brutos são divididos em 80% treino e 20% teste antes de qualquer transformação estatística.
2. **Ajuste Isolado do Scaler:** O `StandardScaler` executa o método `.fit_transform()` **apenas** no conjunto de treinamento para aprender a média ($\mu$) e o desvio padrão ($\sigma$).
3. **Transformação sem Vazamento:** O conjunto de teste é transformado via `.transform()` utilizando exclusivamente os parâmetros extraídos do treino.
4. **Isolamento de Avaliação:** O modelo é treinado estritamente com os dados de treino escalados. O conjunto de teste permanece oculto e é utilizado ao final apenas para geração das probabilidades e cálculo das métricas.

---

## 5. Algoritmo de Machine Learning

O modelo adotado no módulo `model.py` é o `RandomForestClassifier`.

### Justificativa da Escolha

Em vez de justificativas genéricas, a escolha do Random Forest se baseia nos requisitos específicos do projeto:

* **Relações Não-Lineares:** Capacidade intrínseca de mapear interações complexas entre múltiplos sensores industriais contínuos sem premissas de linearidade.
* **Tratamento de Desbalanceamento:** Suporte nativo ao parâmetro `class_weight="balanced"`, que ajusta os pesos das árvores de decisão de forma inversamente proporcional à frequência das classes.
* **Saída Probabilística:** Permite a extração de probabilidades estimadas via predict_proba(X_test)[:, 1], requisito fundamental para a análise e ajuste do threshold de decisão.

---

## 6. Framework de Avaliação Estatística

O cálculo das métricas é realizado em `metrics.py` e dividido entre métricas dependentes de threshold e métricas independentes.

### 6.1 Métricas Dependentes de Threshold

Para um limiar específico (ex: $0,50$), geram-se as previsões binárias e calcula-se a Matriz de Confusão:

* **Verdadeiro Positivo (TP):** Falha real identificada corretamente.
* **Verdadeiro Negativo (TN):** Operação normal identificada corretamente.
* **Falso Positivo (FP):** Alarme falso (operação normal classificada como falha).
* **Falso Negativo (FN):** Falha não detectada pelo modelo (quebra não prevista).

Métricas derivadas:

* **Acurácia:** Proporção total de acertos.
* **Precisão:** Taxa de acerto quando o modelo dispara um alarme de falha.
* **Recall (Sensibilidade):** Proporção de falhas reais que o modelo conseguiu capturar.
* **Métricas da Família $F_\beta$:**
* **$F_1$-Score ($\beta=1.0$):** Média harmônica balanceada entre Precisão e Recall.
* **$F_2$-Score ($\beta=2.0$):** Atribui maior peso ao Recall, sendo especialmente adequado para cenários em que a detecção de falhas reais é prioritária.
* **$F_{0.5}$-Score ($\beta=0.5$):** Atribui maior peso à **Precisão**, sendo indicada para cenários onde a redução de alarmes falsos é prioritária.



### 6.2 Métrica Independente de Threshold (AUC-ROC)

A função `evaluate_metrics()` retorna as métricas operacionais para um corte fixo. A área sob a curva (*AUC-ROC*) é calculada de forma independente diretamente sobre o vetor de probabilidades `y_probs_test` e os rótulos reais `y_test`. Isso avalia a capacidade geral do algoritmo em discriminar as classes em qualquer nível de sensibilidade.

---

## 7. Análise Financeira e Redução de Custos

A análise financeira (módulo `financial_analysis.py`) substitui estimativas genéricas de ROI por um modelo direto de **Custo Total por Erros de Classificação**.

### 7.1 Modelo de Custo Operacional

$$\text{Custo Total} = (FP \times \text{Custo}_{FP}) + (FN \times \text{Custo}_{FN})$$

Premissas padrão configuráveis:

* **Custo do Falso Negativo ($FN$): R$ 50.000,00** (Danos ao equipamento, parada não planejada da linha e perda de produção).
* **Custo do Falso Positivo ($FP$): R$ 500,00** (Inspeção técnica preventiva e checagem breve de rotina).

### 7.2 Otimização do Threshold

O algoritmo avalia o Custo Total em uma faixa de thresholds entre $0,05$ e $0,95$ (passo $0,01$). O threshold ótimo é aquele que minimiza o Custo Total da operação.

$$\text{Economia Absoluta (R\$)} = \text{Custo Total}_{\text{Threshold 0.50}} - \text{Custo Total}_{\text{Threshold Ótimo}}$$

$$\text{Economia Percentual (\%)} = \left( \frac{\text{Economia Absoluta}}{\text{Custo Total}_{\text{Threshold 0.50}}} \right) \times 100$$

---

## 8. Arquitetura de Código e Responsabilidades

```text
.
├── reports/                # Armazenamento dos relatórios em PDF gerados
├── app.py                  # Interface interativa em Streamlit (Módulo Complementar)
├── data_generator.py       # Geração de dados sintéticos de sensores industriais com proporção de 0,5% de falhas.
├── preprocessing.py        # Divisão estratificada e ajuste do StandardScaler
├── model.py                # Instanciação e treinamento do RandomForestClassifier
├── metrics.py              # Funções de cálculo de métricas estatísticas e AUC-ROC
├── financial_analysis.py   # Varredura de thresholds e minimização do custo operacional
├── generate_report.py      # Script de compilação do relatório executivo em PDF (fpdf2)
├── requirements.txt        # Lista de dependências do projeto
└── README.md               # Documentação técnica

```

| Arquivo | Responsabilidade Técnica |
| --- | --- |
| `data_generator.py` | Gera o dataset sintético controlando o volume e a proporção de $0,5\%$ de falhas. |
| `preprocessing.py` | Executa o *split* estratificado e o escalamento sem vazamento de dados. |
| `model.py` | Treina o Random Forest aplicando pesos balanceados para a classe minoritária. |
| `metrics.py` | Calcula métricas baseadas em matriz de confusão e calcula a curva ROC / AUC-ROC. |
| `financial_analysis.py` | Simula os custos operacionais por threshold e identifica o ponto de menor custo. |
| `generate_report.py` | Gera o relatório executivo em PDF no diretório `reports/`. |
| `app.py` | Interface gráfica complementar em Streamlit para exploração visual e interativa. |

---

## 9. Interfaces de Saída: Dashboard e Relatório PDF

### 9.1 Dashboard Interativo em Streamlit (`app.py`)

Desenvolvido como **interface complementar** para facilitar a exploração dos resultados:

* **Aba 1 (Análise Financeira):** Gráficos interativos da curva de custo por threshold e simulação de custos $FN$/$FP$ em tempo real.
* **Aba 2 (Diagnósticos):** Curva ROC, matrizes de confusão comparativas e tabela completa com métricas $F_\beta$.
* **Aba 3 (Telemetria):** Visualização dos dados brutos e exportação tratada para Excel.
* **Aba 4 (Arquitetura MLOps):** Apresentação visual das diretrizes propostas para produção.

### 9.2 Relatório Executivo em PDF (`generate_report.py`)

Script autônomo que gera um arquivo PDF profissional em `reports/relatorio_executivo_mlops.pdf` utilizando a biblioteca `fpdf2`. O documento consolida gráficos, tabelas e pareceres operacionais para tomada de decisão.

---

## 10. Instruções de Instalação e Execução

### 10.1 Pré-requisitos

* Python 3.10 ou superior
* Gerenciador de pacotes `pip`

### 10.2 Configuração do Ambiente

```bash
# 1. Clonar o repositório
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>

# 2. Criar ambiente virtual
python -m venv .venv

# 3. Ativar o ambiente virtual
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# 4. Instalar as dependências
pip install -r requirements.txt

```

### 10.3 Executando os Scripts

* **Gerar o Relatório Executivo em PDF:**
```bash
python generate_report.py

```


*(O arquivo será gerado na pasta `reports/`)*
* **Iniciar o Dashboard Complementar em Streamlit:**
```bash
streamlit run app.py

```



---

## 11. Formatação na Exportação de Dados (CSV)

Os arquivos gerados para download na interface utilizam o padrão compatível com planilhas regionalizadas (Excel em português):

* **Separador de Colunas:** Ponto e vírgula (`;`)
* **Separador Decimal:** Vírgula (`,`)
* **Encoding:** UTF-8 com BOM (`utf-8-sig`)

---

## 12. Arquitetura MLOps Proposta para Produção (Recomendações)

As diretrizes abaixo não representam scripts executáveis no código atual, mas constituem a visão de arquitetura para a implantação do modelo em ambiente produtivo:

1. **Monitoramento de Data Drift:** Aplicação de testes estatísticos contínuos (como *Kolmogorov-Smirnov*) na telemetria dos sensores para detectar desvios de distribuição causados pelo desgaste mecânico dos ativos.
2. **Pipelines de Retreinamento:** Configuração de acionamentos automáticos no Orchestrator (Kubeflow/Airflow) caso a métrica $F_2$-Score caia abaixo de $0,80$ em ambiente de produção.
3. **Feature Store e Servibilidade:** Centralização dos parâmetros de transformação (`StandardScaler`) em uma *Feature Store* para garantir paridade exata entre o treinamento offline e a inferência de baixa latência (*Edge Computing*).
4. **Mecanismo de Safe Fallback:** Protocolo operacional para colocar a máquina em modo preventivo se o modelo indicar incerteza elevada ou se houver perda de sinal na coleta dos sensores.

---

## 13. Matriz de Atendimento aos Requisitos

| Requisito do Trabalho | Estado da Implementação | Detalhes do Atendimento |
| --- | --- | --- |
| **Ambiente de Engenharia de Dados e MLOps** | **Implementado** | Estrutura modular em Python, ambiente virtual, gerenciamento de dependências e separação das etapas do pipeline. |
| **Geração de Dados Sintéticos** | **Implementado** | Simulador de telemetria com proporção ajustada para $0,5\%$ de falhas. |
| **Isolamento e Sem Data Leakage** | **Implementado** | *Split* estratificado executado antes do `.fit()` do `StandardScaler`. |
| **Modelo Preditivo Probabilístico** | **Implementado** | `RandomForestClassifier` com pesagem `class_weight="balanced"`. |
| **Cálculo de Múltiplas Métricas** | **Implementado** | Acurácia, Precisão, Recall, $F_1$, $F_2$ e $F_{0.5}$ em `metrics.py`. |
| **Curva ROC e AUC-ROC** | **Implementado** | Avaliação calculada de forma independente através das probabilidades. |
| **Análise de Custos e Threshold** | **Implementado** | Minimização do Custo Total ($FN$ vs $FP$) e cálculo de economia. |
| **Relatório Executivo em PDF** | **Implementado** | Gerador automático funcional via `generate_report.py` em `reports/`. |
| **Exportação Compatível Excel** | **Implementado** | Exportação tratada com `;`, `,` e codificação UTF-8-BOM. |
| **Interface Visual (Streamlit)** | **Implementado** | Desenvolvido como ferramenta complementar interativa. |
| **Monitoramento de Drift (KS-Test)** | **Proposto** | Diretriz descrita na seção MLOps e no dashboard. |
| **Retreinamento Automatizado** | **Proposto** | Arquitetura MLOps recomendada para integração com MLflow/Kubeflow. |
| **Feature Store & Safe Fallback** | **Proposto** | Recomendação estrutural para mitigação de riscos operacionais. |
