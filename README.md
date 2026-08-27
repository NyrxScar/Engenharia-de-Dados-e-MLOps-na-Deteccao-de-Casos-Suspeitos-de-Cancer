# 🏭 MLOps Indústria 4.0 — Manutenção Preditiva

Dashboard desenvolvido para a disciplina **Engenharia de Dados e MLOps**, com foco na aplicação de Machine Learning para **detecção de falhas em equipamentos industriais** em um cenário de dados altamente desbalanceados.

O projeto simula dados de sensores industriais com **99,5% de operações normais e 0,5% de falhas**, demonstrando como métricas tradicionais podem ser insuficientes em cenários de falhas raras e como o **ajuste do threshold de decisão baseado em custos financeiros** pode apoiar decisões operacionais.

> **Observação:** os dados utilizados neste projeto são sintéticos e foram gerados exclusivamente para fins acadêmicos e de demonstração do pipeline.

---

## 📋 Objetivos

O projeto tem como objetivos:

* Gerar um conjunto de dados sintéticos representando sensores industriais;
* Simular um cenário altamente desbalanceado de **99,5% de operações normais e 0,5% de falhas**;
* Construir um pipeline reprodutível de Machine Learning;
* Evitar **Data Leakage** durante o pré-processamento;
* Treinar um modelo de **Random Forest** para classificação de falhas;
* Avaliar o modelo utilizando diferentes métricas de classificação;
* Analisar a **Matriz de Confusão** e seus componentes TP, TN, FP e FN;
* Calcular Acurácia, Precisão, Recall, F1, F2, F0.5 e AUC-ROC;
* Avaliar diferentes thresholds de decisão;
* Simular custos financeiros associados a falsos positivos e falsos negativos;
* Identificar o threshold de menor custo operacional;
* Demonstrar práticas de monitoramento e governança relacionadas a MLOps;
* Gerar um Relatório Executivo em PDF com os resultados consolidados.

---

## 📁 Estrutura do Projeto

```text
Engenharia-de-Dados-e-MLOps/
│
├── app.py                  # Dashboard interativo (Streamlit)
├── data_generator.py       # Geração de dados sintéticos desbalanceados
├── preprocessing.py        # Pré-processamento com prevenção de Data Leakage
├── model.py                # Treinamento do Random Forest
├── metrics.py              # Métricas de classificação (Acurácia, Precisão, Recall, Fβ, AUC-ROC)
├── financial_analysis.py   # Análise financeira e ajuste dinâmico do threshold
├── generate_report.py      # Geração do Relatório Executivo em PDF
│
├── requirements.txt        # Dependências do projeto
├── README.md               # Documentação (este arquivo)
├── LICENSE                 # Licença
│
└── reports/                # Relatório PDF e gráficos gerados
    ├── relatorio_executivo_mlops.pdf
    ├── curva_roc.png
    ├── matrizes_confusao.png
    └── custo_vs_threshold.png
```

---

## 🏗️ Arquitetura da Solução

O fluxo geral da solução é:

```text
┌─────────────────────────────┐
│ Dados Sintéticos             │
│ Sensores Industriais         │
│ 99,5% Normal / 0,5% Falha   │
│ (data_generator.py)          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Train/Test Split             │
│ 80% Treino / 20% Teste       │
│ Stratified                   │
│ (preprocessing.py)           │
└──────────────┬──────────────┘
               │
               ├──────────────────────┐
               │                      │
               ▼                      ▼
       ┌───────────────┐       ┌───────────────┐
       │ Dados Treino  │       │ Dados Teste   │
       └───────┬───────┘       └───────────────┘
               │
               ▼
       ┌───────────────┐
       │ Pré-process.  │
       │ StandardScaler│
       │ (fit no treino)│
       └───────┬───────┘
               │
               ▼
       ┌───────────────┐
       │ Random Forest │
       │ Classifier    │
       │ (model.py)    │
       └───────┬───────┘
               │
               ▼
       ┌───────────────┐
       │ Probabilidade │
       │ de Falha      │
       └───────┬───────┘
               │
               ▼
       ┌───────────────┐
       │ Métricas      │
       │ (metrics.py)  │
       └───────┬───────┘
               │
               ▼
       ┌──────────────────────┐
       │ Análise Financeira   │
       │ Ajuste de Threshold  │
       │(financial_analysis.py)│
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │ Relatório PDF        │
       │ (generate_report.py) │
       └──────────────────────┘
```

---

## 🔒 Prevenção de Data Leakage

Uma das principais preocupações do pipeline é impedir que informações do conjunto de teste influenciem o treinamento ou o pré-processamento do modelo.

O conjunto de dados é inicialmente dividido em:

* **80% para treinamento**;
* **20% para teste**.

Somente depois dessa divisão o `StandardScaler` é aplicado.

O scaler é ajustado exclusivamente utilizando os dados de treinamento:

```python
scaler.fit_transform(X_train)
```

Enquanto os dados de teste são apenas transformados utilizando os parâmetros aprendidos no treinamento:

```python
scaler.transform(X_test)
```

Dessa forma, informações estatísticas do conjunto de teste não são utilizadas durante o treinamento.

---

## 🤖 Modelo de Machine Learning

O modelo utilizado é um:

**Random Forest Classifier**

Configuração principal:

```python
RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

O parâmetro:

```python
class_weight="balanced"
```

é utilizado devido ao forte desbalanceamento entre as classes.

A classe representa:

```text
0 → Operação normal
1 → Falha
```

---

## 📊 Dados Sintéticos

Os dados são gerados utilizando `make_classification` do Scikit-Learn.

São simuladas 10 variáveis relacionadas a sensores industriais:

| Variável                 | Representação            |
| ------------------------ | ------------------------ |
| `sensor_vibracao_eixo_x` | Vibração no eixo X       |
| `sensor_vibracao_eixo_y` | Vibração no eixo Y       |
| `sensor_vibracao_eixo_z` | Vibração no eixo Z       |
| `temp_motor_principal`   | Temperatura do motor     |
| `pressao_hidraulica`     | Pressão hidráulica       |
| `torque_braco_robotico`  | Torque do braço robótico |
| `rpm_esteira`            | Rotação da esteira       |
| `consumo_corrente_motor` | Consumo de corrente      |
| `ruido_acustico_db`      | Ruído acústico           |
| `fluxo_lubrificante`     | Fluxo de lubrificante    |

A distribuição das classes utilizada na simulação é:

```text
99,5% → Operação normal
 0,5% → Falha
```

O `random_state=42` garante a reprodutibilidade da geração dos dados.

---

## 📐 Métricas de Avaliação

Devido ao forte desbalanceamento das classes, o projeto não utiliza apenas Acurácia.

São avaliadas as seguintes métricas:

### Acurácia

Representa a proporção total de classificações corretas.

```text
Acurácia = (TP + TN) / (TP + TN + FP + FN)
```

Em datasets altamente desbalanceados, uma Acurácia elevada pode ser enganosa.

### Precisão

Mede a proporção das previsões positivas que realmente representam falhas.

```text
Precisão = TP / (TP + FP)
```

### Recall

Mede a capacidade do modelo de identificar as falhas existentes.

```text
Recall = TP / (TP + FN)
```

Em manutenção preditiva, essa métrica é especialmente relevante porque falsos negativos podem representar falhas não identificadas.

### F1-Score

Equilibra Precisão e Recall.

```text
F1 = 2 × (Precisão × Recall) / (Precisão + Recall)
```

### F2-Score

Dá maior peso ao Recall.

É útil em cenários nos quais deixar uma falha passar é mais prejudicial do que gerar um alerta adicional.

### F0.5-Score

Dá maior peso à Precisão.

Pode ser utilizado para avaliar cenários nos quais falsos alarmes também representam um custo relevante.

### AUC-ROC

A AUC-ROC mede a capacidade do modelo de separar as classes considerando diferentes thresholds de decisão.

---

## 🎯 Ajuste de Threshold

O modelo gera uma probabilidade de falha para cada amostra.

Por padrão, utiliza-se:

```text
Threshold = 0,50
```

Porém, o projeto testa thresholds entre:

```text
0,05 e 0,95
```

em intervalos de 0,01.

Para cada threshold são calculados:

* TP;
* TN;
* FP;
* FN;
* Acurácia;
* Precisão;
* Recall;
* F1;
* F2;
* F0.5;
* Custo Total.

O threshold escolhido é aquele que apresenta o **menor custo financeiro total** dentro dos thresholds avaliados.

---

## 💰 Modelo Financeiro

O projeto considera que diferentes tipos de erro possuem diferentes impactos financeiros.

### Falso Negativo

Um falso negativo ocorre quando:

```text
Falha real → Modelo classifica como normal
```

Esse cenário representa uma falha não detectada e possui um custo maior.

### Falso Positivo

Um falso positivo ocorre quando:

```text
Operação normal → Modelo classifica como falha
```

Esse cenário pode representar uma inspeção ou parada preventiva desnecessária.

### Custo Total

O custo é calculado por:

```text
Custo Total =
(FP × Custo do Falso Positivo)
+
(FN × Custo do Falso Negativo)
```

Os custos podem ser configurados diretamente no painel lateral do dashboard.

O sistema então compara o custo obtido pelo threshold padrão `0,50` com o threshold de menor custo.

---

## 📈 Dashboard

O dashboard desenvolvido em Streamlit possui quatro áreas principais:

### 💰 Análise Financeira

Apresenta:

* Custo no threshold 0,50;
* Custo no threshold ótimo;
* Economia gerada;
* Gráfico de custo versus threshold;
* Download dos resultados da análise.

### 📈 Diagnósticos e Curva ROC

Apresenta:

* Curva ROC;
* AUC-ROC;
* Matrizes de confusão;
* Comparação entre threshold padrão e threshold ótimo;
* TP, TN, FP e FN;
* Quadro comparativo das métricas.

### 📊 Exploração dos Dados

Apresenta:

* Amostras dos dados sintéticos;
* Quantidade total de amostras;
* Quantidade de amostras normais;
* Quantidade de falhas;
* Download do dataset.

### 🛠️ Arquitetura MLOps

Apresenta recomendações para utilização do modelo em produção, incluindo:

* Monitoramento de Data Drift;
* Testes estatísticos;
* Estratégia de retreinamento;
* Feature Store e governança;
* Protocolo de fallback.

---

## 📄 Relatório Executivo em PDF

O relatório executivo pode ser gerado automaticamente pelo script `generate_report.py` e contém:

* **Arquitetura da Solução**: Explicação do pipeline e como o isolamento de dados preveniu o Data Leakage.
* **Quadro Comparativo de Métricas**: Tabela com Acurácia, Matriz de Confusão, Precisão, Recall, F1, F2, F0.5 e AUC-ROC.
* **Curva AUC-ROC**: Gráfico da capacidade discriminativa do modelo.
* **Matrizes de Confusão**: Comparativo visual entre threshold padrão e ótimo.
* **Análise Financeira do Threshold**: Gráfico do Custo Total vs. Threshold de Decisão, com destaque para o ponto de menor custo e economia em R$.
* **Recomendações MLOps**: Plano de monitoramento, retreinamento, governança e fallback.

Para gerar o relatório:

```bash
python generate_report.py
```

O PDF será salvo em `reports/relatorio_executivo_mlops.pdf`.

---

## 🛡️ Recomendações MLOps

Para uma aplicação real em ambiente industrial, recomenda-se:

### 1. Monitoramento de Data Drift

Monitorar alterações na distribuição das variáveis de entrada, como:

* vibração;
* temperatura;
* pressão;
* corrente;
* ruído.

Testes estatísticos, como Kolmogorov-Smirnov, podem ser utilizados para identificar mudanças relevantes.

### 2. Retreinamento

O modelo pode ser retreinado periodicamente ou quando indicadores de desempenho apresentarem degradação.

Uma possível regra de negócio é utilizar o F2-Score como indicador de acompanhamento devido à importância do Recall no cenário de manutenção preditiva.

### 3. Governança

As transformações utilizadas no treinamento e na inferência devem ser mantidas de forma consistente, garantindo que o modelo receba as mesmas características durante produção e treinamento.

### 4. Fallback

Em caso de indisponibilidade do modelo ou problemas nos sensores, deve existir um procedimento operacional de segurança para evitar que uma falha de infraestrutura de Machine Learning comprometa a operação industrial.

---

## ⚙️ Instalação

### 1. Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```

### 2. Criar ambiente virtual

No Windows:

```bash
python -m venv .venv
```

### 3. Ativar o ambiente virtual

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Caso esteja utilizando o Prompt de Comando:

```cmd
.venv\Scripts\activate
```

### 4. Instalar as dependências

```bash
pip install -r requirements.txt
```

---

## ▶️ Execução

### Dashboard Interativo

Com o ambiente virtual ativado, execute:

```bash
streamlit run app.py
```

O Streamlit iniciará o dashboard localmente.

### Geração do Relatório PDF

```bash
python generate_report.py
```

O relatório será salvo na pasta `reports/`.

---

## 🔁 Reprodutibilidade

O projeto utiliza `random_state=42` na geração dos dados e no particionamento entre treino e teste.

Isso permite reproduzir o mesmo cenário experimental quando executado sob as mesmas condições e versões das dependências.

---

## 📚 Tecnologias Utilizadas

* **Python**
* **Streamlit**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-Learn**
* **FPDF2** (geração de PDF)
* **Git**
* **VSCode**

---

## 🎓 Contexto Acadêmico

Projeto desenvolvido para a Unidade Curricular:

**Engenharia de Dados e MLOps**

**Professor:** Prof. MSc. Hugo Menezes Barra

**Tema:** Machine Learning aplicado à detecção de falhas em sensores industriais e otimização do threshold de decisão considerando impactos financeiros.

---

## ⚠️ Limitações

Este projeto utiliza dados sintéticos para simulação acadêmica.

Consequentemente, os nomes das variáveis representam sensores industriais hipotéticos e não correspondem necessariamente a medições físicas reais.

Para utilização em ambiente industrial real seriam necessários:

* dados históricos reais;
* definição operacional de falha;
* validação com especialistas;
* calibração dos custos;
* monitoramento em produção;
* validação temporal;
* infraestrutura de inferência;
* governança e versionamento dos modelos.

---
