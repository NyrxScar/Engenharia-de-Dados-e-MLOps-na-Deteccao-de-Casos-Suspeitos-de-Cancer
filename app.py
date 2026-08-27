import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    ConfusionMatrixDisplay,
)

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E CSS CUSTOMIZADO (UX/UI MODERN)
# ==============================================================================
st.set_page_config(
    page_title="MLOps | Manutenção Preditiva Indústria 4.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS de Alta Performance Visual
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Tipografia Global */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
    }

    /* Container Principal */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Estilização dos Métricas (Cards de Destaque) */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stMetric"] label {
        font-size: 0.825rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Cards Informativos Customizados */
    .custom-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }

    .card-accent-blue { border-left: 4px solid #2563eb; }
    .card-accent-emerald { border-left: 4px solid #059669; }
    .card-accent-rose { border-left: 4px solid #e11d48; }

    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }

    .card-body {
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.5;
    }

    /* Estilização das Abas */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 8px !important;
        color: #64748b !important;
    }

    button[aria-selected="true"] {
        color: #2563eb !important;
        background-color: #eff6ff !important;
    }

    /* Ocultar elementos desnecessários */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Configuração de Estilo Global do Matplotlib para combinar com a UI
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Plus Jakarta Sans', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

# ==============================================================================
# 2. SIDEBAR INTERATIVA
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚡ MLOps Control Center")
    st.caption("Ajuste as premissas operacionais e financeiras em tempo real.")
    st.divider()

    st.markdown("**💰 Matriz de Custos Industriais**")
    COST_FALSE_NEGATIVE = st.number_input(
        "Falso Negativo (R$) [Quebra Catastrófica]",
        min_value=1000, max_value=200000, value=50000, step=5000,
        help="Custo direto de parada não planejada, dano ao equipamento e perda de produção."
    )

    COST_FALSE_POSITIVE = st.number_input(
        "Falso Positivo (R$) [Inspeção Preventiva]",
        min_value=100, max_value=10000, value=500, step=100,
        help="Custo operacional de deslocar a equipe técnica para verificação."
    )

    st.divider()
    st.markdown("**🔬 Configurações da Telemetria**")
    N_SAMPLES = st.select_slider(
        "Amostras de Sensores Simuladas",
        options=[5000, 10000, 20000, 50000],
        value=20000
    )

RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_THRESHOLD = 0.50
THRESHOLDS = np.arange(0.05, 0.96, 0.01)

# ==============================================================================
# 3. PIPELINE DE DADOS & FUNÇÕES AUXILIARES
# ==============================================================================

def export_to_excel_csv(df: pd.DataFrame) -> bytes:
    """Exporta o dataframe no formato correto para Excel em Português (UTF-8 com BOM)."""
    return df.to_csv(index=False, sep=";", decimal=",", encoding="utf-8-sig").encode("utf-8-sig")

@st.cache_data
def generate_industrial_data(n_samples=N_SAMPLES, random_state=RANDOM_STATE):
    X, y = make_classification(
        n_samples=n_samples, n_features=10, n_informative=8, n_redundant=2,
        n_repeated=0, n_clusters_per_class=2, weights=[0.995, 0.005],
        flip_y=0, class_sep=1.0, random_state=random_state
    )
    feature_names = [
        "sensor_vibracao_eixo_x", "sensor_vibracao_eixo_y", "sensor_vibracao_eixo_z",
        "temp_motor_principal", "pressao_hidraulica", "torque_braco_robotico",
        "rpm_esteira", "consumo_corrente_motor", "ruido_acustico_db", "fluxo_lubrificante"
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df, feature_names

@st.cache_data
def prepare_data_pipeline(df, feature_names):
    X = df[feature_names]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_metrics(y_true, y_probs, threshold=DEFAULT_THRESHOLD):
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Threshold": threshold, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Acuracia": accuracy_score(y_true, y_pred),
        "Precisao": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": fbeta_score(y_true, y_pred, beta=1.0, zero_division=0),
        "F2_Score": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
        "F0.5_Score": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    }

def analyze_thresholds(y_true, y_probs, fn_cost, fp_cost):
    results = []
    for t in THRESHOLDS:
        metrics = evaluate_metrics(y_true, y_probs, t)
        metrics["Custo_Total"] = (metrics["FP"] * fp_cost) + (metrics["FN"] * fn_cost)
        results.append(metrics)
    return pd.DataFrame(results)

# ==============================================================================
# 4. EXECUÇÃO DO PIPELINE
# ==============================================================================
df, feature_names = generate_industrial_data(N_SAMPLES)
X_train, X_test, y_train, y_test, scaler = prepare_data_pipeline(df, feature_names)
model = train_model(X_train, y_train)
y_probs_test = model.predict_proba(X_test)[:, 1]
auc_roc = roc_auc_score(y_test, y_probs_test)
results_df = analyze_thresholds(y_test, y_probs_test, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE)

best_row = results_df.loc[results_df["Custo_Total"].idxmin()]
default_row = results_df[np.isclose(results_df["Threshold"], DEFAULT_THRESHOLD)].iloc[0]
economy_abs = default_row['Custo_Total'] - best_row['Custo_Total']
economy_pct = (economy_abs / default_row['Custo_Total']) * 100

# ==============================================================================
# 5. HEADER & KPIS EXECUTIVOS
# ==============================================================================
st.markdown("## 🏭 Sistema Preditivo de Manutenção & MLOps")
st.markdown("<p style='color: #64748b; margin-top: -10px; margin-bottom: 25px;'>Otimização dinâmica de limiares de alarme para maximização do ROI fabril em ambientes Indústria 4.0</p>", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Custo Limiar Padrão (0.50)", f"R$ {default_row['Custo_Total']:,.2f}")
kpi2.metric("Custo Limiar Ótimo", f"R$ {best_row['Custo_Total']:,.2f}", delta=f"-R$ {economy_abs:,.2f}")
kpi3.metric("Economia Gerada", f"{economy_pct:.1f}%", delta="Redução de Impacto")
kpi4.metric("Capacidade Preditiva (AUC)", f"{auc_roc:.4f}")

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# 6. NAVEGAÇÃO E NÚCLEO DO DASHBOARD
# ==============================================================================
tab_finance, tab_metrics, tab_data, tab_arch = st.tabs([
    "💵 Otimização Financeira",
    "📈 Diagnósticos & Métricas",
    "📡 Telemetria & Sensores",
    "🏗️ Arquitetura MLOps"
])

# --- TAB 1: OTIMIZAÇÃO FINANCEIRA ---
with tab_finance:
    st.markdown("### Curva de Custo Operacional vs. Limiar de Alarme")
    
    c_graph, c_info = st.columns([2.2, 1])
    
    with c_graph:
        fig_cost, ax_cost = plt.subplots(figsize=(9, 4.5), facecolor='none')
        
        ax_cost.plot(results_df["Threshold"], results_df["Custo_Total"] / 1000, color='#2563eb', linewidth=2.5, label="Custo Total (em mil R$)")
        ax_cost.axvline(DEFAULT_THRESHOLD, linestyle="--", color='#e11d48', linewidth=1.5, label=f"Limiar Padrão (0.50): R$ {default_row['Custo_Total']/1000:,.0f}k")
        ax_cost.axvline(best_row["Threshold"], linestyle="--", color='#059669', linewidth=1.5, label=f"Limiar Ótimo ({best_row['Threshold']:.2f}): R$ {best_row['Custo_Total']/1000:,.0f}k")
        ax_cost.scatter(best_row["Threshold"], best_row["Custo_Total"] / 1000, color='#059669', s=90, zorder=5)
        
        ax_cost.set_xlabel("Limiar de Decisão (Decision Threshold)", fontsize=9, color='#475569')
        ax_cost.set_ylabel("Custo Operacional Total (R$ em Milhares)", fontsize=9, color='#475569')
        ax_cost.spines['top'].set_visible(False)
        ax_cost.spines['right'].set_visible(False)
        ax_cost.grid(True, linestyle=":", alpha=0.5, color='#cbd5e1')
        ax_cost.legend(frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=8.5)
        
        st.pyplot(fig_cost, use_container_width=True)

    with c_info:
        st.markdown(f"""
        <div class="custom-card card-accent-emerald">
            <div class="card-title">💡 Diagnóstico do Negócio</div>
            <div class="card-body">
                Com o limiar padrão de <b>0.50</b>, o modelo falha em detectar falhas incipientes devido ao forte desbalanceamento de classes, gerando <b>{int(default_row['FN'])} Falsos Negativos</b>.
                <br><br>
                Ajustando o limiar para <b>{best_row['Threshold']:.2f}</b>, priorizamos o Recall do sistema. Embora os Falsos Positivos aumentem, o menor custo de inspeção (R$ {COST_FALSE_POSITIVE:,.2f}) compensa drasticamente a prevenção de quebras (R$ {COST_FALSE_NEGATIVE:,.2f}).
                <br><br>
                <b>Economia Líquida:</b> <span style="color:#059669; font-weight:700;">R$ {economy_abs:,.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.download_button(
            label="📥 Exportar Relatório de Custos (CSV Excel)",
            data=export_to_excel_csv(results_df),
            file_name="relatorio_otimizacao_financeira.csv",
            mime="text/csv",
            use_container_width=True
        )

# --- TAB 2: DIAGNÓSTICOS & MÉTRICAS ---
with tab_metrics:
    st.markdown("### Performance Estatística e Curvas de Discriminação")
    
    col_roc, col_cm = st.columns(2)
    
    with col_roc:
        st.markdown("**Curva ROC (Receiver Operating Characteristic)**")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4), facecolor='none')
        fpr, tpr, _ = roc_curve(y_test, y_probs_test)
        
        ax_roc.plot(fpr, tpr, color='#2563eb', linewidth=2, label=f"Random Forest (AUC = {auc_roc:.4f})")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color='#94a3b8', alpha=0.7, label="Baseline Aleatório")
        ax_roc.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=8.5, color='#475569')
        ax_roc.set_ylabel("Taxa de Verdadeiros Positivos (TPR / Recall)", fontsize=8.5, color='#475569')
        ax_roc.spines['top'].set_visible(False)
        ax_roc.spines['right'].set_visible(False)
        ax_roc.grid(True, linestyle=":", alpha=0.5, color='#cbd5e1')
        ax_roc.legend(frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=8)
        
        st.pyplot(fig_roc, use_container_width=True)

    with col_cm:
        st.markdown("**Matrizes de Confusão Comparativas**")
        fig_cm, axes_cm = plt.subplots(1, 2, figsize=(8, 3.8), facecolor='none')
        
        cm_default = confusion_matrix(y_test, (y_probs_test >= DEFAULT_THRESHOLD).astype(int))
        cm_best = confusion_matrix(y_test, (y_probs_test >= best_row["Threshold"]).astype(int))
        
        ConfusionMatrixDisplay(cm_default, display_labels=["Normal", "Falha"]).plot(ax=axes_cm[0], colorbar=False, cmap='Blues')
        axes_cm[0].set_title(f"Limiar Padrão (0.50)\nFN={int(default_row['FN'])}, FP={int(default_row['FP'])}", fontsize=9, fontweight='600')
        
        ConfusionMatrixDisplay(cm_best, display_labels=["Normal", "Falha"]).plot(ax=axes_cm[1], colorbar=False, cmap='Greens')
        axes_cm[1].set_title(f"Limiar Ótimo ({best_row['Threshold']:.2f})\nFN={int(best_row['FN'])}, FP={int(best_row['FP'])}", fontsize=9, fontweight='600')
        
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)

    st.markdown("---")
    st.markdown("**Quadro Comparativo de Métricas**")
    
    summary_df = pd.DataFrame({
        "Métrica de Avaliação": [
            "Acurácia (Accuracy)", 
            "Precisão (Precision)", 
            "Recall (Sensibilidade)", 
            "F1-Score (Equilíbrio Padrão)", 
            "F2-Score (Foco em Minimizar Falsos Negativos)", 
            "F0.5-Score (Foco em Minimizar Falsos Positivos)", 
            "Custo Total Previsto (R$)"
        ],
        "Limiar Padrão (0.50)": [
            f"{default_row['Acuracia']:.4f}", f"{default_row['Precisao']:.4f}", f"{default_row['Recall']:.4f}", 
            f"{default_row['F1_Score']:.4f}", f"{default_row['F2_Score']:.4f}", f"{default_row['F0.5_Score']:.4f}",
            f"R$ {default_row['Custo_Total']:,.2f}"
        ],
        f"Limiar Ótimo ({best_row['Threshold']:.2f})": [
            f"{best_row['Acuracia']:.4f}", f"{best_row['Precisao']:.4f}", f"{best_row['Recall']:.4f}", 
            f"{best_row['F1_Score']:.4f}", f"{best_row['F2_Score']:.4f}", f"{best_row['F0.5_Score']:.4f}",
            f"R$ {best_row['Custo_Total']:,.2f}"
        ]
    })
    
    st.table(summary_df)

# --- TAB 3: TELEMETRIA & SENSORES ---
with tab_data:
    st.markdown("### Telemetria Bruta de Sensores Industriais")
    st.caption("Amostragem em tempo real enviada pelos ativos da planta fabril.")
    
    col_t1, col_t2 = st.columns([3, 1])
    
    with col_t1:
        st.dataframe(df.head(100), use_container_width=True, height=350)
    
    with col_t2:
        st.markdown("""
        <div class="custom-card card-accent-blue">
            <div class="card-title">📊 Perfil das Amostras</div>
            <div class="card-body">
                <b>Volume Total:</b> {:,}<br>
                <b>Operação Normal (0):</b> {:,} (99.5%)<br>
                <b>Eventos de Falha (1):</b> {:,} (0.5%)
            </div>
        </div>
        """.format(len(df), (df['target'] == 0).sum(), (df['target'] == 1).sum()), unsafe_allow_html=True)
        
        st.download_button(
            label="📥 Baixar Base Completa (CSV Excel)",
            data=export_to_excel_csv(df),
            file_name="telemetria_sensores_industriais.csv",
            mime="text/csv",
            use_container_width=True
        )

# --- TAB 4: ARQUITETURA MLOPS ---
with tab_arch:
    st.markdown("### Governança e Arquitetura MLOps para Produção")
    
    st.markdown("""
    <div class="custom-card card-accent-blue">
        <div class="card-title">🛡️ Prevenção Integrada de Data Leakage</div>
        <div class="card-body">
            Neste pipeline, o fracionamento dos dados entre treino (80%) e teste (20%) é efetuado de forma estratificada <b>rigorosamente antes</b> da fase de padronização. 
            O <code>StandardScaler</code> calcula os parâmetros médios e o desvio padrão estritamente no subconjunto de treinamento. Dessa forma, garante-se que nenhuma informação da distribuição de teste 'vaze' para o aprendizado do modelo.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown("""
        #### 1. Monitoramento de Data & Concept Drift
        - **Kolmogorov-Smirnov (KS-Test):** Testes contínuos aplicados às distribuições dos sensores vibracionais e térmicos para detectar desgaste físico mecânico.
        - **Gatilhos de Re-treinamento:** Disparo automático de pipelines no MLflow/Kubeflow assim que a métrica $F_2$-Score cair abaixo de 0.80 no ambiente de produção.
        """)
        
    with col_a2:
        st.markdown("""
        #### 2. Governança e Servibilidade de Borda
        - **Feature Store Centralizada:** As rotinas de transformação do `StandardScaler` são publicadas centralizadamente para prevenir divergências entre o treino offline e a inferência na borda (*Edge Computing*).
        - **Mecanismo de Safe Fallback:** Se a probabilidade estimada residir na zona cinzenta ($0.05 \le p \le 0.10$) ou ocorrer oscilação de conectividade, a máquina entra em modo de segurança preventivo automático.
        """)