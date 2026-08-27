# generate_report.py — Geração do Relatório Executivo em PDF

import os
import io
import datetime
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from fpdf import FPDF

from data_generator import generate_industrial_data
from preprocessing import prepare_pipeline
from model import train_model, predict_probabilities
from metrics import evaluate_metrics, compute_auc_roc
from financial_analysis import analyze_thresholds, compute_savings

# Constantes

N_SAMPLES = 20_000
RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_THRESHOLD = 0.50
COST_FN = 50_000.0
COST_FP = 500.0
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
PDF_FILENAME = "relatorio_executivo_mlops.pdf"


def save_figure_to_path(fig, path, dpi=150):
    """Salva uma figura matplotlib em disco."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


class ReportPDF(FPDF):

    FONTS_DIR = r"C:\Windows\Fonts"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_font("Arial", "", os.path.join(self.FONTS_DIR, "arial.ttf"))
        self.add_font("Arial", "B", os.path.join(self.FONTS_DIR, "arialbd.ttf"))
        self.add_font("Arial", "I", os.path.join(self.FONTS_DIR, "ariali.ttf"))
        self.add_font("Arial", "BI", os.path.join(self.FONTS_DIR, "arialbi.ttf"))

    def header(self):
        self.set_font("Arial", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(
            0, 8,
            "MLOps Industria 4.0 - Relatorio Executivo de Manutencao Preditiva",
            align="C",
        )
        self.ln(4)
        self.set_draw_color(37, 99, 235)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Pagina {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 14)
        self.set_text_color(15, 23, 42)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(37, 99, 235)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(4)

    def section_title(self, title):
        self.set_font("Arial", "B", 11)
        self.set_text_color(30, 58, 138)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Arial", "", 10)
        self.set_text_color(51, 65, 85)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def add_highlight_box(self, text, r=240, g=249, b=255, border_r=37, border_g=99, border_b=235):
        self.set_fill_color(r, g, b)
        self.set_draw_color(border_r, border_g, border_b)
        self.set_line_width(0.3)
        x = self.get_x()
        y = self.get_y()
        w = self.w - 20
        self.set_font("Arial", "", 9)
        self.set_text_color(30, 58, 138)
        # Calcular altura necessaria
        lines = self.multi_cell(w - 10, 5, text, dry_run=True, output="LINES")
        h = max(len(lines) * 5 + 8, 16)
        self.rect(x, y, w, h, style="DF")
        self.set_xy(x + 5, y + 4)
        self.multi_cell(w - 10, 5, text)
        self.set_y(y + h + 4)


def generate_report():
    print("=" * 60)
    print("  GERACAO DO RELATORIO EXECUTIVO EM PDF")
    print("=" * 60)

    # Pipeline de dados
    print("\n[1/6] Gerando dados sinteticos de sensores industriais...")
    df, feature_names = generate_industrial_data(N_SAMPLES, RANDOM_STATE)
    n_normal = (df["target"] == 0).sum()
    n_falha = (df["target"] == 1).sum()
    print(f"       Total: {len(df):,} | Normal: {n_normal:,} | Falha: {n_falha:,}")

    # 2. Pré-processamento (com isolamento contra Data Leakage)
    print("[2/6] Pre-processamento (split -> scale, sem Data Leakage)...")
    X_train, X_test, y_train, y_test, scaler = prepare_pipeline(
        df, feature_names, TEST_SIZE, RANDOM_STATE
    )
    print(f"       Treino: {len(y_train):,} | Teste: {len(y_test):,}")

    # 3. Treinamento do modelo
    print("[3/6] Treinando Random Forest (n_estimators=200, balanced)...")
    rf_model = train_model(X_train, y_train, random_state=RANDOM_STATE)
    y_probs = predict_probabilities(rf_model, X_test)

    # 4. Métricas
    print("[4/6] Calculando metricas de classificacao...")
    auc, fpr, tpr = compute_auc_roc(y_test, y_probs)
    metrics_default = evaluate_metrics(y_test, y_probs, DEFAULT_THRESHOLD)
    print(f"       AUC-ROC: {auc:.4f}")

    # 5. Análise financeira de thresholds
    print("[5/6] Analise financeira de thresholds (0.05 a 0.95)...")
    results_df = analyze_thresholds(y_test, y_probs, COST_FN, COST_FP)
    best_row, default_row, economia_abs, economia_pct = compute_savings(
        results_df, DEFAULT_THRESHOLD
    )
    print(f"       Threshold otimo: {best_row['Threshold']:.2f}")
    print(f"       Economia: R$ {economia_abs:,.2f} ({economia_pct:.1f}%)")

    print("[6/6] Gerando relatorio PDF...")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Gráfico 1: Curva ROC
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4.5))
    ax_roc.plot(fpr, tpr, label=f"Random Forest (AUC = {auc:.4f})", color="#2563eb", linewidth=2)
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aleatório")
    ax_roc.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax_roc.set_ylabel("Taxa de Verdadeiros Positivos (Recall)")
    ax_roc.set_title("Curva ROC", fontsize=12, fontweight="bold")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle=":", alpha=0.5)
    roc_path = os.path.join(REPORTS_DIR, "curva_roc.png")
    save_figure_to_path(fig_roc, roc_path)

    # Gráfico 2: Matrizes de Confusão comparativas
    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(10, 4.5))
    cm_default = confusion_matrix(y_test, (y_probs >= DEFAULT_THRESHOLD).astype(int))
    cm_best = confusion_matrix(y_test, (y_probs >= best_row["Threshold"]).astype(int))

    ConfusionMatrixDisplay(cm_default, display_labels=["Normal", "Falha"]).plot(
        ax=axes_cm[0], colorbar=False, cmap="Blues"
    )
    axes_cm[0].set_title(
        f"Threshold Padrão (t=0.50)\nFN={int(default_row['FN'])}, FP={int(default_row['FP'])}",
        fontsize=10, fontweight="bold",
    )

    ConfusionMatrixDisplay(cm_best, display_labels=["Normal", "Falha"]).plot(
        ax=axes_cm[1], colorbar=False, cmap="Greens"
    )
    axes_cm[1].set_title(
        f"Threshold Ótimo (t={best_row['Threshold']:.2f})\nFN={int(best_row['FN'])}, FP={int(best_row['FP'])}",
        fontsize=10, fontweight="bold",
    )
    fig_cm.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "matrizes_confusao.png")
    save_figure_to_path(fig_cm, cm_path)

    # Gráfico 3: Custo Total vs. Threshold
    fig_cost, ax_cost = plt.subplots(figsize=(8, 4.5))
    ax_cost.plot(
        results_df["Threshold"],
        results_df["Custo_Total"] / 1000,
        label="Custo Total (R$ Milhares)",
        color="#1e3a8a",
        linewidth=2.5,
    )
    ax_cost.axvline(
        DEFAULT_THRESHOLD,
        linestyle="--",
        color="#dc2626",
        linewidth=1.5,
        label=f"Padrão (0.50): R$ {default_row['Custo_Total']/1000:,.0f}k",
    )
    ax_cost.axvline(
        best_row["Threshold"],
        linestyle="--",
        color="#16a34a",
        linewidth=1.5,
        label=f"Ótimo ({best_row['Threshold']:.2f}): R$ {best_row['Custo_Total']/1000:,.0f}k",
    )
    ax_cost.scatter(
        best_row["Threshold"],
        best_row["Custo_Total"] / 1000,
        color="#16a34a",
        s=120,
        zorder=5,
        edgecolors="white",
        linewidth=2,
    )
    ax_cost.set_xlabel("Threshold de Decisão", fontsize=10)
    ax_cost.set_ylabel("Custo Operacional Total (R$ Milhares)", fontsize=10)
    ax_cost.set_title(
        "Análise Financeira: Custo Total vs. Threshold de Decisão",
        fontsize=12,
        fontweight="bold",
    )
    ax_cost.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
    ax_cost.grid(True, linestyle=":", alpha=0.6)
    cost_path = os.path.join(REPORTS_DIR, "custo_vs_threshold.png")
    save_figure_to_path(fig_cost, cost_path)

    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Arial", "B", 26)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 14, "Relatório Executivo", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Arial", "", 16)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 10, "MLOps & Manutenção Preditiva - Indústria 4.0", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_draw_color(37, 99, 235)
    pdf.set_line_width(0.8)
    pdf.line(60, pdf.get_y(), pdf.w - 60, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 7, "Engenharia de Dados e MLOps", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Dados Sintéticos: {N_SAMPLES:,} amostras (99,5% Normal / 0,5% Falha)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Gerado em: {datetime.datetime.now().strftime('%d/%m/%Y às %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Destaques", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(51, 65, 85)
    highlights = [
        f"AUC-ROC do Modelo: {auc:.4f}",
        f"Threshold Ótimo: {best_row['Threshold']:.2f} (vs. padrão 0.50)",
        f"Economia Gerada: R$ {economia_abs:,.2f} ({economia_pct:.1f}%)",
        f"Recall no Threshold Ótimo: {best_row['Recall']:.4f}",
    ]
    for h in highlights:
        pdf.cell(0, 6, f"•  {h}", align="C", new_x="LMARGIN", new_y="NEXT")

    # --- SEÇÃO 1: ARQUITETURA DA SOLUÇÃO ---
    pdf.add_page()
    pdf.chapter_title("1. Arquitetura da Solução")

    pdf.section_title("1.1 Visão Geral do Pipeline")
    pdf.body_text(
        "O pipeline de Machine Learning foi projetado para detectar falhas raras "
        "em equipamentos industriais a partir de dados de sensores. Os dados são "
        "gerados sinteticamente com forte desbalanceamento: 99,5% representam operação "
        "normal e apenas 0,5% representam falhas reais."
    )
    pdf.body_text(
        "O fluxo segue as etapas: (1) Geração de dados sintéticos → (2) Divisão "
        "treino/teste estratificada → (3) Normalização com StandardScaler → "
        "(4) Treinamento do Random Forest → (5) Predição de probabilidades → "
        "(6) Avaliação de métricas → (7) Otimização financeira do threshold."
    )

    pdf.section_title("1.2 Prevenção de Data Leakage")
    pdf.add_highlight_box(
        "GARANTIA CONTRA DATA LEAKAGE: A divisão entre treino (80%) e teste (20%) "
        "foi realizada ANTES de qualquer etapa de pré-processamento. O StandardScaler "
        "foi ajustado (fit) exclusivamente no conjunto de treino, e os dados de teste "
        "foram apenas transformados (transform) com os parâmetros aprendidos no treino. "
        "Isso garante que nenhuma informação estatística do conjunto de teste vazou "
        "para o processo de treinamento, mantendo a avaliação do modelo justa e confiável."
    )
    pdf.body_text(
        "Código-chave da prevenção:\n"
        "  scaler.fit_transform(X_train)   # Aprende média/desvio SÓ do treino\n"
        "  scaler.transform(X_test)        # Aplica os MESMOS parâmetros no teste"
    )

    pdf.section_title("1.3 Modelo Utilizado")
    pdf.body_text(
        "Foi utilizado um Random Forest Classifier com 200 estimadores e "
        "class_weight='balanced'. O parâmetro class_weight='balanced' ajusta "
        "automaticamente os pesos das classes de forma inversamente proporcional "
        "à sua frequência, compensando o forte desbalanceamento entre operação "
        "normal (99,5%) e falha (0,5%)."
    )

    # Seção 2: Quadro Comparativo de Métricas
    pdf.add_page()
    pdf.chapter_title("2. Quadro Comparativo de Métricas")

    pdf.body_text(
        "A tabela abaixo compara todas as métricas de classificação entre o threshold "
        "padrão (0.50) e o threshold ótimo encontrado pela análise financeira. Em cenários "
        "com forte desbalanceamento de classes, a Acurácia pode ser enganosa: um modelo que "
        "classifica tudo como 'Normal' atingiria ~99,5% de acurácia, porém falharia em "
        "detectar qualquer defeito real."
    )

    # Tabela de métricas
    metrics_opt = evaluate_metrics(y_test, y_probs, best_row["Threshold"])

    col_widths = [55, 45, 45, 45]
    total_w = sum(col_widths)
    x_start = (pdf.w - total_w) / 2
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(30, 58, 138)
    pdf.set_text_color(255, 255, 255)
    pdf.set_x(x_start)
    headers = ["Métrica", "Threshold 0.50", f"Threshold {best_row['Threshold']:.2f}", "Variação"]
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, align="C", fill=True)
    pdf.ln()

    rows = [
        ("Acurácia", metrics_default["Acuracia"], metrics_opt["Acuracia"]),
        ("Precisão", metrics_default["Precisao"], metrics_opt["Precisao"]),
        ("Recall", metrics_default["Recall"], metrics_opt["Recall"]),
        ("F1-Score", metrics_default["F1_Score"], metrics_opt["F1_Score"]),
        ("F2-Score", metrics_default["F2_Score"], metrics_opt["F2_Score"]),
        ("F0.5-Score", metrics_default["F0.5_Score"], metrics_opt["F0.5_Score"]),
        ("AUC-ROC", auc, auc),
    ]

    pdf.set_font("Arial", "", 9)
    for i, (name, val_def, val_opt) in enumerate(rows):
        fill = i % 2 == 0
        if fill:
            pdf.set_fill_color(241, 245, 249)
        pdf.set_text_color(51, 65, 85)
        pdf.set_x(x_start)
        pdf.cell(col_widths[0], 7, f"  {name}", border=1, fill=fill)
        pdf.cell(col_widths[1], 7, f"{val_def:.4f}", border=1, align="C", fill=fill)
        pdf.cell(col_widths[2], 7, f"{val_opt:.4f}", border=1, align="C", fill=fill)
        diff = val_opt - val_def
        sign = "+" if diff >= 0 else ""
        pdf.cell(col_widths[3], 7, f"{sign}{diff:.4f}", border=1, align="C", fill=fill)
        pdf.ln()

    pdf.set_fill_color(254, 242, 242)
    pdf.set_text_color(51, 65, 85)
    pdf.set_x(x_start)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(col_widths[0], 7, "  Custo Total (R$)", border=1, fill=True)
    pdf.cell(col_widths[1], 7, f"R$ {default_row['Custo_Total']:,.0f}", border=1, align="C", fill=True)
    pdf.cell(col_widths[2], 7, f"R$ {best_row['Custo_Total']:,.0f}", border=1, align="C", fill=True)
    pdf.set_text_color(22, 163, 74)
    pdf.cell(col_widths[3], 7, f"-R$ {economia_abs:,.0f}", border=1, align="C", fill=True)
    pdf.ln(6)
    pdf.set_text_color(51, 65, 85)
    pdf.section_title("2.1 Matriz de Confusão Detalhada")
    pdf.set_font("Arial", "", 9)
    cm_col_widths = [40, 30, 30, 10, 30, 30]
    cm_total_w = sum(cm_col_widths)
    cm_x_start = (pdf.w - cm_total_w) / 2
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(30, 58, 138)
    pdf.set_text_color(255, 255, 255)
    pdf.set_x(cm_x_start)
    pdf.cell(cm_col_widths[0], 7, "", border=1, fill=True)
    pdf.cell(cm_col_widths[1], 7, "TN", border=1, align="C", fill=True)
    pdf.cell(cm_col_widths[2], 7, "FP", border=1, align="C", fill=True)
    pdf.cell(cm_col_widths[3], 7, "", border=0, fill=False)
    pdf.cell(cm_col_widths[4], 7, "FN", border=1, align="C", fill=True)
    pdf.cell(cm_col_widths[5], 7, "TP", border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(51, 65, 85)
    for label, row_data in [("Threshold 0.50", metrics_default), (f"Threshold {best_row['Threshold']:.2f}", metrics_opt)]:
        pdf.set_x(cm_x_start)
        pdf.set_font("Arial", "B", 9)
        pdf.cell(cm_col_widths[0], 7, f"  {label}", border=1)
        pdf.set_font("Arial", "", 9)
        pdf.cell(cm_col_widths[1], 7, str(row_data["TN"]), border=1, align="C")
        pdf.cell(cm_col_widths[2], 7, str(row_data["FP"]), border=1, align="C")
        pdf.cell(cm_col_widths[3], 7, "", border=0)
        pdf.cell(cm_col_widths[4], 7, str(row_data["FN"]), border=1, align="C")
        pdf.cell(cm_col_widths[5], 7, str(row_data["TP"]), border=1, align="C")
        pdf.ln()

    pdf.ln(4)
    pdf.section_title("2.2 Visualização das Matrizes de Confusão")
    pdf.image(cm_path, x=15, w=pdf.w - 30)
    pdf.ln(4)

    # Seção 3: Curva AUC-ROC
    pdf.add_page()
    pdf.chapter_title("3. Curva AUC-ROC")

    pdf.body_text(
        f"A Curva ROC (Receiver Operating Characteristic) avalia a capacidade do modelo "
        f"de discriminar entre falhas e operações normais em diferentes thresholds. A área "
        f"sob a curva (AUC) obtida foi de {auc:.4f}, indicando excelente capacidade "
        f"discriminativa. Um modelo aleatório teria AUC = 0.50."
    )
    pdf.image(roc_path, x=30, w=pdf.w - 60)

    # Seção 4: Análise Financeira do Threshold
    pdf.add_page()
    pdf.chapter_title("4. Análise Financeira do Threshold")

    pdf.body_text(
        "Em manutenção preditiva industrial, os custos de diferentes tipos de erro "
        "são assimétricos. Um Falso Negativo (falha não detectada) resulta em quebra "
        f"de equipamento com custo estimado de R$ {COST_FN:,.0f}. Um Falso Positivo "
        f"(alarme falso) gera uma inspeção desnecessária com custo de R$ {COST_FP:,.0f}."
    )

    pdf.section_title("4.1 Custo Total vs. Threshold de Decisão")
    pdf.image(cost_path, x=15, w=pdf.w - 30)
    pdf.ln(4)

    pdf.section_title("4.2 Resultados da Otimização")
    pdf.add_highlight_box(
        f"PONTO DE MENOR CUSTO\n"
        f"Threshold Ótimo: {best_row['Threshold']:.2f}\n"
        f"Custo Total (Ótimo): R$ {best_row['Custo_Total']:,.2f}\n"
        f"Custo Total (Padrão 0.50): R$ {default_row['Custo_Total']:,.2f}\n"
        f"Economia Absoluta: R$ {economia_abs:,.2f}\n"
        f"Economia Percentual: {economia_pct:.1f}%",
        r=240, g=253, b=244, border_r=22, border_g=163, border_b=74,
    )

    pdf.body_text(
        f"Ao reduzir o threshold de decisão de 0.50 para {best_row['Threshold']:.2f}, "
        f"o modelo torna-se mais sensível a falhas, capturando mais defeitos antes "
        f"que ocorram. Embora isso aumente levemente os falsos positivos (alarmes "
        f"falsos), a economia total gerada pela redução drástica de falsos negativos "
        f"é de R$ {economia_abs:,.2f}, representando uma redução de {economia_pct:.1f}% "
        f"no custo operacional total."
    )

    pdf.add_page()
    pdf.chapter_title("5. Recomendações MLOps")

    pdf.body_text(
        "Para que o modelo de manutenção preditiva opere de forma confiável em "
        "ambiente de produção industrial, é fundamental implementar um plano "
        "estruturado de monitoramento e governança."
    )

    pdf.section_title("5.1 Monitoramento de Data Drift")
    pdf.body_text(
        "Implementar testes estatísticos de Kolmogorov-Smirnov (KS) em tempo real "
        "nos sensores de vibração, temperatura e pressão para identificar mudanças "
        "na distribuição dos dados que podem indicar desgaste natural do maquinário "
        "ou alterações no processo produtivo. Quando detectado drift significativo, "
        "disparar alerta para reavaliação do modelo."
    )

    pdf.section_title("5.2 Estratégia de Retreinamento")
    pdf.body_text(
        "Estabelecer retreinamento automático em duas condições:\n"
        "  • Periódico: mensal, com dados acumulados dos últimos 90 dias.\n"
        "  • Por gatilho: quando a métrica F2-Score cair abaixo de 0.80, "
        "    indicando degradação na capacidade de detecção de falhas.\n\n"
        "O retreinamento deve incluir validação temporal (time-series split) para "
        "garantir que o modelo generaliza para dados futuros."
    )

    pdf.section_title("5.3 Feature Store e Governança")
    pdf.body_text(
        "Centralizar o pipeline de transformação (Scalers, Encoders) em uma Feature Store "
        "para garantir simetria entre o ambiente de treinamento e a inferência na borda "
        "(Edge Computing). Todas as transformações devem ser versionadas e auditáveis."
    )

    pdf.section_title("5.4 Protocolo de Fallback de Segurança")
    pdf.body_text(
        "Em caso de indisponibilidade do modelo (timeout, erro de infraestrutura) "
        "ou dados corrompidos nos sensores, a linha de montagem deve assumir um "
        "estado preventivo reduzido com alerta automático à equipe de manutenção. "
        "Isso garante que falhas de ML não comprometam a segurança operacional."
    )

    pdf.section_title("5.5 Registro e Versionamento de Modelos")
    pdf.body_text(
        "Utilizar ferramentas como MLflow ou similares para registrar:\n"
        "  • Versão do modelo e hiperparâmetros utilizados.\n"
        "  • Métricas de avaliação em cada retreinamento.\n"
        "  • Threshold de decisão ativo em produção.\n"
        "  • Dataset e features utilizados para treinamento.\n\n"
        "Isso permite rastreabilidade completa e rollback rápido caso um novo "
        "modelo apresente performance inferior."
    )

    pdf_path = os.path.join(REPORTS_DIR, PDF_FILENAME)
    pdf.output(pdf_path)

    print(f"\n{'=' * 60}")
    print(f"  PDF gerado com sucesso!")
    print(f"  Arquivo: {pdf_path}")
    print(f"{'=' * 60}")
    return pdf_path


if __name__ == "__main__":
    generate_report()
