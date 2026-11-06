"""DengueAI Pediátrico — plataforma académica de predicción de complicaciones del dengue."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="DengueAI Pediátrico", page_icon="🧬", layout="wide",
                   initial_sidebar_state="expanded")

NAVY, BLUE, SKY, TEAL, CORAL, GREY = "#0B1F3A", "#1D4ED8", "#7DD3FC", "#14B8A6", "#F97316", "#64748B"
DATA_PATH = Path(__file__).parent / "data" / "MASTER_CHART_F1000Research.xlsx"
DISCLAIMER = ("⚠️ **Herramienta académica.** Las estimaciones provienen de modelos entrenados con 303 pacientes y "
              "**no constituyen un diagnóstico médico** ni sustituyen la valoración de un profesional de salud.")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3 {{ font-family: 'Sora', sans-serif; color: {NAVY}; letter-spacing: -0.02em; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {NAVY} 0%, #12305a 100%); }}
[data-testid="stSidebar"] * {{ color: #E6F1FF !important; }}
[data-testid="stSidebar"] .stRadio label {{ padding: 6px 10px; border-radius: 10px; transition: background .2s; }}
[data-testid="stSidebar"] .stRadio label:hover {{ background: rgba(125,211,252,.15); }}
.hero {{ background: radial-gradient(1200px 400px at 85% -20%, rgba(20,184,166,.45), transparent),
         linear-gradient(120deg, {NAVY}, #153a73 60%, #0e7490); color: white; padding: 42px 38px;
         border-radius: 22px; animation: fadeUp .7s ease both; }}
.hero h1 {{ color: white; font-size: 2.3rem; margin: 6px 0 10px; }}
.hero p {{ color: #D6E8FF; max-width: 820px; font-size: 1.04rem; line-height: 1.6; }}
.pill {{ display:inline-block; background: rgba(20,184,166,.22); color:#7CF3E4; border:1px solid rgba(20,184,166,.5);
        padding: 3px 12px; border-radius: 999px; font-size: .78rem; letter-spacing:.08em; text-transform:uppercase; }}
.kpi {{ background: white; border: 1px solid #E2EAF4; border-radius: 16px; padding: 18px 20px;
        box-shadow: 0 4px 18px rgba(11,31,58,.06); animation: fadeUp .6s ease both; transition: transform .2s; }}
.kpi:hover {{ transform: translateY(-3px); }}
.kpi .v {{ font-family:'Sora'; font-size: 1.9rem; font-weight: 700; color: {BLUE}; }}
.kpi .l {{ color: {GREY}; font-size: .85rem; }}
.warn {{ background:#FFF7ED; border-left: 5px solid {CORAL}; padding: 12px 16px; border-radius: 10px; color:#7c2d12; }}
.note {{ background:#F0F9FF; border-left: 5px solid {BLUE}; padding: 12px 16px; border-radius: 10px; color:{NAVY}; }}
@keyframes fadeUp {{ from {{opacity:0; transform: translateY(10px);}} to {{opacity:1; transform:none;}} }}
</style>
""", unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="Inter"), title_font=dict(family="Sora", size=16, color=NAVY),
                      colorway=[BLUE, TEAL, CORAL, SKY, NAVY, GREY])
    return fig


def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


def pct(x):
    return f"{100 * x:.1f} %"


# ------------------------------------------------------------------ datos y modelos (con caché)
@st.cache_data(show_spinner=False)
def _load_default():
    return core.load_raw(DATA_PATH)


@st.cache_data(show_spinner=False)
def _prepare(raw):
    return core.prepare(raw)


@st.cache_resource(show_spinner="Entrenando regresión logística y red neuronal (búsqueda de hiperparámetros con validación cruzada)…")
def _train(df):
    return core.train_all(df)


raw = st.session_state.get("raw")
source = st.session_state.get("source", "MASTER_CHART_F1000Research.xlsx (dataset integrado)")
if raw is None:
    raw = _load_default()
try:
    df, report = _prepare(raw)
    data_error = None
except Exception as exc:  # dataset subido con formato incorrecto
    data_error = str(exc)
    raw = _load_default()
    df, report = _prepare(raw)
    st.session_state.pop("raw", None)

# ------------------------------------------------------------------ navegación
PAGES = ["🏠 Inicio", "📂 Datos", "📈 Regresión logística", "🧠 Red neuronal", "⚖️ Comparación de modelos",
         "🩺 Calculadora clínica (0–7)", "📊 Dashboard"]
with st.sidebar:
    st.markdown("## 🧬 DengueAI\n**Pediátrico** · riesgo de complicaciones")
    page = st.radio("Navegación", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"Dataset activo: {source}")
    st.caption("Herramienta académica · no es diagnóstico médico")

if data_error:
    st.error(f"El archivo subido no es válido: {data_error}. Se mantiene el dataset integrado.")

res = _train(df)
LR, NN = res["models"]["Regresión logística"], res["models"]["Red neuronal (MLP)"]


# ------------------------------------------------------------------ componentes reutilizables
def patient_form(key: str) -> dict:
    med = df[core.NUMERIC_FEATURES].median()
    c1, c2, c3 = st.columns(3)
    age = c1.radio("Edad", ["< 6 años", "> 6 años"], index=1, horizontal=True, key=f"{key}_age")
    sex = c2.radio("Sexo", ["Masculino", "Femenino"], horizontal=True, key=f"{key}_sex")
    fever = c3.slider("Días de fiebre", 1, 10, int(med["dias_fiebre"]), key=f"{key}_fever")
    st.markdown("**Síntomas y signos al ingreso**")
    cols = st.columns(4)
    flags = {}
    for i, f in enumerate(["cefalea", "mialgia", "dolor_abdominal", "exantema", "vomito", "disnea",
                           "sangrado", "organomegalia", "hipotension"]):
        flags[f] = cols[i % 4].checkbox(core.LABELS[f], key=f"{key}_{f}")
    st.markdown("**Laboratorio**")
    l = st.columns(4)
    hb = l[0].number_input(core.LABELS["hb"], 5.0, 25.0, float(med["hb"]), 0.1, key=f"{key}_hb")
    pcv = l[1].number_input(core.LABELS["pcv"], 15.0, 70.0, float(med["pcv"]), 0.1, key=f"{key}_pcv")
    wmin = l[2].number_input(core.LABELS["wbc_min"], 500, 40000, int(med["wbc_min"]), 100, key=f"{key}_wmin")
    wmax = l[3].number_input(core.LABELS["wbc_max"], 500, 50000, int(med["wbc_max"]), 100, key=f"{key}_wmax")
    l2 = st.columns(3)
    plt_ = l2[0].number_input(core.LABELS["plaquetas"], 1000, 800000, int(med["plaquetas"]), 1000, key=f"{key}_plt")
    alt = l2[1].number_input(core.LABELS["alt"], 1.0, 3000.0, float(med["alt"]), 1.0, key=f"{key}_alt")
    cr = l2[2].number_input(core.LABELS["creatinina"], 0.1, 10.0, float(med["creatinina"]), 0.1, key=f"{key}_cr")
    p = {"edad_mayor6": int(age.startswith(">")), "sexo_femenino": int(sex == "Femenino"), "dias_fiebre": fever,
         "hb": hb, "pcv": pcv, "wbc_min": wmin, "wbc_max": wmax, "plaquetas": plt_, "alt": alt, "creatinina": cr}
    p.update({k: int(v) for k, v in flags.items()})
    return p


BANDS = [(0.0, 0.33, "Bajo", TEAL), (0.33, 0.66, "Intermedio", "#F59E0B"), (0.66, 1.01, "Alto", "#DC2626")]


def band(prob):
    for lo, hi, name, color in BANDS:
        if lo <= prob < hi:
            return name, color
    return BANDS[-1][2], BANDS[-1][3]


def gauge(prob, title):
    name, color = band(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number=dict(suffix=" %", font=dict(family="Sora")),
        title=dict(text=title), gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=color),
                                          steps=[dict(range=[0, 33], color="#E6FAF7"), dict(range=[33, 66], color="#FEF3C7"),
                                                 dict(range=[66, 100], color="#FEE2E2")])))
    return style(fig, 300)


def metrics_cards(m):
    c = st.columns(6)
    for col, (k, lab) in zip(c, [("accuracy", "Exactitud"), ("precision", "Precisión"), ("recall", "Sensibilidad"),
                                 ("specificity", "Especificidad"), ("f1", "F1-score"), ("auc", "AUC")]):
        kpi(col, f"{m[k]:.3f}", lab)


def cm_fig(cm, title):
    fig = go.Figure(go.Heatmap(z=cm, x=["Pred. sin complicación", "Pred. complicado"], y=["Real sin complicación", "Real complicado"],
                               text=cm, texttemplate="%{text}", colorscale=[[0, "#EAF4FF"], [1, BLUE]], showscale=False,
                               textfont=dict(size=22)))
    fig.update_yaxes(autorange="reversed")
    return style(fig.update_layout(title=title), 330)


def roc_fig(curves, title="Curva ROC"):
    fig = go.Figure()
    for name, (fpr, tpr, auc, color) in curves.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})", line=dict(width=3, color=color)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(dash="dash", color=GREY)))
    fig.update_layout(title=title, xaxis_title="1 − Especificidad", yaxis_title="Sensibilidad")
    return style(fig, 420)


def model_page(name, m, icon_txt):
    st.header(icon_txt)
    s = res["split"]
    st.markdown(f'<div class="note">Entrenamiento: <b>{s["n_train"]}</b> pacientes ({s["pos_train"]} complicados) · Prueba (hold-out '
                f'estratificado, nunca vista durante el ajuste): <b>{s["n_test"]}</b> pacientes ({s["pos_test"]} complicados). '
                f'Desbalance de clases controlado: {"class_weight=balanced" if name.startswith("Reg") else "sobremuestreo aleatorio solo del conjunto de entrenamiento"}. '
                f'Hiperparámetros elegidos por validación cruzada 5-fold <b>solo dentro del entrenamiento</b>: {m["best_params"]}.</div>',
                unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🧪 Predecir paciente", "📏 Métricas", "🔍 Interpretación"])
    with tab1:
        p = patient_form("lr" if name.startswith("Reg") else "nn")
        if st.button("Calcular riesgo", type="primary", key=f"btn_{name}"):
            prob = core.predict_patient(m["estimator"], p)
            lab, _ = band(prob)
            c1, c2 = st.columns([1, 1])
            c1.plotly_chart(gauge(prob, "Probabilidad estimada de complicación"), use_container_width=True)
            c2.markdown(f"### Banda de probabilidad: **{lab}**")
            c2.markdown(f"Clasificación del modelo (umbral 0,5): **{'Complicado' if prob >= 0.5 else 'No complicado'}**")
            c2.caption("Las bandas (<33 %, 33–66 %, ≥66 %) son una convención visual de esta aplicación, no criterios clínicos validados.")
            c2.markdown(DISCLAIMER)
    with tab2:
        st.subheader("Conjunto de prueba (hold-out)")
        metrics_cards(m)
        c1, c2 = st.columns(2)
        c1.plotly_chart(cm_fig(m["cm"], "Matriz de confusión · hold-out"), use_container_width=True)
        c2.plotly_chart(roc_fig({name: (m["fpr"], m["tpr"], m["auc"], BLUE)}), use_container_width=True)
        st.markdown('<div class="warn">El hold-out contiene pocos casos complicados; sus métricas varían mucho según el corte. '
                    'Por eso se reporta también la validación cruzada 5-fold sobre los 303 pacientes (predicciones out-of-fold):</div>',
                    unsafe_allow_html=True)
        st.write("")
        metrics_cards(m["oof"])
        c1, c2 = st.columns(2)
        c1.plotly_chart(cm_fig(m["oof"]["cm"], "Matriz de confusión · validación cruzada (n = 303)"), use_container_width=True)
        cv = pd.DataFrame({k: f"{v[0]:.3f} ± {v[1]:.3f}" for k, v in m["cv_full"].items()}, index=["media ± DE (5 folds)"]).T
        c2.markdown("**Variabilidad entre folds**")
        c2.dataframe(cv, use_container_width=True)
    with tab3:
        if "coef" in m:
            c = m["coef"].copy()
            c["variable"] = c["variable"].map(core.LABELS)
            fig = px.bar(c, x="coef_std", y="variable", orientation="h", title="Coeficientes estandarizados (log-odds por 1 DE)",
                         color="coef_std", color_continuous_scale=[[0, TEAL], [0.5, "#E2EAF4"], [1, CORAL]])
            st.plotly_chart(style(fig, 520).update_coloraxes(showscale=False), use_container_width=True)
            st.caption("Positivo = mayor riesgo de complicación. Con regularización L2 fuerte (C pequeño) los coeficientes se contraen: "
                       "interprételos como ranking relativo, no como odds ratios exactos.")
            st.dataframe(c.rename(columns={"variable": "Variable", "coef_std": "Coeficiente", "OR_por_DE": "OR por 1 DE"})
                         .sort_values("Coeficiente", ascending=False).round(3), use_container_width=True, hide_index=True)
        else:
            c = m["importance"].copy()
            c["variable"] = c["variable"].map(core.LABELS)
            fig = px.bar(c, x="importancia", y="variable", orientation="h", error_x="de",
                         title="Importancia por permutación (caída del AUC en hold-out)")
            st.plotly_chart(style(fig, 520), use_container_width=True)
            lc = go.Figure(go.Scatter(y=m["loss_curve"], mode="lines", line=dict(color=TEAL, width=3)))
            lc.update_layout(title="Curva de pérdida del entrenamiento", xaxis_title="Época", yaxis_title="Pérdida")
            st.plotly_chart(style(lc, 300), use_container_width=True)
            st.caption("Arquitectura: perceptrón multicapa (ReLU, optimizador Adam, regularización L2) entrenado con retropropagación "
                       "mediante scikit-learn `MLPClassifier`.")


# ------------------------------------------------------------------ páginas
if page == PAGES[0]:
    st.markdown(f"""<div class="hero"><span class="pill">IA aplicada a la salud · proyecto académico</span>
    <h1>Sistema inteligente de predicción del riesgo de complicaciones del dengue en pacientes pediátricos</h1>
    <p>Plataforma que entrena y compara una <b>regresión logística</b> y una <b>red neuronal artificial</b> con datos reales de
    303 niños con dengue confirmado (dos hospitales terciarios, 2016–2020) y los contrasta con el sistema de puntuación clínica
    de 0 a 7 publicado en F1000Research (2022).</p></div>""", unsafe_allow_html=True)
    st.write("")
    c = st.columns(5)
    kpi(c[0], len(df), "Pacientes en el dataset")
    kpi(c[1], int(df.grupo.sum()), f"Dengue complicado ({pct(df.grupo.mean())})")
    kpi(c[2], len(core.FEATURES), "Variables predictoras")
    kpi(c[3], f"{LR['oof']['auc']:.2f} / {NN['oof']['auc']:.2f}", "AUC validación cruzada (LR / red neuronal)")
    kpi(c[4], f"{res['score']['all']['auc']:.2f}", "AUC del score clínico 0–7 (dataset completo)")
    st.write("")
    a, b = st.columns(2)
    with a:
        st.subheader("Objetivo general")
        st.write("Construir y evaluar modelos de analítica de datos que estimen la probabilidad de complicación del dengue pediátrico "
                 "a partir de variables clínicas y de laboratorio al ingreso, y compararlos con el score clínico de referencia.")
        st.subheader("Metodología")
        st.markdown("- Preprocesamiento sin fuga de información: imputación por mediana y estandarización ajustadas solo con el entrenamiento.\n"
                    "- Variables de tratamiento (fluidos, antibióticos, antipiréticos, esteroides) **excluidas**: se indican después de valorar al paciente.\n"
                    "- Desbalance (≈21 % complicados) controlado con pesos de clase / sobremuestreo solo en entrenamiento.\n"
                    "- Evaluación en hold-out estratificado y en validación cruzada 5-fold.")
    with b:
        st.subheader("Limitaciones")
        st.markdown("- Muestra pequeña (303 pacientes, 63 complicados) → métricas con alta variabilidad.\n"
                    "- Sin validación externa; los autores del score también la señalan como necesaria.\n"
                    "- Las predicciones no son diagnósticos ni guía de tratamiento.")
        st.markdown(f'<div class="warn">{DISCLAIMER}</div>', unsafe_allow_html=True)

elif page == PAGES[1]:
    st.header("📂 Carga y preparación de datos")
    up = st.file_uploader("Sube un Excel o CSV con el mismo formato del dataset (códigos del README)", type=["xlsx", "xls", "csv"])
    c1, c2 = st.columns(2)
    if up is not None and c1.button("Usar este archivo", type="primary"):
        try:
            st.session_state["raw"] = core.load_raw(up)
            st.session_state["source"] = up.name
            st.rerun()
        except Exception as exc:
            st.error(f"No se pudo leer el archivo: {exc}")
    if "raw" in st.session_state and c2.button("Volver al dataset integrado"):
        st.session_state.pop("raw")
        st.session_state.pop("source", None)
        st.rerun()
    k = st.columns(4)
    kpi(k[0], report["filas_finales"], "Registros válidos")
    kpi(k[1], len(raw.columns), "Columnas originales")
    kpi(k[2], report["duplicados"], "Filas duplicadas")
    kpi(k[3], pct(report["prevalencia"]), "Prevalencia de complicación")
    t1, t2, t3, t4 = st.tabs(["Registros", "Valores faltantes", "Limpieza aplicada", "Variables"])
    with t1:
        st.dataframe(raw, use_container_width=True, height=420)
    with t2:
        miss = raw.isna().sum().rename("Faltantes").to_frame()
        miss["%"] = (100 * miss["Faltantes"] / len(raw)).round(1)
        miss = miss[miss["Faltantes"] > 0]
        if miss.empty:
            st.success("No hay valores faltantes.")
        else:
            st.dataframe(miss, use_container_width=True)
            st.caption("Los faltantes de las variables predictoras se imputan con la mediana ajustada solo en el entrenamiento.")
    with t3:
        st.markdown("- Nombres de columna normalizados y renombrados.\n- Códigos del README recodificados a 0/1 (1 = Sí / 2 = No → 1/0; "
                    "Edad 2 = > 6 años → 1; Sexo 2 = femenino → 1; Grupo 2 = complicado → 1).\n"
                    "- `BP AT ADMISSION`: el código 2 se interpreta como hipotensión. Esta lectura se verificó empíricamente: con ella el score del artículo "
                    "reproduce exactamente su sensibilidad (84,13 %) y especificidad (72,50 %) en este dataset.\n"
                    "- Variables numéricas asimétricas (leucocitos, plaquetas, ALT, creatinina) transformadas con log(1+x).\n"
                    "- Imputación por mediana y estandarización dentro del pipeline (solo con datos de entrenamiento).")
        st.dataframe(df.head(50), use_container_width=True, height=300)
    with t4:
        rows = [{"Variable": core.LABELS[f], "Rol": "Predictora", "Nota": ""} for f in core.FEATURES]
        rows += [{"Variable": core.LABELS[f], "Rol": "Excluida", "Nota": why} for f, why in core.EXCLUDED.items()]
        rows.append({"Variable": "Grupo (dengue complicado)", "Rol": "Objetivo", "Nota": "1 = complicado"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

elif page == PAGES[2]:
    model_page("Regresión logística", LR, "📈 Modelo de regresión logística")

elif page == PAGES[3]:
    model_page("Red neuronal (MLP)", NN, "🧠 Modelo de red neuronal artificial")

elif page == PAGES[4]:
    st.header("⚖️ Comparación de modelos")
    basis = st.radio("Base de evaluación", ["Validación cruzada 5-fold (out-of-fold, n = 303)", "Hold-out (20 %, n = 61)"], horizontal=True)
    oof = basis.startswith("Validación")
    get = lambda m: m["oof"] if oof else m
    sc = res["score"]["all"] if oof else res["score"]["test"]
    rows = []
    for name, m in [("Regresión logística", LR), ("Red neuronal (MLP)", NN)]:
        g = get(m)
        rows.append({"Modelo": name, "Exactitud": g["accuracy"], "Precisión": g["precision"], "Sensibilidad": g["recall"],
                     "Especificidad": g["specificity"], "F1-score": g["f1"], "AUC": g["auc"]})
    rows.append({"Modelo": "Score clínico 0–7 (≥ 2)", "Exactitud": sc["accuracy"], "Precisión": sc["precision"], "Sensibilidad": sc["recall"],
                 "Especificidad": sc["specificity"], "F1-score": sc["f1"], "AUC": sc["auc"]})
    tab = pd.DataFrame(rows)
    st.dataframe(tab.style.format({c: "{:.3f}" for c in tab.columns[1:]}), use_container_width=True, hide_index=True)
    long = tab.melt("Modelo", var_name="Métrica", value_name="Valor")
    fig = px.bar(long, x="Métrica", y="Valor", color="Modelo", barmode="group", title="Métricas por modelo",
                 color_discrete_sequence=[BLUE, TEAL, CORAL])
    st.plotly_chart(style(fig.update_yaxes(range=[0, 1])), use_container_width=True)
    curves = {"Regresión logística": (get(LR)["fpr"], get(LR)["tpr"], get(LR)["auc"], BLUE),
              "Red neuronal": (get(NN)["fpr"], get(NN)["tpr"], get(NN)["auc"], TEAL),
              "Score clínico 0–7": (sc["fpr"], sc["tpr"], sc["auc"], CORAL)}
    st.plotly_chart(roc_fig(curves, "Curvas ROC"), use_container_width=True)
    c = st.columns(3)
    c[0].plotly_chart(cm_fig(get(LR)["cm"], "Regresión logística"), use_container_width=True)
    c[1].plotly_chart(cm_fig(get(NN)["cm"], "Red neuronal"), use_container_width=True)
    c[2].plotly_chart(cm_fig(sc["cm"], "Score clínico (≥ 2)"), use_container_width=True)
    st.markdown('<div class="note">Todas las métricas se calculan en vivo sobre predicciones reales no vistas durante el ajuste. '
                'El score clínico no se entrena aquí: es el del artículo, aplicado tal cual a los mismos pacientes. '
                'Con 63 casos complicados las diferencias entre modelos pueden no ser estadísticamente significativas; '
                'revise la desviación estándar entre folds en cada página de modelo.</div>', unsafe_allow_html=True)

elif page == PAGES[5]:
    st.header("🩺 Calculadora de riesgo clínico (score 0–7)")
    st.markdown('<div class="note">Implementación del sistema de puntuación de <i>Predictive scoring system for risk of complications in pediatric '
                'dengue infection</i> (F1000Research 2022, 11:446). Variables, puntos y criterios tal como los publican los autores: '
                'PCV ≥ 42 % (1) · plaquetas ≤ 75 000/mm³ (1) · leucocitos ≥ 7 000/mm³ (1) · ALT ≥ 70 U/L (1) · hipotensión (3). '
                'Punto de corte de los autores: ≥ 2.</div>', unsafe_allow_html=True)
    st.write("")
    med = df[core.NUMERIC_FEATURES].median()
    c = st.columns(5)
    pcv = c[0].number_input("PCV (%)", 15.0, 70.0, float(med["pcv"]), 0.1)
    plt_ = c[1].number_input("Plaquetas (/mm³)", 1000, 800000, int(med["plaquetas"]), 1000)
    wbc = c[2].number_input("Leucocitos (/mm³)", 500, 50000, int(med["wbc_max"]), 100)
    alt = c[3].number_input("ALT (U/L)", 1.0, 3000.0, float(med["alt"]), 1.0)
    hypo = c[4].checkbox("Hipotensión (según tablas por edad)")
    out = core.clinical_score(pcv, plt_, wbc, alt, hypo)
    tot = out["total"]
    a, b = st.columns([1, 1])
    fig = go.Figure(go.Indicator(mode="gauge+number", value=tot, title=dict(text="Puntaje total"),
                                 number=dict(suffix=" / 7", font=dict(family="Sora")),
                                 gauge=dict(axis=dict(range=[0, 7], dtick=1), bar=dict(color=NAVY),
                                            steps=[dict(range=[0, 2], color="#E6FAF7"), dict(range=[2, 3], color="#FEF3C7"),
                                                   dict(range=[3, 7], color="#FEE2E2")])))
    a.plotly_chart(style(fig, 320), use_container_width=True)
    with b:
        st.markdown("### Resultado")
        if tot >= 2:
            st.error(f"Puntaje {tot}: **riesgo aumentado de complicación** (punto de corte ≥ 2 del artículo).")
        else:
            st.success(f"Puntaje {tot}: **por debajo del punto de corte** (< 2) del artículo.")
        if tot <= 1:
            st.write("En la discusión del artículo, los autores sugieren que 0–1 punto podría manejarse de forma ambulatoria.")
        else:
            st.write("En la discusión, los autores comentan que 2 puntos puede requerir hospitalización para observación "
                     "y ≥ 3 puntos atención más intensiva.")
        st.caption("Esas sugerencias de manejo provienen de la discusión del artículo; no son categorías validadas formalmente.")
        det = pd.DataFrame({"Criterio": list(out["puntos"]), "Puntos": list(out["puntos"].values())})
        st.dataframe(det, use_container_width=True, hide_index=True)
    d = res["score"]["dist"].rename(columns={0: "Sin complicación", 1: "Complicado"}).reset_index(names="Puntaje")
    d["% complicado"] = (100 * d["Complicado"] / (d["Complicado"] + d["Sin complicación"])).round(1)
    st.subheader("Cómo se distribuyen los puntajes en este dataset")
    st.dataframe(d, use_container_width=True, hide_index=True)
    st.caption("Con el dataset completo y corte ≥ 2 se obtienen sensibilidad 84,13 % y especificidad 72,50 %, idénticas a las publicadas.")
    st.markdown(DISCLAIMER)

else:
    st.header("📊 Dashboard")
    k = st.columns(5)
    kpi(k[0], len(df), "Pacientes")
    kpi(k[1], pct(df.grupo.mean()), "Complicados")
    kpi(k[2], pct(df.edad_mayor6.mean()), "Mayores de 6 años")
    kpi(k[3], pct((1 - df.sexo_femenino).mean()), "Masculinos")
    kpi(k[4], f"{df.dias_fiebre.mean():.1f}", "Días de fiebre (media)")
    st.write("")
    lab = df.assign(Grupo=df.grupo.map({0: "Sin complicación", 1: "Complicado"}))
    r1 = st.columns(3)
    r1[0].plotly_chart(style(px.pie(lab, names="Grupo", hole=.55, title="Distribución de pacientes",
                                    color_discrete_sequence=[BLUE, CORAL]), 340), use_container_width=True)
    g = lab.assign(Edad=lab.edad_mayor6.map({0: "< 6 años", 1: "> 6 años"})).groupby(["Edad", "Grupo"]).size().reset_index(name="n")
    r1[1].plotly_chart(style(px.bar(g, x="Edad", y="n", color="Grupo", barmode="group", title="Edad × grupo",
                                    color_discrete_sequence=[BLUE, CORAL]), 340), use_container_width=True)
    s_all = core.clinical_score_df(df)
    ct = pd.crosstab(s_all, lab.Grupo)
    ct.index.name = "Puntaje"
    sd = ct.reset_index().melt("Puntaje", var_name="Grupo", value_name="n")
    r1[2].plotly_chart(style(px.bar(sd, x="Puntaje", y="n", color="Grupo", title="Distribución del score clínico 0–7",
                                    color_discrete_sequence=[BLUE, CORAL]), 340), use_container_width=True)
    sym = ["cefalea", "mialgia", "dolor_abdominal", "exantema", "vomito", "disnea", "sangrado", "organomegalia", "hipotension"]
    prev = df.groupby("grupo")[sym].mean().T.rename(columns={0: "Sin complicación", 1: "Complicado"}) * 100
    prev.index = [core.LABELS[i] for i in prev.index]
    r2 = st.columns(2)
    r2[0].plotly_chart(style(px.bar(prev.reset_index().melt("index"), x="value", y="index", color="variable", barmode="group",
                                    orientation="h", title="Prevalencia de síntomas y signos (%)",
                                    labels={"value": "%", "index": "", "variable": ""},
                                    color_discrete_sequence=[BLUE, CORAL]), 440), use_container_width=True)
    var = r2[1].selectbox("Variable de laboratorio", ["pcv", "plaquetas", "wbc_max", "wbc_min", "alt", "creatinina", "hb", "dias_fiebre"],
                          format_func=lambda v: core.LABELS[v])
    r2[1].plotly_chart(style(px.box(lab, x="Grupo", y=var, color="Grupo", points="outliers", title=core.LABELS[var],
                                    color_discrete_sequence=[BLUE, CORAL]), 380), use_container_width=True)
    corr = df[core.FEATURES + ["grupo"]].rename(columns=core.LABELS).corr()
    st.plotly_chart(style(px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlaciones (Pearson)"), 640),
                    use_container_width=True)
    st.subheader("Modelos y predicciones")
    probs = pd.DataFrame({"Grupo real": lab.Grupo, "Regresión logística": LR["p_oof"], "Red neuronal": NN["p_oof"]}).melt("Grupo real", var_name="Modelo",
                                                                                                                      value_name="Probabilidad")
    st.plotly_chart(style(px.box(probs, x="Modelo", y="Probabilidad", color="Grupo real", title="Probabilidades predichas (out-of-fold) según el grupo real",
                                 color_discrete_sequence=[BLUE, CORAL]), 400), use_container_width=True)
    cmp_ = pd.DataFrame({"Modelo": ["Regresión logística", "Red neuronal", "Score clínico"],
                         "AUC": [LR["oof"]["auc"], NN["oof"]["auc"], res["score"]["all"]["auc"]]})
    st.plotly_chart(style(px.bar(cmp_, x="Modelo", y="AUC", text_auto=".3f", title="AUC (validación cruzada / dataset completo para el score)",
                                 color="Modelo", color_discrete_sequence=[BLUE, TEAL, CORAL]).update_yaxes(range=[0, 1]), 340), use_container_width=True)
    pred = pd.DataFrame({"Real": res["yte"], "P(regresión logística)": LR["p_test"], "P(red neuronal)": NN["p_test"]}, index=res["Xte"].index)
    pred["Real"] = pred["Real"].map({0: "Sin complicación", 1: "Complicado"})
    st.markdown("**Predicciones sobre el conjunto de prueba (hold-out)**")
    st.dataframe(pred.round(3), use_container_width=True, height=300)
    buf = io.StringIO()
    pred.to_csv(buf)
    st.download_button("Descargar predicciones (CSV)", buf.getvalue(), "predicciones_holdout.csv", "text/csv")
    st.markdown(DISCLAIMER)
