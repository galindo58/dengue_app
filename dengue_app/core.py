"""Núcleo de datos y machine learning (sin dependencias de Streamlit).

Todo lo que muestra la aplicación se calcula aquí a partir del dataset real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

# Columna original del Excel -> nombre interno
RAW_TO_INTERNAL = {
    "GROUPS: DENGUE FEVER OR COMPLICATED DENGUE": "grupo",
    "AGE": "edad_mayor6", "GENDER": "sexo_femenino", "DURATION OF FEVER": "dias_fiebre",
    "HEADACHE": "cefalea", "MYALGIA": "mialgia", "ABDOMINAL PAIN": "dolor_abdominal",
    "RASH": "exantema", "VOMITING": "vomito", "BREATHLESSNESS": "disnea", "BLEEDING": "sangrado",
    "BP AT ADMISSION": "hipotension", "ORGANOMEGALY": "organomegalia",
    "HB": "hb", "PCV": "pcv", "LOWEST WBC": "wbc_min", "HIGHEST WBC": "wbc_max",
    "PLATELET COUNT": "plaquetas", "ALT": "alt", "CREATININE": "creatinina", "FERRITIN": "ferritina",
    "IV FLUIDS": "fluidos_iv", "ANTIBIOTICS": "antibioticos", "ANTIPYRETICS": "antipireticos",
    "STEROIDS": "esteroides",
}

LABELS = {
    "edad_mayor6": "Edad > 6 años", "sexo_femenino": "Sexo femenino", "dias_fiebre": "Duración de la fiebre (días)",
    "cefalea": "Cefalea", "mialgia": "Mialgia", "dolor_abdominal": "Dolor abdominal", "exantema": "Exantema",
    "vomito": "Vómito", "disnea": "Disnea", "sangrado": "Sangrado", "hipotension": "Hipotensión al ingreso",
    "organomegalia": "Organomegalia", "hb": "Hemoglobina (g/dL)", "pcv": "Hematocrito / PCV (%)",
    "wbc_min": "Leucocitos mínimos (/mm³)", "wbc_max": "Leucocitos máximos (/mm³)",
    "plaquetas": "Plaquetas (/mm³)", "alt": "ALT (U/L)", "creatinina": "Creatinina (mg/dL)",
    "ferritina": "Ferritina (ng/mL)", "fluidos_iv": "Fluidos i/v", "antibioticos": "Antibióticos i/v",
    "antipireticos": "Antipiréticos i/v", "esteroides": "Esteroides", "grupo": "Complicado",
}

YES_NO_1YES = ["cefalea", "mialgia", "dolor_abdominal", "exantema", "vomito", "disnea", "sangrado",
               "organomegalia", "fluidos_iv", "antibioticos", "antipireticos", "esteroides"]

# Predictores usados por los modelos: solo información disponible al ingreso.
BINARY_FEATURES = ["edad_mayor6", "sexo_femenino", "cefalea", "mialgia", "dolor_abdominal", "exantema",
                   "vomito", "disnea", "sangrado", "hipotension", "organomegalia"]
NUMERIC_FEATURES = ["dias_fiebre", "hb", "pcv", "wbc_min", "wbc_max", "plaquetas", "alt", "creatinina"]
FEATURES = BINARY_FEATURES + NUMERIC_FEATURES
LOG_FEATURES = ["wbc_min", "wbc_max", "plaquetas", "alt", "creatinina"]  # asimétricas -> log1p
EXCLUDED = {
    "fluidos_iv": "Tratamiento: se indica después de evaluar al paciente (fuga de información).",
    "antibioticos": "Tratamiento: se indica después de evaluar al paciente (fuga de información).",
    "antipireticos": "Tratamiento: se indica después de evaluar al paciente (fuga de información).",
    "esteroides": "Tratamiento: se indica después de evaluar al paciente (fuga de información).",
    "ferritina": "37 % de valores faltantes; el artículo también la excluyó de la regresión por este motivo.",
}


# --------------------------------------------------------------------------- datos
def load_raw(file) -> pd.DataFrame:
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def prepare(raw: pd.DataFrame):
    """Limpia y recodifica. Devuelve (df_limpio, informe)."""
    df = raw.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    missing_cols = [c for c in RAW_TO_INTERNAL if c not in df.columns and RAW_TO_INTERNAL[c] in FEATURES + ["grupo"]]
    if missing_cols:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(missing_cols))
    df = df.rename(columns=RAW_TO_INTERNAL)
    keep = [c for c in RAW_TO_INTERNAL.values() if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    report = {"filas_iniciales": len(df), "nulos_por_columna": df.isna().sum().to_dict()}
    n0 = len(df)
    df = df[df["grupo"].isin([1, 2])].copy()
    report["filas_sin_grupo_valido"] = n0 - len(df)
    # Recodificación (README: 1 = Sí / 2 = No; Edad 1 = <6, 2 = >6; Sexo 1 = M, 2 = F; Grupo 2 = complicado)
    df["grupo"] = (df["grupo"] == 2).astype(int)
    df["edad_mayor6"] = (df["edad_mayor6"] == 2).astype(int)
    df["sexo_femenino"] = (df["sexo_femenino"] == 2).astype(int)
    for c in YES_NO_1YES + [] :
        if c in df:
            df[c] = df[c].map({1: 1, 2: 0})
    # BP AT ADMISSION: el código 2 = hipotensión (verificado: reproduce la sensibilidad/especificidad del artículo)
    df["hipotension"] = df["hipotension"].map({1: 0, 2: 1})
    report["duplicados"] = int(df.duplicated().sum())
    report["filas_finales"] = len(df)
    report["prevalencia"] = float(df["grupo"].mean())
    report["nulos_finales"] = df[FEATURES + ["grupo"]].isna().sum().to_dict()
    return df.reset_index(drop=True), report


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Transformaciones sin parámetros aprendidos (no hay fuga): log1p de variables asimétricas."""
    X = df[FEATURES].astype(float).copy()
    for c in LOG_FEATURES:
        X[c] = np.log1p(X[c])
    return X


# --------------------------------------------------------------------------- modelos
class OversampledMLP(ClassifierMixin, BaseEstimator):
    """Red neuronal (perceptrón multicapa, retropropagación) con sobremuestreo aleatorio de la
    clase minoritaria aplicado SOLO a los datos de entrenamiento que recibe fit()."""

    def __init__(self, hidden_layer_sizes=(16, 8), alpha=1e-2, max_iter=2000, random_state=SEED):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
        small, big = (idx1, idx0) if len(idx1) < len(idx0) else (idx0, idx1)
        extra = rng.choice(small, size=len(big) - len(small), replace=True)
        idx = np.concatenate([np.arange(len(y)), extra])
        self.net_ = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                                  max_iter=self.max_iter, random_state=self.random_state,
                                  activation="relu", solver="adam")
        self.net_.fit(X[idx], y[idx])
        self.classes_ = self.net_.classes_
        return self

    def predict_proba(self, X):
        return self.net_.predict_proba(np.asarray(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _pipe(clf):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", clf)])


def _metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat), "f1": f1_score(y, yhat),
        "specificity": recall_score(y, yhat, pos_label=0), "auc": roc_auc_score(y, p),
        "cm": confusion_matrix(y, yhat, labels=[0, 1]), "fpr": fpr, "tpr": tpr,
    }


def clinical_score(pcv, platelets, wbc, alt, hypotension) -> dict:
    """Puntuación 0–7 de Predictive scoring system for risk of complications in pediatric dengue
    infection (F1000Research 2022, 11:446). Ponderaciones y puntos de corte originales."""
    parts = {
        "PCV ≥ 42 %": 1 * (pcv >= 42), "Plaquetas ≤ 75 000/mm³": 1 * (platelets <= 75000),
        "Leucocitos ≥ 7 000/mm³": 1 * (wbc >= 7000), "ALT ≥ 70 U/L": 1 * (alt >= 70),
        "Hipotensión": 3 * bool(hypotension),
    }
    return {"puntos": parts, "total": int(sum(parts.values()))}


def clinical_score_df(df: pd.DataFrame) -> pd.Series:
    return ((df["pcv"] >= 42).astype(int) + (df["plaquetas"] <= 75000).astype(int)
            + (df["wbc_max"] >= 7000).astype(int) + (df["alt"] >= 70).astype(int) + 3 * df["hipotension"]).astype(int)


def train_all(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED) -> dict:
    X, y = model_matrix(df), df["grupo"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    grids = {
        "Regresión logística": (_pipe(LogisticRegression(class_weight="balanced", max_iter=5000, solver="lbfgs")),
                                {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}),
        "Red neuronal (MLP)": (_pipe(OversampledMLP(random_state=seed)),
                               {"clf__hidden_layer_sizes": [(8,), (16, 8), (32, 16)],
                                "clf__alpha": [1e-3, 1e-2, 1e-1, 1.0]}),
    }
    out = {"models": {}, "split": {"n_train": len(ytr), "n_test": len(yte), "pos_train": int(ytr.sum()),
                                   "pos_test": int(yte.sum())}, "seed": seed, "Xte": Xte, "yte": yte}
    for name, (pipe, grid) in grids.items():
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=1).fit(Xtr, ytr)  # búsqueda SOLO en train
        best = gs.best_estimator_
        p_te = best.predict_proba(Xte)[:, 1]
        res = _metrics(yte, p_te)
        cvres = cross_validate(clone(best), X, y, cv=cv, scoring=["roc_auc", "recall", "precision", "f1", "accuracy"])
        oof = cross_val_predict(clone(best), X, y, cv=cv, method="predict_proba")[:, 1]
        res["oof"] = _metrics(y, oof)
        res["p_oof"] = oof
        res.update({
            "estimator": best, "best_params": gs.best_params_, "cv_auc_train": float(gs.best_score_),
            "cv_full": {k.replace("test_", ""): (float(v.mean()), float(v.std())) for k, v in cvres.items() if k.startswith("test_")},
            "p_test": p_te,
        })
        if name.startswith("Regresión"):
            coef = best.named_steps["clf"].coef_[0]
            res["coef"] = pd.DataFrame({"variable": FEATURES, "coef_std": coef, "OR_por_DE": np.exp(coef)}).sort_values("coef_std")
        else:
            pi = permutation_importance(best, Xte, yte, scoring="roc_auc", n_repeats=30, random_state=seed)
            res["importance"] = pd.DataFrame({"variable": FEATURES, "importancia": pi.importances_mean,
                                              "de": pi.importances_std}).sort_values("importancia")
            res["loss_curve"] = list(best.named_steps["clf"].net_.loss_curve_)
        out["models"][name] = res

    # Score clínico del artículo sobre el mismo conjunto de prueba y sobre todos los datos
    s_all = clinical_score_df(df)
    s_te = s_all.loc[Xte.index].values
    out["score"] = {
        "test": _metrics(yte, (s_te >= 2).astype(float)) | {"auc": roc_auc_score(yte, s_te)},
        "all": _metrics(y, (s_all.values >= 2).astype(float)) | {"auc": roc_auc_score(y, s_all.values)},
        "dist": pd.crosstab(s_all, df["grupo"]),
    }
    out["score"]["test"]["fpr"], out["score"]["test"]["tpr"], _ = roc_curve(yte, s_te)
    return out


def predict_patient(estimator, patient: dict) -> float:
    row = pd.DataFrame([{f: patient[f] for f in FEATURES}])
    return float(estimator.predict_proba(model_matrix(row))[0, 1])
