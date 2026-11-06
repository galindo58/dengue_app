"""Verificación independiente de los modelos y del score (python verify.py)."""
import core

df, rep = core.prepare(core.load_raw("data/MASTER_CHART_F1000Research.xlsx"))
assert len(df) == 303 and int(df.grupo.sum()) == 63, "El dataset no coincide con el esperado"
res = core.train_all(df)
s = res["score"]["all"]
print(f"Score clínico (≥2) sobre los 303: sens={s['recall']:.4f} esp={s['specificity']:.4f} AUC={s['auc']:.3f}  (artículo: 0.8413 / 0.7250 / 0.86)")
assert abs(s["recall"] - 0.8413) < 1e-3 and abs(s["specificity"] - 0.725) < 1e-3
for name, m in res["models"].items():
    h, o = m, m["oof"]
    print(f"\n{name}  hiperparámetros={m['best_params']}")
    print(f"  hold-out  acc={h['accuracy']:.3f} prec={h['precision']:.3f} rec={h['recall']:.3f} f1={h['f1']:.3f} auc={h['auc']:.3f}  CM={h['cm'].tolist()}")
    print(f"  CV 5-fold acc={o['accuracy']:.3f} prec={o['precision']:.3f} rec={o['recall']:.3f} f1={o['f1']:.3f} auc={o['auc']:.3f}  CM={o['cm'].tolist()}")
print("\nOK")
