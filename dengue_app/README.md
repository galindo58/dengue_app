# DengueAI Pediátrico

Plataforma académica (Streamlit) que entrena y compara **regresión logística** y una **red neuronal artificial**
para estimar el riesgo de complicaciones del dengue pediátrico, y la contrasta con el **score clínico 0–7**
de *Predictive scoring system for risk of complications in pediatric dengue infection* (F1000Research 2022, 11:446).

> Herramienta académica. No es un diagnóstico y no sustituye la valoración de un profesional de salud.

## Ejecutar en local
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
python verify.py                                         # opcional: verifica modelos y score
```
Use Python 3.11 o 3.12.

## Publicar y obtener el enlace público (gratis)
1. Cree un repositorio en GitHub y suba **todo el contenido de esta carpeta** (incluida `data/`).
2. Entre a https://share.streamlit.io con su cuenta de GitHub → **New app**.
3. Elija el repositorio, rama `main` y archivo principal `app.py`. En *Advanced settings* seleccione Python 3.12.
4. **Deploy**. En 2–5 minutos obtendrá un enlace `https://<nombre>.streamlit.app` para el profesor.
(La primera carga entrena los modelos, ~30–60 s; después queda en caché.)

## Metodología
- Dataset: `data/MASTER_CHART_F1000Research.xlsx` (303 niños; 63 complicados = 20,8 %).
- **Predictores (19)**: edad, sexo, días de fiebre, 9 síntomas/signos, hipotensión, hemoglobina, PCV, leucocitos mín./máx., plaquetas, ALT, creatinina.
- **Excluidos**: fluidos i/v, antibióticos, antipiréticos y esteroides (tratamientos posteriores a la valoración → fuga de información);
  ferritina (37 % de faltantes; el artículo también la excluyó).
- Sin fuga: imputación (mediana) y estandarización se ajustan solo con datos de entrenamiento (dentro de un `Pipeline`);
  hiperparámetros elegidos con validación cruzada 5-fold **solo en el entrenamiento**; log(1+x) en variables asimétricas (transformación sin parámetros).
- Desbalance: `class_weight="balanced"` (regresión logística) y sobremuestreo aleatorio solo del entrenamiento (red neuronal).
- Evaluación: hold-out estratificado 80/20 y predicciones out-of-fold con 5-fold sobre los 303 pacientes.
- Red neuronal: perceptrón multicapa de scikit-learn (`MLPClassifier`, ReLU, Adam, L2, retropropagación). No se usó TensorFlow/PyTorch.

## Nota sobre la codificación
El README del dataset describe 18 columnas, pero el Excel tiene 25 (incluye laboratorio) y no contiene la columna *Outcomes*.
`BP AT ADMISSION`: el código 2 se interpreta como hipotensión; con esa lectura (y `HIGHEST WBC` para leucocitos) el score del
artículo reproduce exactamente su sensibilidad (84,13 %) y especificidad (72,50 %).

## Limitaciones
Muestra pequeña (303; 63 complicados), un solo conjunto de datos, sin validación externa; las métricas del hold-out (61 pacientes) son muy variables.
