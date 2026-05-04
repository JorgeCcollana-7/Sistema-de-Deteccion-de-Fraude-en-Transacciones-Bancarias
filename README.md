
# 🏦 Sistema de Detección de Fraude en Transacciones Bancarias

> Proyecto Final — Machine Learning Aplicado al Riesgo Financiero  
> **Autor:** Jorge Ccollana  
> **Herramientas:** Python · scikit-learn · XGBoost · LightGBM · SHAP

---

## 📋 Descripción del Proyecto

Este proyecto desarrolla un sistema de detección de fraude bancario aplicando técnicas de Machine Learning supervisado sobre un dataset de 10,000 transacciones con una tasa de incidencia del 2%. El reto central es el **desbalance severo de clases (49:1)** y la construcción de un modelo interpretable que pueda justificarse ante un Comité de Riesgos.

---

## 📁 Estructura del Repositorio

```
├── base.csv                          # Dataset original (10,000 transacciones)
├── proyecto_fraude_completo.ipynb    # Notebook principal con los 5 módulos
└── README.md                         # Este archivo
```

---

## 🗂️ Módulos del Proyecto

| Módulo | Contenido |
|---|---|
| **Módulo 1 — EDA** | Estadísticos descriptivos, distribución de variables, outliers, correlaciones, comportamiento por clase |
| **Módulo 2 — Preparación** | Feature engineering temporal, encoding, RobustScaler, split 70/15/15, SMOTE |
| **Módulo 3 — Modelos** | Random Forest, XGBoost, LightGBM con RandomizedSearchCV + StratifiedKFold · Isolation Forest (no supervisado) |
| **Módulo 4 — Evaluación** | Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, KS · Matriz de confusión · Curva ROC y Precision-Recall |
| **Módulo 5 — Interpretabilidad** | Feature importance · SHAP global (beeswarm, bar) · SHAP local (waterfall) · Análisis de umbrales · Concept drift |

---

## 📊 Dataset

| Campo | Detalle |
|---|---|
| **Filas** | 10,000 transacciones bancarias |
| **Variables** | 9 numéricas · 4 categóricas · 1 timestamp |
| **Variable objetivo** | `fraud` (0 = legítima · 1 = fraude) |
| **Desbalance** | 98% legítimas vs 2% fraudes (ratio 49:1) |
| **Valores nulos** | Ninguno |

---

## 🔧 Pipeline de Preprocesamiento

```
Datos crudos (base.csv)
        │
        ▼
1. Extracción de features temporales desde timestamp
   → hora, dia_semana, es_fin_semana, es_noche
        │
        ▼
2. Eliminación de columnas no predictivas
   → transaction_id, timestamp
        │
        ▼
3. Codificación — OrdinalEncoder (variables categóricas)
        │
        ▼
4. División Train / Validación / Test (70 / 15 / 15) con stratify=y
        │
        ▼
5. Escalado — RobustScaler (fit solo en train)
        │
        ▼
6. Balanceo — SMOTE (solo sobre train)
        │
        ▼
Datos listos para modelado
```

---

## 📈 Resultados

| Modelo | AUC-ROC | KS | Recall (u=0.35) | Veredicto |
|---|---|---|---|---|
| Random Forest | 0.5202 | — | — | Descartado |
| XGBoost | 0.4502 | — | — | Ranking invertido |
| **LightGBM** | **0.5860** | **0.205** | **26.7%** | ✅ **Seleccionado** |

> **Umbral operativo recomendado: 0.35**  
> Con este umbral, LightGBM detecta el 26.7% de los fraudes a un costo de 156 revisiones manuales adicionales por cada 1,500 transacciones.

---

## 🔍 Interpretabilidad — Hallazgos SHAP

- Las **features temporales** (`dia_semana`, `hora`) son las más importantes del modelo — el momento de la transacción es el predictor más fuerte.
- El `amount` no es tan determinante como se esperaría intuitivamente — el fraude en este dataset no se distingue por el monto sino por cuándo ocurre.
- `marital_status` aparece entre las variables top — requiere validación adicional para descartar sesgos en producción.

---

## ⚙️ Requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
pip install xgboost lightgbm shap imbalanced-learn
```

> **Nota de compatibilidad:** se requiere `numpy < 2.0` para compatibilidad con todas las librerías.
> ```bash
> pip install "numpy<2" --force-reinstall
> ```

---

## 🚀 Cómo Ejecutar

1. Clonar el repositorio
```bash
git clone https://github.com/JorgeCcollana-7/Sistema-de-Deteccion-de-Fraude-en-Transacciones-Bancarias.git
cd Sistema-de-Deteccion-de-Fraude-en-Transacciones-Bancarias
```

2. Instalar dependencias
```bash
pip install "numpy<2" pandas matplotlib seaborn scikit-learn scipy xgboost lightgbm shap imbalanced-learn
```

3. Abrir el notebook
```bash
jupyter notebook proyecto_fraude_completo.ipynb
```

4. Ejecutar todas las celdas en orden (`Kernel → Restart & Run All`)

---

## 📌 Decisiones Técnicas Clave

| Decisión | Justificación |
|---|---|
| **RobustScaler** sobre StandardScaler | `amount` tiene curtosis = 268 — la media y std no son representativas |
| **SMOTE** solo en train | Evita data leakage — validación y test reflejan distribución real |
| **StratifiedKFold** en CV | Preserva el 2% de fraudes en cada fold |
| **AUC-ROC** como métrica de selección | Independiente del umbral — permite comparación justa bajo desbalance |
| **Umbral 0.35** en lugar de 0.5 | El modelo fue entrenado con datos 50/50 (SMOTE) — el umbral por defecto no es óptimo |

---

## ⚠️ Limitaciones y Próximos Pasos

**Limitaciones actuales:**
- Dataset semi-sintético con variables limitadas
- Sin historial de dispositivo, IP ni geolocalización
- Sin validación Out-of-Time (OOT)
- Las probabilidades están infladas por el entrenamiento con SMOTE

**Mejoras para V2:**
- Incorporar variables de comportamiento del cliente
- Calibrar probabilidades con `CalibratedClassifierCV`
- Evaluar entrenamiento sin SMOTE usando solo `class_weight`
- Implementar validación OOT antes del despliegue

---

## 📄 Licencia

Este proyecto fue desarrollado con fines académicos.
