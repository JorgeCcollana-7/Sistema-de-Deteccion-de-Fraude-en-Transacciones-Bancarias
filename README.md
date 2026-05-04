#  Sistema de Detección de Fraude en Transacciones Bancarias

> Proyecto Final — Machine Learning Aplicado al Riesgo Financiero  
> **Autor:** Jorge Ccollana  
> **Herramientas:** Python · scikit-learn · XGBoost 3.2.0 · LightGBM 4.6.0 · SHAP 0.51.0

---

##  Descripción del Proyecto

Sistema de detección de fraude bancario sobre un dataset de **10,000 transacciones** con una tasa de incidencia del **2%**. El reto central es el desbalance severo de clases (ratio 49:1) y la construcción de un modelo interpretable que pueda justificarse ante un Comité de Riesgos.

---

##  Estructura del Repositorio

```
├── base.csv                          # Dataset original (10,000 transacciones)
├── proyecto_fraude_completo.ipynb    # Notebook principal con los 5 módulos
└── README.md                         # Este archivo
```

---

##  Módulos del Proyecto

| Módulo | Contenido |
|---|---|
| **Módulo 1 — EDA** | Estadísticos descriptivos · outliers · correlaciones · comportamiento por clase |
| **Módulo 2 — Preparación** | Feature engineering temporal · OrdinalEncoder · RobustScaler · split 70/15/15 · SMOTE |
| **Módulo 3 — Modelos** | Random Forest · XGBoost · LightGBM con early stopping + RandomizedSearchCV · Isolation Forest |
| **Módulo 4 — Evaluación** | Accuracy · Precision · Recall · F1 · AUC-ROC · AUC-PR · KS · Matriz de confusión · Curva ROC |
| **Módulo 5 — Interpretabilidad** | Feature importance · SHAP beeswarm · SHAP waterfall · Dependence plots · Análisis de umbrales |

---

##  Dataset

| Campo | Detalle |
|---|---|
| **Filas** | 10,000 transacciones |
| **Columnas** | 17 (9 numéricas · 4 categóricas · 1 timestamp · 1 ID · 1 target) |
| **Memoria** | 1.78 MB |
| **Variable objetivo** | `fraud` — 9,800 legítimas (98%) · 200 fraudes (2%) |
| **Desbalance** | Ratio 49:1 |
| **Valores nulos** | Ninguno |

---

##  Pipeline de Preprocesamiento

```
base.csv (10,000 filas · 17 columnas)
        │
        ▼
1. Feature engineering temporal desde timestamp
   → hora · dia_semana · dia_mes · mes · es_fin_semana · es_noche
        │
        ▼
2. Eliminar transaction_id y timestamp
        │
        ▼
3. OrdinalEncoder → 4 variables categóricas
        │
        ▼
4. Train / Validación / Test  →  7,004 / 1,496 / 1,500  (stratify=y)
   Fraude preservado: 2.00% / 2.01% / 2.00%
        │
        ▼
5. RobustScaler — fit en train, transform en val y test
   → X_train / X_val / X_test : (7004,20) / (1496,20) / (1500,20)
        │
        ▼
6. SMOTE — solo sobre train
   Antes: 6,864 legítimas · 140 fraudes (49:1)
   Después: 6,864 · 6,864 (1:1) — 6,724 muestras sintéticas generadas
```

---

## 📈 Resultados del Modelo

### Tabla Comparativa — Conjunto de Test

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC | AUC-PR | KS | Fraudes detectados |
|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.9787 | 0.0000 | 0.0000 | 0.0000 | 0.5202 | 0.0306 | 0.1510 | 0/30 (0.0%) |
| XGBoost | 0.7227 | 0.0175 | 0.2333 | 0.0326 | 0.4502 | 0.0392 | 0.0701 | 7/30 (23.3%) |
| **LightGBM**  | **0.9760** | **0.1250** | **0.0333** | **0.0526** | **0.5860** | **0.0417** | **0.2048** | **1/30 (3.3%)** |

> **Modelo seleccionado: LightGBM** — mayor AUC-ROC (0.586) y KS (0.205)

### Matriz de Confusión — LightGBM (umbral = 0.5)

| | Pred. Legítima | Pred. Fraude |
|---|---|---|
| **Real Legítima** | 1,463 (TN) | 7 (FP) |
| **Real Fraude** | 29 (FN) | 1 (TP) |

### Modelo No Supervisado — Isolation Forest

| Métrica | Fraude | Nota |
|---|---|---|
| Precision | 0.00 | — |
| Recall | 0.00 | — |
| AUC-ROC | 0.5337 | Prácticamente aleatorio |

> El fraude en este dataset no es una anomalía estadística clara. Los patrones están en la combinación de variables, no en valores extremos individuales.

---

##  Umbral Óptimo de Decisión

Con umbral por defecto (0.5) el modelo detecta 1 de 30 fraudes. El análisis de trade-off Precision-Recall identifica:

| Métrica | Umbral 0.50 | **Umbral 0.35** |
|---|---|---|
| Recall | 3.3% | **26.7%** |
| Precision | 12.5% | 4.9% |
| F1-Score | 5.3% | **8.2%** |
| Falsos Positivos | 7 | 156 |
| Falsos Negativos | 29 | **22** |

> **Recomendación operativa:** usar umbral **0.35**. El modelo pasa de detectar 1 fraude a detectar 8, a un costo de 156 revisiones manuales adicionales por cada 1,500 transacciones.

---

##  Interpretabilidad — Hallazgos SHAP

**Top variables por importancia — LightGBM:**

| Ranking | Variable | Importancia | Tipo |
|---|---|---|---|
| 1 | `dia_semana` | 59 | ⭐ Feature ingeniada (temporal) |
| 2 | `hora` | 29 | ⭐ Feature ingeniada (temporal) |
| 2 | `marital_status` | 29 | Categórica original |
| 4 | `num_dependents` | 25 | Numérica original |
| 5 | `transaction_frequency` | 23 | Numérica original |
| 5 | `housing_type` | 23 | Categórica original |
| 7 | `transaction_type` | 21 | Categórica original |

**SHAP expected value:** 0.3119 (inflado por entrenamiento con datos SMOTE 50/50)

> Las features temporales `dia_semana` y `hora` dominan el modelo — el momento de la transacción es más predictivo que el monto.


---

##  Decisiones Técnicas

| Decisión | Justificación |
|---|---|
| **RobustScaler** | `amount` curtosis=268 · `account_balance` curtosis=112 — StandardScaler sería inútil |
| **SMOTE solo en train** | Evita data leakage — val y test reflejan distribución real 98/2 |
| **StratifiedKFold n=3** | Preserva el 2% de fraudes en cada fold |
| **AUC-ROC como criterio** | Independiente del umbral — comparación justa bajo desbalance severo |
| **Umbral 0.35** | Entrenamiento con SMOTE 50/50 infla probabilidades — 0.5 no es óptimo |
| **Early stopping XGB/LGB** | XGBoost converge en 39 árboles · LightGBM en 6 — evita sobreajuste a datos sintéticos |
| **n_jobs=1** | Incompatibilidad numpy/scipy en el entorno — evita BrokenProcessPool |

---

## Limitaciones

- Dataset semi-sintético sin historial de dispositivo, IP ni geolocalización
- **Distributional shift:** modelos entrenados con SMOTE (50/50) evaluados en datos reales (98/2) — AUC en CV (≈0.99) no refleja performance real en test (0.586)
- Probabilidades infladas por SMOTE — requieren calibración con `CalibratedClassifierCV`
- Sin validación Out-of-Time (OOT)
- XGBoost obtuvo AUC=0.4502 (ranking invertido) por combinación de SMOTE + `scale_pos_weight`

---

## 📄 Licencia

Proyecto desarrollado con fines académicos.
