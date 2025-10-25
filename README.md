# 🍷 MLWines — Análisis y Modelado de Calidad del Vino

Proyecto de **Machine Learning** desarrollado como parte de la *Maestría en IA y Big Data*.  
El objetivo es analizar, preprocesar y modelar datos del **Wine Quality Dataset (UCI Repository)** para resolver dos tareas:

1. **Clasificación:** Predecir la categoría de calidad del vino (`Low`, `Medium`, `High`).
2. **Regresión:** Estimar la calidad numérica (`quality` de 3 a 9).

---

## 📂 Estructura del proyecto

```
Wines/
├── data/
│   ├── wine_data_train_classification.csv
│   └── wine_data_train_regression.csv
├── models/
│   ├── modelo_clasificacion_vino.pkl
│   └── modelo_regresion_vino.pkl
├── notebook/
│   └── 01_eda_modelado.ipynb
├── reports/
│   └── informe_final.pdf
├── EDAWines.py
└── README.md
```

---

## ⚙️ Instalación y uso

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/restrepogerard/MLWines.git
   cd MLWines
   ```

2. Crear y activar entorno virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Ejecutar análisis o notebook:
   ```bash
   jupyter notebook notebook/01_eda_modelado.ipynb
   ```

---

## 🧠 Contenido del proyecto

- **EDA (Exploratory Data Analysis):** Análisis de distribución, correlaciones y outliers.  
- **Preprocesamiento:** Normalización, codificación de variables categóricas y manejo de desbalance.  
- **Modelado:**  
  - Clasificación (Logistic Regression, RandomForest, SMOTE, F1-score macro ≥ 0.80)  
  - Regresión (RandomForestRegressor, R² ≥ 0.65)  
- **Modelos guardados en `.pkl`**  
- **Informe técnico PDF (5 páginas)**  

---

## 📚 Fuente de datos

**UCI Machine Learning Repository:**  
[https://archive.ics.uci.edu/ml/datasets/wine+quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

## 👤 Autor

**Luis Restrepo**  
Maestría en Inteligencia Artificial y Big Data  
2025
