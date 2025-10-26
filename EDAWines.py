# %%%writefile EDAWines.py
# Librerias y carga de datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%# ETL Exploratorio de datos de vinos
# Carga de datos
# Dataset de Clasificación
df_cls = pd.read_csv("data/wine_data_train_classification.csv")
print(f"Dataset Clasificación cargado: {df_cls.shape[0]} filas, {df_cls.shape[1]} columnas")

# Dataset de Regresión
df_reg = pd.read_csv("data/wine_data_train_regression.csv")
print(f"Dataset Regresión cargado: {df_reg.shape[0]} filas, {df_reg.shape[1]} columnas")

#información básica de los datasets
print("Información del dataset de Clasificación:")
print(df_cls.info())
print("\nInformación del dataset de Regresión:")
print(df_reg.info())


# %%# #Identificación de datos faltantes, duplicados y tipos de datos incorrectos
print("\n=== NULOS Y DUPLICADOS ===")
print("Clasificación - Nulos:\n", df_cls.isnull().sum())
print("Regresión - Nulos:\n", df_reg.isnull().sum())
print(f"Duplicados Clasificación: {df_cls.duplicated().sum()}")
print(f"Duplicados Regresión: {df_reg.duplicated().sum()}")

# Análisis estadístico descriptivo
print("\nDescripción estadística del dataset de Clasificación:")
print(df_cls.describe())    
print("\nDescripción estadística del dataset de Regresión:")
print(df_reg.describe())  

# Visualización de datos

# Identificación de outliers
def detectar_outliers(df, columna):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    out = df[(df[columna] < Q1 - 1.5*IQR) | (df[columna] > Q3 + 1.5*IQR)]
    return out.shape[0]

print("\n=== OUTLIERS DETECTADOS (Clasificación) ===")
for c in df_cls.select_dtypes(include=[np.number]).columns:
    print(f"{c}: {detectar_outliers(df_cls, c)}")
    print("\n=== OUTLIERS DETECTADOS (Regresión) ===")
for c in df_reg.select_dtypes(include=[np.number]).columns:
    print(f"{c}: {detectar_outliers(df_reg, c)}")

# Visualización de distribuciones de variables numéricas
num_cols_cls = df_cls.select_dtypes(include=[np.number]).columns
num_cols_reg = df_reg.select_dtypes(include=[np.number]).columns
for col in num_cols_cls:
    plt.figure(figsize=(8, 4))
    sns.histplot(df_cls[col], kde=True)
    plt.title(f'Distribución de {col} (Clasificación)')
    plt.show()
for col in num_cols_reg:
    plt.figure(figsize=(8, 4))
    sns.histplot(df_reg[col], kde=True)
    plt.title(f'Distribución de {col} (Regresión)')
    plt.show()


# Correlación entre variables
# print("\n=== MATRIZ DE CORRELACIÓN (Clasificación) ===")
# corr_cls = df_cls.corr()
# sns.heatmap(corr_cls, annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Matriz de Correlación - Clasificación")
# plt.show()
# print("\n=== MATRIZ DE CORRELACIÓN (Regresión) ===")
# corr_reg = df_reg.corr()
# sns.heatmap(corr_reg, annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Matriz de Correlación - Regresión")
# plt.show()

# Preparación de datos para modelado
# Analisis multivariado
# limpieza y transformación de datos
print("EDA Wines script loaded successfully.")
# %%
