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

# %% Visualización de datos

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


# %% Correlacion entre variables
print("\n=== CORRELACION ENTRE VARIABLES ===")

print("\n=== MATRIZ DE CORRELACIÓN (Clasificación) ===")
corr_cls = df_cls.corr(numeric_only=True)
print(corr_cls)

print("\n=== MATRIZ DE CORRELACIÓN (Regresión) ===")
corr_reg = df_reg.corr(numeric_only=True)
print(corr_reg)

# Visualización de la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(corr_cls, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación - Dataset de Clasificación')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_reg, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación - Dataset de Regresión')
plt.show()

# Identificar correlaciones más fuertes (positivas y negativas)
print("\n=== CORRELACIONES FUERTES (Clasificación) ===")
corr_pairs_cls = corr_cls.unstack().sort_values(ascending=False)
print(corr_pairs_cls[(corr_pairs_cls < 1) & (abs(corr_pairs_cls) > 0.5)])

print("\n=== CORRELACIONES FUERTES (Regresión) ===")
corr_pairs_reg = corr_reg.unstack().sort_values(ascending=False)
print(corr_pairs_reg[(corr_pairs_reg < 1) & (abs(corr_pairs_reg) > 0.5)])

# %%  Preparación de datos para modelado

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("\n=== PREPARACIÓN DE DATOS PARA MODELADO ===")

# Copias de los datasets originales
data_cls = df_cls.copy()
data_reg = df_reg.copy()

# --- 1. Eliminación de duplicados y valores nulos ---
data_cls.drop_duplicates(inplace=True)
data_reg.drop_duplicates(inplace=True)
data_cls.dropna(inplace=True)
data_reg.dropna(inplace=True)

print(f"Dataset Clasificación limpio: {data_cls.shape[0]} filas, {data_cls.shape[1]} columnas")
print(f"Dataset Regresión limpio: {data_reg.shape[0]} filas, {data_reg.shape[1]} columnas")

# --- 2. Separar variables predictoras (X) y objetivo (y) ---
# Ajusta el nombre de la columna objetivo según tu dataset (por ejemplo 'quality' o 'target')
target_cls = 'quality_label'  # 
target_reg = 'quality'  # 

X_cls = data_cls.drop(columns=[target_cls])
y_cls = data_cls[target_cls]

X_reg = data_reg.drop(columns=[target_reg])
y_reg = data_reg[target_reg]

# --- 3. Codificación de variables categóricas (si existen) ---
cat_cols_cls = X_cls.select_dtypes(include=['object']).columns
cat_cols_reg = X_reg.select_dtypes(include=['object']).columns

if len(cat_cols_cls) > 0:
    print(f"Codificando variables categóricas (Clasificación): {list(cat_cols_cls)}")
    X_cls = pd.get_dummies(X_cls, columns=cat_cols_cls, drop_first=True)

if len(cat_cols_reg) > 0:
    print(f"Codificando variables categóricas (Regresión): {list(cat_cols_reg)}")
    X_reg = pd.get_dummies(X_reg, columns=cat_cols_reg, drop_first=True)

# --- 4. Escalado de variables numéricas ---
scaler = StandardScaler()

X_cls_scaled = scaler.fit_transform(X_cls)
X_reg_scaled = scaler.fit_transform(X_reg)

print("Variables numéricas escaladas correctamente.")

# --- 5. División en entrenamiento y prueba ---
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls_scaled, y_cls, test_size=0.3, random_state=42, stratify=y_cls)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_scaled, y_reg, test_size=0.3, random_state=42)

print("Conjuntos de entrenamiento y prueba creados exitosamente.")
print(f"Clasificación -> Entrenamiento: {X_train_cls.shape}, Prueba: {X_test_cls.shape}")
print(f"Regresión -> Entrenamiento: {X_train_reg.shape}, Prueba: {X_test_reg.shape}")


# %%  Analisis multivariado

print("\n=== ANÁLISIS MULTIVARIADO ===")

# --- 1. Pares de variables numéricas más relevantes ---
# Muestra cómo se relacionan las variables entre sí y con la variable objetivo
num_cols_cls = df_cls.select_dtypes(include=[np.number]).columns.tolist()
num_cols_reg = df_reg.select_dtypes(include=[np.number]).columns.tolist()

# Limitar el número de columnas para evitar gráficos muy cargados
cols_to_plot_cls = num_cols_cls[:5] + ['target'] if 'target' in df_cls.columns else num_cols_cls[:5]
cols_to_plot_reg = num_cols_reg[:5] + ['target'] if 'target' in df_reg.columns else num_cols_reg[:5]

# Pairplot (gráficos de dispersión por pares)
sns.pairplot(df_cls[cols_to_plot_cls], diag_kind='kde')
plt.suptitle("Análisis Multivariado - Dataset de Clasificación", y=1.02)
plt.show()

sns.pairplot(df_reg[cols_to_plot_reg], diag_kind='kde')
plt.suptitle("Análisis Multivariado - Dataset de Regresión", y=1.02)
plt.show()

# --- 2. Relación entre la variable objetivo y algunas características ---
if 'target' in df_cls.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='target', y=df_cls.columns[0], data=df_cls)
    plt.title(f'Relación entre {df_cls.columns[0]} y la variable objetivo (Clasificación)')
    plt.show()

if 'target' in df_reg.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='target', y=df_reg.columns[0], data=df_reg)
    plt.title(f'Relación entre {df_reg.columns[0]} y la variable objetivo (Regresión)')
    plt.show()

# --- 3. Mapa de calor de correlaciones fuertes ---
plt.figure(figsize=(10, 8))
corr = df_cls.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Mapa de calor - Relaciones multivariadas (Clasificación)")
plt.show()


# %%# limpieza y transformación de datos

print("\n=== LIMPIEZA Y TRANSFORMACIÓN DE DATOS ===")

# Copias de seguridad
df_cls_clean = df_cls.copy()
df_reg_clean = df_reg.copy()

# --- 1. ELIMINAR DUPLICADOS Y VALORES NULOS ---
print("Antes de limpieza:")
print(f"Clasificación: {df_cls_clean.shape[0]} filas | {df_cls_clean.duplicated().sum()} duplicados")
print(f"Regresión: {df_reg_clean.shape[0]} filas | {df_reg_clean.duplicated().sum()} duplicados")

# Eliminar duplicados y filas con nulos
df_cls_clean.drop_duplicates(inplace=True)
df_reg_clean.drop_duplicates(inplace=True)
df_cls_clean.dropna(inplace=True)
df_reg_clean.dropna(inplace=True)

print("Después de limpieza:")
print(f"Clasificación: {df_cls_clean.shape}, Regresión: {df_reg_clean.shape}")

# --- 2. DETECCIÓN Y TRATAMIENTO DE OUTLIERS ---
def eliminar_outliers(df, columnas):
    for c in columnas:
        Q1 = df[c].quantile(0.25)
        Q3 = df[c].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        df = df[(df[c] >= lim_inf) & (df[c] <= lim_sup)]
    return df

num_cols_cls = df_cls_clean.select_dtypes(include=[np.number]).columns
num_cols_reg = df_reg_clean.select_dtypes(include=[np.number]).columns

df_cls_clean = eliminar_outliers(df_cls_clean, num_cols_cls)
df_reg_clean = eliminar_outliers(df_reg_clean, num_cols_reg)

print(f"Datos sin outliers -> Clasificación: {df_cls_clean.shape}, Regresión: {df_reg_clean.shape}")

# --- 3. CODIFICACIÓN DE VARIABLES CATEGÓRICAS ---
# Variable 'type' (vino tinto/blanco) se convierte en 0 y 1
df_cls_clean = pd.get_dummies(df_cls_clean, columns=['type'], drop_first=True)
df_reg_clean = pd.get_dummies(df_reg_clean, columns=['type'], drop_first=True)
print("Codificación de variable 'type' completada.")

# --- 4. ESCALAMIENTO DE VARIABLES NUMÉRICAS ---
scaler = MinMaxScaler()
num_cols_cls = df_cls_clean.select_dtypes(include=[np.number]).columns
num_cols_reg = df_reg_clean.select_dtypes(include=[np.number]).columns

df_cls_clean[num_cols_cls] = scaler.fit_transform(df_cls_clean[num_cols_cls])
df_reg_clean[num_cols_reg] = scaler.fit_transform(df_reg_clean[num_cols_reg])

print("Escalamiento Min-Max aplicado correctamente (rango 0–1).")

# --- 5. GUARDAR DATASETS TRANSFORMADOS ---
df_cls_clean.to_csv("data/wine_class_clean.csv", index=False)
df_reg_clean.to_csv("data/wine_reg_clean.csv", index=False)
print("✅ Datasets limpios y transformados guardados con éxito.")

print("EDA Wines script loaded successfully.")
# %%
