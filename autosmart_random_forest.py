import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)

# -------------------------------------------------
# CARGA DE DATOS
# -------------------------------------------------

df = pd.read_csv("autosmart_1000.csv")

print("\nPrimeros registros:")
print(df.head())

# -------------------------------------------------
# VARIABLES DEL MODELO 1
# PREDICE SI NECESITA MANTENIMIENTO
# -------------------------------------------------

X = df.drop(
    [
        "NecesitaMantenimiento",
        "TipoServicio"
    ],
    axis=1
)

y = df["NecesitaMantenimiento"]

# -------------------------------------------------
# ENTRENAMIENTO / TEST
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# RANDOM FOREST - MODELO 1
# -------------------------------------------------

modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

modelo.fit(
    X_train,
    y_train
)

# -------------------------------------------------
# PREDICCIONES
# -------------------------------------------------

y_pred = modelo.predict(
    X_test
)

# -------------------------------------------------
# MÉTRICAS
# -------------------------------------------------

print("\n================================")
print("MODELO 1: NECESIDAD DE MANTENIMIENTO")
print("================================")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

probas = modelo.predict_proba(X_test)[:, 1]
print("ROC AUC  :", roc_auc_score(y_test, probas))

print("\nReporte:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No necesita", "Necesita"]
    )
)

# -------------------------------------------------
# GRÁFICO 1: MATRIZ DE CONFUSIÓN - MODELO 1
# -------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No necesita", "Necesita"],
    yticklabels=["No necesita", "Necesita"]
)
plt.title("Matriz de Confusión - Necesidad de Mantenimiento")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()

# -------------------------------------------------
# GRÁFICO 2: IMPORTANCIA DE VARIABLES - MODELO 1
# -------------------------------------------------

importancias = modelo.feature_importances_

variables = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": importancias
}).sort_values(by="Importancia", ascending=False)

print("\nImportancia de Variables")
print(variables)

plt.figure(figsize=(9, 6))
sns.barplot(
    data=variables,
    x="Importancia",
    y="Variable",
    hue="Variable",
    legend=False
)
plt.title("Importancia de Variables - AutoSmart")
plt.tight_layout()

# -------------------------------------------------
# GRÁFICO 3: CURVA ROC - MODELO 1
# -------------------------------------------------

RocCurveDisplay.from_estimator(
    modelo,
    X_test,
    y_test
)
plt.title("Curva ROC - Necesidad de Mantenimiento")
plt.tight_layout()

# =================================================
# MODELO 2
# PREDICE EL TIPO DE SERVICIO
# SOLO PARA VEHÍCULOS QUE NECESITAN MANTENIMIENTO
# =================================================

df_servicio = df[df["NecesitaMantenimiento"] == 1].copy()

X2 = df_servicio.drop(
    [
        "NecesitaMantenimiento",
        "TipoServicio"
    ],
    axis=1
)

y2 = df_servicio["TipoServicio"]

# -------------------------------------------------
# ENTRENAMIENTO / TEST - MODELO 2
# -------------------------------------------------

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2,
    y2,
    test_size=0.30,
    random_state=42,
    stratify=y2
)

# -------------------------------------------------
# RANDOM FOREST - MODELO 2
# -------------------------------------------------

modelo_servicio = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

modelo_servicio.fit(X2_train, y2_train)

# -------------------------------------------------
# PREDICCIONES - MODELO 2
# -------------------------------------------------

y2_pred = modelo_servicio.predict(X2_test)

# -------------------------------------------------
# MÉTRICAS - MODELO 2
# -------------------------------------------------

print("\n================================")
print("MODELO 2: TIPO DE SERVICIO")
print("================================")

print("Accuracy :", accuracy_score(y2_test, y2_pred))
print("Precision:", precision_score(y2_test, y2_pred, average="weighted", zero_division=0))
print("Recall   :", recall_score(y2_test, y2_pred, average="weighted", zero_division=0))
print("F1 Score :", f1_score(y2_test, y2_pred, average="weighted", zero_division=0))

print("\nReporte:")
print(classification_report(y2_test, y2_pred, zero_division=0))

# -------------------------------------------------
# GRÁFICO 4: MATRIZ DE CONFUSIÓN - MODELO 2
# -------------------------------------------------

clases_servicio = modelo_servicio.classes_
cm2 = confusion_matrix(y2_test, y2_pred, labels=clases_servicio)

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm2,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=clases_servicio,
    yticklabels=clases_servicio
)
plt.title("Matriz de Confusión - Tipo de Servicio")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# MOSTRAR LAS 4 VENTANAS AL MISMO TIEMPO SIN BLOQUEAR LA CONSOLA
plt.show(block=False)
plt.pause(0.5)

# -------------------------------------------------
# PREDICCIÓN DE NUEVO VEHÍCULO
# -------------------------------------------------

while True:

    print("\n================================")
    print("PREDICTOR AUTOSMART")
    print("================================")

    kilometraje = int(input("Kilometraje total: "))
    antiguedad = int(input("Antigüedad del vehículo (años): "))
    km_ultimo = int(input("Km desde el último servicio: "))
    meses_ultimo = int(input("Meses desde el último servicio: "))
    fallas = int(input("Fallas previas totales: "))
    fallas_motor = int(input("Fallas de motor: "))
    fallas_frenos = int(input("Fallas de frenos: "))
    fallas_electricas = int(input("Fallas eléctricas: "))
    uso = float(input("Uso promedio diario (km): "))

    nuevo = pd.DataFrame({
        "Kilometraje": [kilometraje],
        "Antiguedad": [antiguedad],
        "KmDesdeUltimoServicio": [km_ultimo],
        "MesesDesdeUltimoServicio": [meses_ultimo],
        "FallasPrevias": [fallas],
        "FallasMotor": [fallas_motor],
        "FallasFrenos": [fallas_frenos],
        "FallasElectricas": [fallas_electricas],
        "UsoPromedioDiario": [uso]
    })

    # -------------------------------------------------
    # RESULTADO MODELO 1
    # -------------------------------------------------

    pred = modelo.predict(nuevo)[0]
    prob = modelo.predict_proba(nuevo)[0]

    print("\n================================")
    print("RESULTADO AUTOSMART")
    print("================================")

    if pred == 1:

        print("EL VEHÍCULO NECESITA MANTENIMIENTO")
        print(f"Probabilidad NO necesita: {prob[0]:.2%}")
        print(f"Probabilidad SÍ necesita: {prob[1]:.2%}")

        # ---------------------------------------------
        # SI NECESITA, SE EJECUTA EL MODELO 2
        # ---------------------------------------------

        servicio = modelo_servicio.predict(nuevo)[0]
        prob_servicio = modelo_servicio.predict_proba(nuevo)[0]

        print("\nServicio estimado:", servicio)

        resultados = pd.DataFrame({
            "Servicio": modelo_servicio.classes_,
            "Probabilidad": prob_servicio
        }).sort_values(by="Probabilidad", ascending=False)

        print("\nProbabilidad por servicio:")
        for fila in resultados.itertuples(index=False):
            print(f"{fila.Servicio:<30} {fila.Probabilidad:.2%}")

        # ---------------------------------------------
        # GRÁFICO DE PROBABILIDADES DEL SERVICIO
        # ---------------------------------------------

        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=resultados,
            x="Probabilidad",
            y="Servicio",
            hue="Servicio",
            legend=False
        )
        plt.title("Probabilidad del Tipo de Servicio")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)

    else:

        print("EL VEHÍCULO NO NECESITA MANTENIMIENTO PRÓXIMO")
        print(f"Probabilidad NO necesita: {prob[0]:.2%}")
        print(f"Probabilidad SÍ necesita: {prob[1]:.2%}")

    # -------------------------------------------------
    # MENÚ: PREDECIR OTRO VEHÍCULO O FINALIZAR
    # -------------------------------------------------

    while True:

        opcion = input("\n¿Deseas predecir otro vehículo? (S/N): ").strip().upper()

        if opcion == "S":
            break

        elif opcion == "N":
            print("\n================================")
            print("PROGRAMA FINALIZADO")
            print("================================")
            plt.close("all")
            raise SystemExit

        else:
            print("Opción inválida. Escribe S para continuar o N para finalizar.")