import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


class AutoSmartML:
    def __init__(self, csv_path: str = "autosmart_1000.csv"):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.feature_cols = [
            "Kilometraje",
            "Antiguedad",
            "KmDesdeUltimoServicio",
            "MesesDesdeUltimoServicio",
            "FallasPrevias",
            "FallasMotor",
            "FallasFrenos",
            "FallasElectricas",
            "UsoPromedioDiario",
        ]
        self.modelo_1 = None
        self.modelo_2 = None
        self.metrics_cache = {}
        self._entrenar_modelos()

    def _entrenar_modelos(self):
        # ---------------- MODELO 1: NECESIDAD DE MANTENIMIENTO ----------------
        X = self.df[self.feature_cols]
        y = self.df["NecesitaMantenimiento"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )

        self.modelo_1 = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
        self.modelo_1.fit(X_train, y_train)

        y_pred = self.modelo_1.predict(X_test)
        probas = self.modelo_1.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probas)

        # Reducir puntos ROC a ~40 para renderizado fluido en frontend
        step = max(1, len(fpr) // 40)
        roc_points = [
            {"x": round(float(fpr[i]), 4), "y": round(float(tpr[i]), 4)}
            for i in range(0, len(fpr), step)
        ]
        if roc_points[-1]["x"] != 1.0:
            roc_points.append({"x": 1.0, "y": 1.0})

        importancias = [
            {"variable": col, "importancia": round(float(imp), 4)}
            for col, imp in zip(self.feature_cols, self.modelo_1.feature_importances_)
        ]
        importancias.sort(key=lambda item: item["importancia"], reverse=True)

        # ---------------- MODELO 2: TIPO DE SERVICIO ----------------
        df_serv = self.df[self.df["NecesitaMantenimiento"] == 1].copy()
        X2 = df_serv[self.feature_cols]
        y2 = df_serv["TipoServicio"]

        X2_train, X2_test, y2_train, y2_test = train_test_split(
            X2, y2, test_size=0.30, random_state=42, stratify=y2
        )

        self.modelo_2 = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
        self.modelo_2.fit(X2_train, y2_train)
        y2_pred = self.modelo_2.predict(X2_test)
        clases_m2 = list(self.modelo_2.classes_)

        self.metrics_cache = {
            "total_registros": int(len(self.df)),
            "modelo_1": {
                "muestra_test": int(len(y_test)),
                "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
                "precision": round(float(precision_score(y_test, y_pred)) * 100, 2),
                "recall": round(float(recall_score(y_test, y_pred)) * 100, 2),
                "f1_score": round(float(f1_score(y_test, y_pred)) * 100, 2),
                "roc_auc": round(float(roc_auc_score(y_test, probas)), 4),
                "matriz_confusion": confusion_matrix(y_test, y_pred).tolist(),
                "roc_curve": roc_points,
                "importancia_variables": importancias,
            },
            "modelo_2": {
                "muestra_test": int(len(y2_test)),
                "accuracy": round(float(accuracy_score(y2_test, y2_pred)) * 100, 2),
                "precision": round(
                    float(precision_score(y2_test, y2_pred, average="weighted", zero_division=0)) * 100, 2
                ),
                "recall": round(
                    float(recall_score(y2_test, y2_pred, average="weighted", zero_division=0)) * 100, 2
                ),
                "f1_score": round(
                    float(f1_score(y2_test, y2_pred, average="weighted", zero_division=0)) * 100, 2
                ),
                "clases": clases_m2,
                "matriz_confusion": confusion_matrix(
                    y2_test, y2_pred, labels=clases_m2
                ).tolist(),
            },
        }

    def predecir(self, datos: dict) -> dict:
        nuevo = pd.DataFrame([{col: datos[col] for col in self.feature_cols}])
        pred = int(self.modelo_1.predict(nuevo)[0])
        prob = self.modelo_1.predict_proba(nuevo)[0]

        resultado = {
            "necesita_mantenimiento": bool(pred == 1),
            "probabilidad_no": round(float(prob[0]) * 100, 2),
            "probabilidad_si": round(float(prob[1]) * 100, 2),
            "servicio_estimado": "Sin mantenimiento próximo",
            "probabilidades_servicio": [],
        }

        if pred == 1:
            serv = str(self.modelo_2.predict(nuevo)[0])
            prob_serv = self.modelo_2.predict_proba(nuevo)[0]
            ranking = [
                {"servicio": str(clase), "probabilidad": round(float(p) * 100, 2)}
                for clase, p in zip(self.modelo_2.classes_, prob_serv)
            ]
            ranking.sort(key=lambda x: x["probabilidad"], reverse=True)
            resultado["servicio_estimado"] = serv
            resultado["probabilidades_servicio"] = ranking

        return resultado