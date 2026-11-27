import hashlib
import json
import datetime


class AutoSmartBlockchain:
    def __init__(self, dificultad: int = 2):
        self.dificultad = dificultad  # Hash debe iniciar con "00"

    @staticmethod
    def calcular_hash(
        index_bloque: int,
        timestamp: str,
        placa: str,
        evento: str,
        datos_json: str,
        previous_hash: str,
        nonce: int,
    ) -> str:
        contenido = (
            f"{index_bloque}|{timestamp}|{placa}|{evento}|"
            f"{datos_json}|{previous_hash}|{nonce}"
        )
        return hashlib.sha256(contenido.encode("utf-8")).hexdigest()

    def minar_bloque(
        self,
        index_bloque: int,
        placa: str,
        evento: str,
        datos: dict,
        previous_hash: str,
    ) -> dict:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        datos_json = json.dumps(datos, sort_keys=True, ensure_ascii=False)
        prefijo = "0" * self.dificultad
        nonce = 0

        while True:
            hash_actual = self.calcular_hash(
                index_bloque, timestamp, placa, evento, datos_json, previous_hash, nonce
            )
            if hash_actual.startswith(prefijo):
                return {
                    "index_bloque": index_bloque,
                    "timestamp": timestamp,
                    "placa": placa,
                    "evento": evento,
                    "datos_json": datos_json,
                    "previous_hash": previous_hash,
                    "nonce": nonce,
                    "hash_actual": hash_actual,
                }
            nonce += 1

    def crear_bloque_genesis(self) -> dict:
        return self.minar_bloque(
            index_bloque=0,
            placa="GENESIS-AUTOSMART",
            evento="INICIO_CADENA",
            datos={"mensaje": "Bloque Génesis del Sistema Inmutable AutoSmart Lima"},
            previous_hash="0" * 64,
        )

    def verificar_cadena(self, bloques: list[dict]) -> dict:
        if not bloques:
            return {"valida": True, "total_bloques": 0, "errores": []}

        bloques_ordenados = sorted(bloques, key=lambda b: b["index_bloque"])
        errores = []
        prefijo = "0" * self.dificultad

        for i, bloque in enumerate(bloques_ordenados):
            hash_recalculado = self.calcular_hash(
                bloque["index_bloque"],
                bloque["timestamp"],
                bloque["placa"],
                bloque["evento"],
                bloque["datos_json"],
                bloque["previous_hash"],
                bloque["nonce"],
            )

            if bloque["hash_actual"] != hash_recalculado:
                errores.append(
                    {
                        "index_bloque": bloque["index_bloque"],
                        "motivo": "Hash alterado: los datos del bloque fueron modificados sin autorización.",
                    }
                )
            elif not bloque["hash_actual"].startswith(prefijo):
                errores.append(
                    {
                        "index_bloque": bloque["index_bloque"],
                        "motivo": "Prueba de trabajo (Proof of Work) inválida.",
                    }
                )

            if i > 0:
                bloque_anterior = bloques_ordenados[i - 1]
                if bloque["previous_hash"] != bloque_anterior["hash_actual"]:
                    errores.append(
                        {
                            "index_bloque": bloque["index_bloque"],
                            "motivo": f"Enlace roto con el bloque anterior #{bloque_anterior['index_bloque']}.",
                        }
                    )

        return {
            "valida": len(errores) == 0,
            "total_bloques": len(bloques_ordenados),
            "errores": errores,
        }