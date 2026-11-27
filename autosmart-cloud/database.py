import os
import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from security import hash_password
from blockchain import AutoSmartBlockchain

load_dotenv()


class AutoSmartDB:
    def __init__(self, blockchain: AutoSmartBlockchain):
        self.blockchain = blockchain
        self.url = os.getenv("SUPABASE_URL", "").strip()
        self.key = os.getenv("SUPABASE_KEY", "").strip()
        self.client: Client | None = None
        self.cloud_connected = False

        # Respaldo local automático si aún no se configuran llaves .env
        self._mem = {
            "usuarios": [],
            "ordenes_servicio": [],
            "predicciones_ml": [],
            "blockchain_bloques": [],
            "auditoria_logs": [],
        }

        if self.url and self.key and "TU-PROYECTO" not in self.url:
            try:
                self.client = create_client(self.url, self.key)
                # Prueba rápida de conexión
                self.client.table("usuarios").select("id").limit(1).execute()
                self.cloud_connected = True
                print("[CLOUD] Conectado exitosamente a Supabase PostgreSQL.")
            except Exception as e:
                print(f"[AVISO] No se pudo conectar a Supabase ({e}). Usando modo local temporal.")

        self._inicializar_datos_base()

    def _inicializar_datos_base(self):
        # 1. Usuarios por defecto para los 3 roles RBAC (contraseñas hasheadas con bcrypt)
        usuarios_demo = [
            ("admin@autosmart.pe", "Gabriel Apac (Jefe de Taller)", "Admin2026!", "admin"),
            ("mecanico@autosmart.pe", "Carlos Mendoza (Mecánico)", "Mecanico2026!", "mecanico"),
            ("recepcion@autosmart.pe", "Sofía Ramos (Recepción)", "Recepcion2026!", "recepcion"),
            ("cliente@autosmart.pe", "Luis Fernández (Cliente)", "Cliente2026!", "cliente"),
        ]
        for email, nombre, pwd, rol in usuarios_demo:
            if not self.obtener_usuario(email):
                self.crear_usuario(email, nombre, hash_password(pwd), rol)

        # 2. Bloque Génesis de Blockchain
        if len(self.listar_bloques()) == 0:
            genesis = self.blockchain.crear_bloque_genesis()
            self.guardar_bloque(genesis)

        # 3. Orden de ejemplo para probar el Live Tracker de inmediato
        if len(self.listar_ordenes()) == 0:
            orden_demo = {
                "codigo": "ORD-2026-001",
                "placa": "ABC-789",
                "cliente_nombre": "Luis Fernández Torres",
                "cliente_telefono": "987654321",
                "vehiculo_modelo": "Toyota Yaris 2020",
                "kilometraje": 85400,
                "estado": "Diagnóstico",
                "diagnostico": "Desgaste en pastillas de freno delanteras detectado por ML.",
                "servicio_estimado": "Frenos",
            }
            self.crear_orden(orden_demo)
            self.agregar_bloque("ABC-789", "INGRESO_VEHICULO", orden_demo)

    # ---------------- USUARIOS ----------------
    def obtener_usuario(self, email: str) -> dict | None:
        if self.cloud_connected:
            res = self.client.table("usuarios").select("*").eq("email", email).execute()
            return res.data[0] if res.data else None
        return next((u for u in self._mem["usuarios"] if u["email"] == email), None)

    def crear_usuario(self, email: str, nombre: str, password_hash: str, rol: str):
        registro = {
            "email": email,
            "nombre": nombre,
            "password_hash": password_hash,
            "rol": rol,
        }
        if self.cloud_connected:
            self.client.table("usuarios").insert(registro).execute()
        else:
            registro["id"] = len(self._mem["usuarios"]) + 1
            self._mem["usuarios"].append(registro)

    # ---------------- ÓRDENES / LIVE TRACKER ----------------
    def listar_ordenes(self) -> list[dict]:
        if self.cloud_connected:
            res = (
                self.client.table("ordenes_servicio")
                .select("*")
                .order("id", desc=True)
                .execute()
            )
            return res.data
        return list(reversed(self._mem["ordenes_servicio"]))

    def obtener_orden(self, codigo: str) -> dict | None:
        if self.cloud_connected:
            res = self.client.table("ordenes_servicio").select("*").eq("codigo", codigo).execute()
            return res.data[0] if res.data else None
        return next((o for o in self._mem["ordenes_servicio"] if o["codigo"] == codigo), None)

    def crear_orden(self, orden: dict) -> dict:
        ahora = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if self.cloud_connected:
            res = self.client.table("ordenes_servicio").insert(orden).execute()
            return res.data[0]
        orden["id"] = len(self._mem["ordenes_servicio"]) + 1
        orden["creado_en"] = ahora
        orden["actualizado_en"] = ahora
        self._mem["ordenes_servicio"].append(orden)
        return orden

    def actualizar_estado_orden(self, codigo: str, estado: str, diagnostico: str | None = None) -> dict | None:
        ahora = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cambios = {"estado": estado, "actualizado_en": ahora}
        if diagnostico:
            cambios["diagnostico"] = diagnostico

        if self.cloud_connected:
            res = (
                self.client.table("ordenes_servicio")
                .update(cambios)
                .eq("codigo", codigo)
                .execute()
            )
            return res.data[0] if res.data else None

        for o in self._mem["ordenes_servicio"]:
            if o["codigo"] == codigo:
                o.update(cambios)
                return o
        return None

    def eliminar_orden(self, codigo: str) -> bool:
        if self.cloud_connected:
            res = self.client.table("ordenes_servicio").delete().eq("codigo", codigo).execute()
            return len(res.data) > 0
        antes = len(self._mem["ordenes_servicio"])
        self._mem["ordenes_servicio"] = [
            o for o in self._mem["ordenes_servicio"] if o["codigo"] != codigo
        ]
        return len(self._mem["ordenes_servicio"]) < antes

    # ---------------- PREDICCIONES ML ----------------
    def guardar_prediccion(self, registro: dict):
        if self.cloud_connected:
            self.client.table("predicciones_ml").insert(registro).execute()
        else:
            registro["id"] = len(self._mem["predicciones_ml"]) + 1
            registro["creado_en"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._mem["predicciones_ml"].append(registro)

    # ---------------- BLOCKCHAIN ----------------
    def listar_bloques(self) -> list[dict]:
        if self.cloud_connected:
            res = (
                self.client.table("blockchain_bloques")
                .select("*")
                .order("index_bloque", desc=False)
                .execute()
            )
            return res.data
        return sorted(self._mem["blockchain_bloques"], key=lambda b: b["index_bloque"])

    def guardar_bloque(self, bloque: dict):
        if self.cloud_connected:
            self.client.table("blockchain_bloques").insert(bloque).execute()
        else:
            self._mem["blockchain_bloques"].append(bloque)

    def agregar_bloque(self, placa: str, evento: str, datos: dict) -> dict:
        bloques = self.listar_bloques()
        ultimo = bloques[-1]
        nuevo = self.blockchain.minar_bloque(
            index_bloque=ultimo["index_bloque"] + 1,
            placa=placa,
            evento=evento,
            datos=datos,
            previous_hash=ultimo["hash_actual"],
        )
        self.guardar_bloque(nuevo)
        return nuevo

    def simular_adulteracion(self, index_bloque: int) -> bool:
        datos_falsos = '{"ALERTA": "KILOMETRAJE ADULTERADO MANUALMENTE A 1000 KM"}'
        if self.cloud_connected:
            res = (
                self.client.table("blockchain_bloques")
                .update({"datos_json": datos_falsos})
                .eq("index_bloque", index_bloque)
                .execute()
            )
            return len(res.data) > 0
        for b in self._mem["blockchain_bloques"]:
            if b["index_bloque"] == index_bloque:
                b["datos_json"] = datos_falsos
                return True
        return False

    def reparar_cadena_demo(self):
        bloques = self.listar_bloques()
        prev_hash = "0" * 64
        for b in bloques:
            re_minado = self.blockchain.minar_bloque(
                index_bloque=b["index_bloque"],
                placa=b["placa"],
                evento=b["evento"],
                datos={"estado": "Registro restaurado y re-sellado por Auditoría Admin"},
                previous_hash=prev_hash,
            )
            prev_hash = re_minado["hash_actual"]
            if self.cloud_connected:
                self.client.table("blockchain_bloques").update(
                    {
                        "timestamp": re_minado["timestamp"],
                        "datos_json": re_minado["datos_json"],
                        "previous_hash": re_minado["previous_hash"],
                        "nonce": re_minado["nonce"],
                        "hash_actual": re_minado["hash_actual"],
                    }
                ).eq("index_bloque", b["index_bloque"]).execute()
            else:
                b.update(re_minado)

    # ---------------- AUDITORÍA ----------------
    def registrar_auditoria(self, email: str, rol: str, accion: str, detalle: str, ip: str = "127.0.0.1"):
        log = {
            "usuario_email": email,
            "rol": rol,
            "accion": accion,
            "detalle": detalle,
            "ip_origen": ip,
        }
        if self.cloud_connected:
            self.client.table("auditoria_logs").insert(log).execute()
        else:
            log["id"] = len(self._mem["auditoria_logs"]) + 1
            log["creado_en"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._mem["auditoria_logs"].append(log)

    def listar_auditoria(self) -> list[dict]:
        if self.cloud_connected:
            res = (
                self.client.table("auditoria_logs")
                .select("*")
                .order("id", desc=True)
                .limit(50)
                .execute()
            )
            return res.data
        return list(reversed(self._mem["auditoria_logs"]))[:50]