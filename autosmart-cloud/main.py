import random
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ml_engine import AutoSmartML
from blockchain import AutoSmartBlockchain
from database import AutoSmartDB
from security import (
    verify_password,
    create_access_token,
    get_current_user,
    require_roles,
    enmascarar_nombre,
    enmascarar_telefono,
)

app = FastAPI(
    title="AutoSmart Cloud Platform API",
    description="Sistema Digital con Machine Learning, Cloud (Supabase), Blockchain y Ciberseguridad",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar motores al arrancar
ml_engine = AutoSmartML("autosmart_1000.csv")
blockchain_engine = AutoSmartBlockchain(dificultad=2)
db = AutoSmartDB(blockchain_engine)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------- ESQUEMAS PYDANTIC (VALIDACIÓN ANTI-INYECCIÓN) ----------------
class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterClientRequest(BaseModel):
    nombre: str = Field(..., min_length=3, max_length=80)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=6, max_length=50)

class PredictRequest(BaseModel):
    placa: str = Field(..., min_length=5, max_length=12)
    Kilometraje: int = Field(..., ge=0, le=1000000)
    Antiguedad: int = Field(..., ge=0, le=60)
    KmDesdeUltimoServicio: int = Field(..., ge=0, le=500000)
    MesesDesdeUltimoServicio: int = Field(..., ge=0, le=240)
    FallasPrevias: int = Field(..., ge=0, le=100)
    FallasMotor: int = Field(..., ge=0, le=50)
    FallasFrenos: int = Field(..., ge=0, le=50)
    FallasElectricas: int = Field(..., ge=0, le=50)
    UsoPromedioDiario: float = Field(..., ge=0, le=1000)


class OrdenCreateRequest(BaseModel):
    placa: str = Field(..., min_length=5, max_length=12)
    cliente_nombre: str = Field(..., min_length=3, max_length=80)
    cliente_telefono: str = Field(..., min_length=7, max_length=20)
    vehiculo_modelo: str = Field(..., min_length=2, max_length=60)
    kilometraje: int = Field(..., ge=0, le=1000000)
    diagnostico: str = "Recepción inicial registrada"


class EstadoUpdateRequest(BaseModel):
    estado: str
    diagnostico: str | None = None


# ---------------- INTERFAZ WEB ----------------
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/api/status")
def system_status():
    return {
        "cloud_provider": "Supabase PostgreSQL (DBaaS)" if db.cloud_connected else "Modo Local Demo (Configurar .env)",
        "cloud_connected": db.cloud_connected,
        "total_bloques": len(db.listar_bloques()),
        "total_ordenes": len(db.listar_ordenes()),
    }


# ---------------- AUTENTICACIÓN Y CIBERSEGURIDAD ----------------
@app.post("/api/auth/login")
def login(data: LoginRequest, request: Request):
    ip = request.client.host if request.client else "127.0.0.1"
    user = db.obtener_usuario(data.email.strip().lower())

    if not user or not verify_password(data.password, user["password_hash"]):
        db.registrar_auditoria(
            data.email, "desconocido", "LOGIN_FALLIDO", "Credenciales incorrectas", ip
        )
        raise HTTPException(status_code=401, detail="Correo o contraseña incorrectos.")

    token = create_access_token(user["email"], user["nombre"], user["rol"])
    db.registrar_auditoria(
        user["email"], user["rol"], "LOGIN_EXITOSO", "Inicio de sesión verificado con bcrypt + JWT", ip
    )
    return {
        "access_token": token,
        "user": {
            "email": user["email"],
            "nombre": user["nombre"],
            "rol": user["rol"],
        },
    }

@app.post("/api/auth/register-cliente")
def register_cliente(data: RegisterClientRequest, request: Request):
    ip = request.client.host if request.client else "127.0.0.1"
    email = data.email.strip().lower()
    if db.obtener_usuario(email):
        raise HTTPException(status_code=400, detail="Ese correo ya está registrado.")
    from security import hash_password
    db.crear_usuario(email, data.nombre.strip(), hash_password(data.password), "cliente")
    db.registrar_auditoria(email, "cliente", "REGISTRO_CLIENTE", f"Nuevo cliente registrado: {data.nombre}", ip)
    return {"ok": True, "mensaje": "Cuenta de cliente creada. Ya puedes iniciar sesión."}

# ---------------- MACHINE LEARNING ----------------
@app.get("/api/ml/metrics")
def get_ml_metrics(user: dict = Depends(get_current_user)):
    return ml_engine.metrics_cache


@app.post("/api/ml/predict")
def predict_maintenance(
    data: PredictRequest,
    request: Request,
    user: dict = Depends(require_roles(["admin", "mecanico", "cliente"])),
):
    ip = request.client.host if request.client else "127.0.0.1"
    placa = data.placa.upper().strip()
    resultado = ml_engine.predecir(data.model_dump())

    # 1. Guardar predicción en la nube (Supabase)
    db.guardar_prediccion(
        {
            "placa": placa,
            "kilometraje": data.Kilometraje,
            "antiguedad": data.Antiguedad,
            "necesita_mantenimiento": resultado["necesita_mantenimiento"],
            "probabilidad_si": resultado["probabilidad_si"],
            "servicio_estimado": resultado["servicio_estimado"],
            "usuario_email": user["email"],
        }
    )

    # 2. Sellar el diagnóstico en Blockchain (Inmutabilidad)
    bloque = db.agregar_bloque(
        placa=placa,
        evento="PREDICCION_ML",
        datos={
            "kilometraje": data.Kilometraje,
            "necesita_mantenimiento": resultado["necesita_mantenimiento"],
            "probabilidad_si": resultado["probabilidad_si"],
            "servicio_estimado": resultado["servicio_estimado"],
            "evaluado_por": user["email"],
        },
    )

    # 3. Registrar en Auditoría
    db.registrar_auditoria(
        user["email"],
        user["rol"],
        "PREDICCION_ML",
        f"Placa {placa}: {resultado['servicio_estimado']} ({resultado['probabilidad_si']}%) -> Bloque #{bloque['index_bloque']}",
        ip,
    )

    return {"prediccion": resultado, "bloque": bloque}


# ---------------- ÓRDENES Y LIVE TRACKER ----------------
@app.get("/api/ordenes")
def list_ordenes(user: dict = Depends(get_current_user)):
    return db.listar_ordenes()


@app.post("/api/ordenes")
def create_orden(
    data: OrdenCreateRequest,
    request: Request,
    user: dict = Depends(require_roles(["admin", "recepcion"])),
):
    ip = request.client.host if request.client else "127.0.0.1"
    codigo = f"ORD-2026-{random.randint(100, 999)}"
    placa = data.placa.upper().strip()

    orden = db.crear_orden(
        {
            "codigo": codigo,
            "placa": placa,
            "cliente_nombre": data.cliente_nombre.strip(),
            "cliente_telefono": data.cliente_telefono.strip(),
            "vehiculo_modelo": data.vehiculo_modelo.strip(),
            "kilometraje": data.kilometraje,
            "estado": "Recepción",
            "diagnostico": data.diagnostico.strip(),
            "servicio_estimado": "Pendiente ML",
        }
    )

    bloque = db.agregar_bloque(
        placa=placa,
        evento="NUEVA_ORDEN",
        datos={"codigo": codigo, "kilometraje": data.kilometraje, "modelo": data.vehiculo_modelo},
    )

    db.registrar_auditoria(
        user["email"], user["rol"], "CREAR_ORDEN", f"Orden {codigo} ({placa}) registrada -> Bloque #{bloque['index_bloque']}", ip
    )
    return {"orden": orden, "bloque": bloque}


@app.patch("/api/ordenes/{codigo}/estado")
def update_orden_estado(
    codigo: str,
    data: EstadoUpdateRequest,
    request: Request,
    user: dict = Depends(require_roles(["admin", "mecanico"])),
):
    estados_validos = ["Recepción", "Diagnóstico", "Repuestos", "Reparación", "Listo"]
    if data.estado not in estados_validos:
        raise HTTPException(status_code=400, detail="Estado no válido.")

    orden = db.actualizar_estado_orden(codigo, data.estado, data.diagnostico)
    if not orden:
        raise HTTPException(status_code=404, detail="Orden no encontrada.")

    bloque = db.agregar_bloque(
        placa=orden["placa"],
        evento="CAMBIO_ESTADO",
        datos={"codigo": codigo, "nuevo_estado": data.estado, "tecnico": user["email"]},
    )

    ip = request.client.host if request.client else "127.0.0.1"
    db.registrar_auditoria(
        user["email"], user["rol"], "ACTUALIZAR_ESTADO", f"Orden {codigo} pasó a '{data.estado}'", ip
    )
    return {"orden": orden, "bloque": bloque}


@app.delete("/api/ordenes/{codigo}")
def delete_orden(
    codigo: str,
    request: Request,
    user: dict = Depends(require_roles(["admin"])),
):
    if not db.eliminar_orden(codigo):
        raise HTTPException(status_code=404, detail="Orden no encontrada.")
    ip = request.client.host if request.client else "127.0.0.1"
    db.registrar_auditoria(user["email"], user["rol"], "ELIMINAR_ORDEN", f"Orden {codigo} eliminada por Admin", ip)
    return {"ok": True}


@app.get("/api/tracker/{codigo}")
def public_live_tracker(codigo: str):
    """Endpoint público para el cliente (QR). Aplica enmascaramiento PII (Ley N.° 29733)."""
    orden = db.obtener_orden(codigo.strip().upper())
    if not orden:
        raise HTTPException(status_code=404, detail="Código de orden no encontrado.")
    return {
        "codigo": orden["codigo"],
        "placa": orden["placa"],
        "vehiculo_modelo": orden["vehiculo_modelo"],
        "cliente_enmascarado": enmascarar_nombre(orden["cliente_nombre"]),
        "telefono_enmascarado": enmascarar_telefono(orden["cliente_telefono"]),
        "estado": orden["estado"],
        "diagnostico": orden["diagnostico"],
        "servicio_estimado": orden["servicio_estimado"],
        "actualizado_en": orden.get("actualizado_en", ""),
    }


# ---------------- BLOCKCHAIN ----------------
@app.get("/api/blockchain")
def get_blockchain(user: dict = Depends(get_current_user)):
    bloques = db.listar_bloques()
    verificacion = blockchain_engine.verificar_cadena(bloques)
    return {"bloques": bloques, "verificacion": verificacion}


@app.post("/api/blockchain/tamper/{index_bloque}")
def tamper_block_demo(
    index_bloque: int,
    request: Request,
    user: dict = Depends(require_roles(["admin"])),
):
    """Simula un ataque interno alterando un bloque para demostrar cómo Blockchain lo detecta."""
    if index_bloque == 0:
        raise HTTPException(status_code=400, detail="Selecciona un bloque posterior al Génesis (#1 o superior).")
    ok = db.simular_adulteracion(index_bloque)
    if not ok:
        raise HTTPException(status_code=404, detail="Bloque no encontrado.")
    ip = request.client.host if request.client else "127.0.0.1"
    db.registrar_auditoria(
        user["email"], user["rol"], "SIMULACION_ATAQUE", f"Se alteró el bloque #{index_bloque} para prueba de integridad", ip
    )
    return {"ok": True}


@app.post("/api/blockchain/restore")
def restore_chain_demo(
    request: Request,
    user: dict = Depends(require_roles(["admin"])),
):
    db.reparar_cadena_demo()
    ip = request.client.host if request.client else "127.0.0.1"
    db.registrar_auditoria(
        user["email"], user["rol"], "RESTAURACION_CADENA", "Cadena Blockchain re-sellada tras auditoría", ip
    )
    return {"ok": True}


# ---------------- AUDITORÍA (SOLO ADMIN) ----------------
@app.get("/api/auditoria")
def get_auditoria(user: dict = Depends(require_roles(["admin"]))):
    return db.listar_auditoria()