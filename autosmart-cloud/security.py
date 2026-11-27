import os
import datetime
import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

JWT_SECRET = os.getenv("JWT_SECRET", "AutoSmart_UTP_2026_SuperClaveSegura!")
JWT_ALGORITHM = "HS256"
JWT_EXP_HOURS = 8

bearer_scheme = HTTPBearer()


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=10)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_access_token(email: str, nombre: str, rol: str) -> str:
    payload = {
        "sub": email,
        "nombre": nombre,
        "rol": rol,
        "iat": datetime.datetime.now(datetime.timezone.utc),
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=JWT_EXP_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {
            "email": payload["sub"],
            "nombre": payload["nombre"],
            "rol": payload["rol"],
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sesión expirada. Inicia sesión nuevamente.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de seguridad inválido.",
        )


def require_roles(allowed_roles: list[str]):
    def role_checker(user: dict = Depends(get_current_user)) -> dict:
        if user["rol"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Acceso denegado (RBAC). Rol '{user['rol']}' no autorizado para esta acción.",
            )
        return user

    return role_checker


def enmascarar_nombre(nombre: str) -> str:
    partes = nombre.strip().split()
    return " ".join(p[0] + "*" * max(2, len(p) - 1) for p in partes)


def enmascarar_telefono(telefono: str) -> str:
    digitos = "".join(c for c in telefono if c.isdigit())
    if len(digitos) < 5:
        return "***"
    return f"{digitos[:3]}***{digitos[-3:]}"