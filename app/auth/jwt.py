import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

from app.config import settings


class JWTError(Exception):
    pass


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            options={"require": ["exp", "iat", "sub", "role"]},
        )
    except ExpiredSignatureError as e:
        raise JWTError("expired") from e
    except InvalidTokenError as e:
        raise JWTError("invalid") from e
    if payload.get("role") != "trader":
        raise JWTError("invalid_role")
    return payload
