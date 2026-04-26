"""Dev helper to mint JWTs. Usage: python -m scripts.mint_token <userId> [hours]"""
import sys
import time

import jwt

from app.config import settings


def mint(sub: str, hours: int = 24, name: str = "Dev") -> str:
    now = int(time.time())
    payload = {"sub": sub, "iat": now, "exp": now + hours * 3600, "role": "trader", "name": name}
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


if __name__ == "__main__":
    sub = sys.argv[1] if len(sys.argv) > 1 else "f412f236-4edc-47a2-8f54-8763a6ed2ce8"
    hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    print(mint(sub, hours))
