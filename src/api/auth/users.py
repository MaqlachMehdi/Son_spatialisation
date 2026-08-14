"""
auth/users.py
-------------
Configuration fastapi-users : accès à la table users, UserManager (logique
métier autour d'un compte), et backend d'authentification (JWT porté par un
cookie httpOnly, pas de token en localStorage côté frontend).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.authentication import AuthenticationBackend, CookieTransport, JWTStrategy
from fastapi_users.db import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_async_session
from ..models import User

JWT_SECRET = os.environ["JWT_SECRET"]
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "false").lower() == "true"
COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 jours


async def get_user_db(
    session: AsyncSession = Depends(get_async_session),
) -> AsyncGenerator[SQLAlchemyUserDatabase, None]:
    yield SQLAlchemyUserDatabase(session, User)


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = JWT_SECRET
    verification_token_secret = JWT_SECRET

    async def on_after_register(self, user: User, request: Request | None = None) -> None:
        print(f"[auth] Nouveau compte : {user.email}")


async def get_user_manager(
    user_db: SQLAlchemyUserDatabase = Depends(get_user_db),
) -> AsyncGenerator[UserManager, None]:
    yield UserManager(user_db)


# Cookie httpOnly plutôt qu'un JWT en localStorage : un XSS côté frontend ne
# peut pas lire un cookie httpOnly, donc pas voler la session. cookie_secure
# vient de l'env (False en dev http local, True en prod https).
cookie_transport = CookieTransport(
    cookie_name="son_spatialisation_auth",
    cookie_max_age=COOKIE_MAX_AGE,
    cookie_secure=COOKIE_SECURE,
    cookie_httponly=True,
    cookie_samesite="lax",
)


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=JWT_SECRET, lifetime_seconds=COOKIE_MAX_AGE)


auth_backend = AuthenticationBackend(
    name="cookie",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

# Dépendance à utiliser sur toute route qui doit être scopée à l'utilisateur
# connecté (ex: les futures routes /scenes).
current_active_user = fastapi_users.current_user(active=True)