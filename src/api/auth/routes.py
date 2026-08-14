"""
auth/routes.py
--------------
Monte les routes fastapi-users : inscription (POST /auth/register),
connexion/déconnexion (POST /auth/cookie/login, /auth/cookie/logout),
consultation/mise à jour du compte courant (GET/PATCH /users/me).

Vérification d'email et réinitialisation de mot de passe par email sont
volontairement omises pour l'instant : elles nécessitent un service d'envoi
d'email pas encore en place dans le projet (voir plan).
"""

from __future__ import annotations

from fastapi import APIRouter

from ..schemas import UserCreate, UserRead, UserUpdate
from .users import auth_backend, fastapi_users

router = APIRouter()

router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/cookie",
    tags=["auth"],
)
router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)