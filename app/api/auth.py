import os
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel
from app.db.database import get_db
from app.db import models

router = APIRouter(prefix="/auth", tags=["auth"])

# Password hashing — bcrypt is the industry standard
# The intuition: we never store raw passwords, only their hash.
# Even if the database leaks, passwords can't be recovered.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings — tokens expire after 7 days
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# --- Pydantic schemas (request/response shapes) ---

class UserRegister(BaseModel):
    email: str
    password: str
    player_tag: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class UserOut(BaseModel):
    id: int
    email: str
    player_tag: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class PlayerTagUpdate(BaseModel):
    player_tag: str


# --- Helper functions ---

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    return jwt.encode(
        {"sub": str(user_id), "exp": expire},
        SECRET_KEY,
        algorithm=ALGORITHM
    )

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> models.User:
    """
    FastAPI dependency — extracts and validates the JWT token
    from the Authorization header, returns the current user.
    Used to protect endpoints that require authentication.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError):
        raise credentials_exception

    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise credentials_exception
    return user


# --- Endpoints ---

@router.post("/register", response_model=UserOut)
def register(data: UserRegister, db: Session = Depends(get_db)):
    """Create a new user account."""
    existing = db.query(models.User).filter(
        models.User.email == data.email
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = models.User(
        email=data.email,
        hashed_password=hash_password(data.password),
        player_tag=data.player_tag.lstrip("#").upper() if data.player_tag else None
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(data: UserRegister, db: Session = Depends(get_db)):
    """Login and return a JWT token."""
    user = db.query(models.User).filter(
        models.User.email == data.email
    ).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {"access_token": create_token(user.id), "token_type": "bearer"}


@router.get("/me", response_model=UserOut)
def get_me(current_user: models.User = Depends(get_current_user)):
    """Get current logged-in user's profile."""
    return current_user


@router.patch("/me/player-tag")
def update_player_tag(
    data: PlayerTagUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update the player's Clash Royale tag."""
    current_user.player_tag = data.player_tag.lstrip("#").upper()
    db.commit()
    return {"player_tag": current_user.player_tag}