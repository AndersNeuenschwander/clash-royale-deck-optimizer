from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from app.db.database import get_db
from app.db import models
from app.api.auth import get_current_user

router = APIRouter(prefix="/user", tags=["user"])


class SaveSnapshotRequest(BaseModel):
    player_tag: str
    arena: Optional[str]
    cards: list[str]
    analysis: dict
    weighted_score: Optional[float]


class SaveFavoriteRequest(BaseModel):
    label: Optional[str]
    cards: list[str]
    notes: Optional[str]
    analysis: Optional[dict]


@router.post("/snapshots")
def save_snapshot(
    data: SaveSnapshotRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save a deck analysis snapshot for the current user."""
    snapshot = models.DeckSnapshot(
        user_id=current_user.id,
        player_tag=data.player_tag,
        arena=data.arena,
        cards=data.cards,
        analysis=data.analysis,
        weighted_score=str(data.weighted_score),
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    return {"id": snapshot.id, "created_at": snapshot.created_at}


@router.get("/snapshots")
def get_snapshots(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all deck snapshots for the current user."""
    snapshots = db.query(models.DeckSnapshot).filter(
        models.DeckSnapshot.user_id == current_user.id
    ).order_by(models.DeckSnapshot.created_at.desc()).all()

    return [
        {
            "id": s.id,
            "arena": s.arena,
            "cards": s.cards,
            "analysis": s.analysis,
            "weighted_score": s.weighted_score,
            "created_at": s.created_at,
        }
        for s in snapshots
    ]


@router.post("/favorites")
def save_favorite(
    data: SaveFavoriteRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Star/favorite a deck."""
    fav = models.Favorite(
        user_id=current_user.id,
        label=data.label,
        cards=data.cards,
        notes=data.notes,
        analysis=data.analysis,
    )
    db.add(fav)
    db.commit()
    db.refresh(fav)
    return {"id": fav.id, "created_at": fav.created_at}


@router.get("/favorites")
def get_favorites(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all favorited decks for the current user."""
    favs = db.query(models.Favorite).filter(
        models.Favorite.user_id == current_user.id
    ).order_by(models.Favorite.created_at.desc()).all()

    return [
        {
            "id": f.id,
            "label": f.label,
            "cards": f.cards,
            "notes": f.notes,
            "analysis": f.analysis,
            "created_at": f.created_at,
        }
        for f in favs
    ]


@router.delete("/favorites/{favorite_id}")
def delete_favorite(
    favorite_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a favorited deck."""
    fav = db.query(models.Favorite).filter(
        models.Favorite.id == favorite_id,
        models.Favorite.user_id == current_user.id
    ).first()
    if not fav:
        raise HTTPException(status_code=404, detail="Favorite not found")
    db.delete(fav)
    db.commit()
    return {"deleted": True}