from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.db.database import Base


class User(Base):
    """
    Core user account. Email + hashed password for auth,
    player_tag links to their Clash Royale account.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    player_tag = Column(String, nullable=True)  # set after registration
    created_at = Column(DateTime, default=datetime.utcnow)

    deck_snapshots = relationship("DeckSnapshot", back_populates="user")
    favorites = relationship("Favorite", back_populates="user")


class DeckSnapshot(Base):
    """
    A saved snapshot of a player's deck analysis at a point in time.
    This lets users track how their deck has evolved and compare scores.
    The intuition: like a portfolio snapshot in quant finance —
    you want to see how your risk profile changed over time.
    """
    __tablename__ = "deck_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    player_tag = Column(String, nullable=False)
    arena = Column(String, nullable=True)
    cards = Column(JSON, nullable=False)          # list of card names
    analysis = Column(JSON, nullable=False)        # full vulnerability profile
    weighted_score = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="deck_snapshots")


class Favorite(Base):
    """
    A deck the user has starred/favorited for quick reference.
    Useful for saving a 'best deck found so far' or
    bookmarking optimized suggestions.
    """
    __tablename__ = "favorites"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    label = Column(String, nullable=True)          # user-defined name
    cards = Column(JSON, nullable=False)           # list of card names
    notes = Column(Text, nullable=True)            # user notes
    analysis = Column(JSON, nullable=True)         # snapshot of analysis
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="favorites")