import httpx
import os
from dotenv import load_dotenv

load_dotenv()

# We route through the RoyaleAPI proxy so we don't need a static IP.
# The proxy forwards our request to api.clashroyale.com using the
# whitelisted IP we registered (45.79.218.79).
BASE_URL = "https://api.clashroyale.com/v1"


class ClashRoyaleClient:
    """
    Thin async wrapper around the Clash Royale API.
    
    The intuition here is to keep all HTTP logic in one place —
    if the API changes or we swap the proxy, we only touch this file.
    Every other module just calls these methods and gets clean data back.
    """

    def __init__(self):
        api_key = os.getenv("CLASH_ROYALE_API_KEY")
        if not api_key:
            raise ValueError("CLASH_ROYALE_API_KEY not found in environment")
        
        # httpx.AsyncClient is like requests but async — it lets FastAPI
        # handle many requests concurrently without blocking.
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0
        )

    async def get_cards(self) -> dict:
        """Fetch all cards in the game."""
        response = await self.client.get("/cards")
        response.raise_for_status()
        return response.json()

    async def get_player(self, player_tag: str) -> dict:
        """
        Fetch a player's profile including their current deck and arena.
        Player tags look like '#ABC123' — we strip the '#' since it
        needs to be URL-encoded as %23 which httpx handles for us.
        """
        tag = player_tag.strip().lstrip("#")
        response = await self.client.get(f"/players/%23{tag}")
        response.raise_for_status()
        return response.json()

    async def get_location_rankings(self, location_id: int = 57000249) -> dict:
        """
        Fetch top player rankings for a location.
        Default location is global (57000249).
        This will later help us build meta-weighted scoring per arena.
        """
        response = await self.client.get(
            f"/locations/{location_id}/rankings/players"
        )
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Always close the HTTP client when done to free connections."""
        await self.client.aclose()