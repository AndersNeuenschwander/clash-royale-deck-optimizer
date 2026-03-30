# ⚔️ Clash Royale Deck Optimizer

A full-stack web application that connects to the official Clash Royale API, analyzes a player's current deck, and recommends specific card swaps based on **real battle data from their trophy range**.

> **Core insight**: Deck quality is not absolute — it is relative to what you are actually facing in your arena. A deck that performs well at 3,000 trophies may be poorly suited for 11,000 trophies.

---

## 🏗️ System Architecture

The application is organized into four distinct layers that communicate in one direction — each layer depends only on the layer below it, never the other way around. This makes the codebase easier to test, modify, and scale.

```
app/
├── data/       # API client, card datasets, card loader
├── core/       # Domain models, analyzer, meta weighter, optimizer
├── api/        # FastAPI route handlers (auth, user data)
├── db/         # SQLAlchemy database models and session management
├── static/     # Frontend HTML/CSS/JavaScript (single file)
tests/          # Test scripts for each layer
```

### Request Flow

```
Browser → GET /player/{tag}
        → Fetch player profile + last 25 battle logs (concurrent)
        → Analyze battle log → compute arena meta weights
        → Map deck to Card objects
        → DeckAnalyzer scores deck across 6 vulnerability factors
        → DeckOptimizer runs greedy search across all single-card swaps
        → Serialize to JSON → return to browser
        → (If logged in) Save DeckSnapshot to PostgreSQL
```

---

## 📡 Data Layer

### Clash Royale API Client (`app/data/client.py`)

Connects to Supercell's official REST API via the [RoyaleAPI proxy](https://proxy.royaleapi.dev) (required for dynamic IPs). Uses `httpx.AsyncClient` for non-blocking async requests — multiple users can be analyzed concurrently.

| Method | Description |
|--------|-------------|
| `get_cards()` | Fetches all 120+ playable cards with icon URLs |
| `get_player(tag)` | Fetches player profile including current 8-card deck |
| `get_location_rankings(location_id)` | Fetches top player rankings for a region |

### Card Registry (`app/data/card_loader.py`)

Supercell's API returns only basic card data. The app merges it with four JSON stat files from RoyaleAPI's GitHub (characters, spells, buildings) — analogous to a **SQL left join** on the card's unique key. The resulting registry is a Python dictionary of 120 `Card` objects loaded once at startup.

Key extracted fields: `attacks_ground`, `attacks_air`, `area_damage_radius`, `multiple_targets`, `hitpoints`, `spawn_character`

Hand-defined overlays (requiring game knowledge no dataset captures):
- `WIN_CONDITION_KEYS` — 20 cards whose primary role is threatening towers
- `SWARM_KEYS` — cards that deploy 3+ units at once
- `ALL_SPELL_KEYS` / `SPLASH_SPELL_KEYS` — spell classifications

---

## 🧱 Domain Models (`app/core/models.py`)

Implemented as Python `dataclasses` with `enums` for type safety — any typo in a card type becomes an immediate error rather than a silent bug.

| Enum | Values |
|------|--------|
| `CardType` | TROOP, SPELL, BUILDING |
| `Rarity` | COMMON, RARE, EPIC, LEGENDARY, CHAMPION |
| `DamageType` | SINGLE, SPLASH, CHAIN |
| `TargetType` | GROUND, AIR, BOTH, BUILDINGS |

**`VulnerabilityProfile`** — the output of the analyzer: six float scores (0–1) representing deck exposure to each threat archetype, plus a `win_condition_score` for offensive strength.

---

## 🔍 Deck Analyzer (`app/core/analyzer.py`)

Computes a `VulnerabilityProfile` for any deck. Each factor is scored independently on a 0–1 scale where **1 = maximally vulnerable**. The design mirrors factor analysis in quantitative finance — each score represents a specific risk exposure.

| Factor | Description |
|--------|-------------|
| **Air Exposure** | Cards that can target air units. 0 air counters → score 1.0 |
| **Tank Exposure** | Spells, buildings, and high-cost troops as tank answers |
| **Swarm Exposure** | Splash damage cards. Zero splash → score 1.0 |
| **Spell Exposure** | Swarm/"spell bait" cards vulnerable to area spells |
| **Building Exposure** | Win condition decks with no spells to remove buildings |
| **Elixir Risk** | Weighted combo of elixir curve risk + cycle risk |
| **Win Condition Score** | Offensive strength metric (offsets vulnerability in optimizer) |

---

## ⚖️ Meta Weighter (`app/core/meta_weighter.py`)

The component that makes the optimizer **arena-aware**. Rather than treating all vulnerability factors equally, it uses real battle data to weight scores by how often each threat actually appears in the player's trophy range.

> The analogy to quantitative finance is direct: just as a portfolio manager weights factor exposures by how likely each factor is to materialize, the meta weighter weights deck vulnerabilities by how often those threats appear in the player's arena.

### How It Works
1. Fetch the player's last 25 battle logs
2. Examine **only the opponent's deck** (what you face, not what you play)
3. Count threat archetype frequency → compute threat rates (0–1)
4. Add 0.1 baseline so no factor ever gets zero weight
5. Normalize to a weight vector summing to 1.0

### Example — Summit of Heroes (~11,000 trophies)

| Threat | Rate | Weight |
|--------|------|--------|
| Spell threats | 90% | 32% |
| Swarm threats | 57% | 21% |
| Building threats | 53% | 19% |
| Tank threats | 30% | 11% |
| Air threats | 13% | 7% |
| High elixir | 13% | 7% |

---

## 🤖 Deck Optimizer (`app/core/optimizer.py`)

Finds the single best card swap that reduces the arena-weighted vulnerability score using a **greedy local search**.

### Why Greedy?

Exhaustive search over all 8-card combinations from 120 cards = **~800 million combinations** — infeasible per request. Greedy local search evaluates at most **8 × 112 = 896 combinations**, running in under a second.

### Algorithm
1. Score current deck with `apply_meta_weights()`
2. For each of the 8 cards: try replacing with every card NOT in the deck (up to 112)
3. Record all swaps that improve the score
4. Sort by score delta, deduplicate on `card_in`
5. Return `best_swap` + top 4 alternatives

Each `SwapSuggestion` includes: card removed, card added, before/after scores, improvement %, and a human-readable explanation of which exposure factors improved and by how much.

---

## 🔐 Authentication & User Data

- Passwords hashed with **bcrypt** before storage
- Sessions managed with **JWT tokens** (7-day expiry), stored in `localStorage`
- Every analysis auto-saved as a `DeckSnapshot` in PostgreSQL — analogous to a portfolio performance history

### Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/auth/register` | Create account |
| POST | `/auth/login` | Login, returns JWT |
| GET | `/auth/me` | Current user profile |
| GET | `/player/{tag}` | Core analysis endpoint |
| GET | `/user/snapshots` | Deck analysis history |
| POST | `/user/favorites` | Star/save a deck |

---

## 🖥️ Frontend (`app/static/index.html`)

Single HTML file with embedded CSS and JavaScript — no framework, no build step, no npm. Trivially deployable alongside the FastAPI backend.

**Main panels:**
- Current Deck (4×2 card image grid)
- Vulnerability Profile (6 color-coded exposure bars)
- Arena Meta (threat rate percentages from real battle data)
- Factor Weights (normalized weight bars)
- Swap Suggestions (up to 5 recommendations with card images + improvement %)

---

## 🚀 Deployment

Designed for deployment on **Railway** — FastAPI backend + PostgreSQL as separate services. Frontend served as a static file by FastAPI.

**Required environment variables:**
```
CLASH_ROYALE_API_KEY   # Supercell developer API token
DATABASE_URL           # PostgreSQL connection string (Railway injects automatically)
SECRET_KEY             # Random string for signing JWT tokens
```

---

## 🔮 Planned Improvements

- Score calibration to prevent optimizer scores from bottoming out at 0.000
- Win condition compatibility (avoid suggesting conflicting win conditions)
- Elixir curve constraint on suggested decks
- Card level awareness (only suggest cards the player owns and has leveled)
- Deck history timeline UI
- Increase battle sample size to 50–100 for more accurate meta weights

---

## 📄 Full Technical Overview

See [`clash_optimizer_overview.pdf`](clash_optimizer_overview.pdf) for the complete technical documentation.
