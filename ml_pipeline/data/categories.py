"""
categories.py — 8-class taxonomy for the advanced news classifier.
Every model, dataset builder, and API references this single source of truth.
"""

from dataclasses import dataclass


# ─── Category definitions ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Category:
    id:          int
    name:        str
    short:       str      # slug used in filenames / API
    color:       str      # hex for UI
    icon:        str      # emoji for display
    description: str
    keywords:    tuple    # seed keywords for dataset creation helper


CATEGORIES: list[Category] = [
    Category(
        id=0, name="World, National & International", short="world", color="#3B82F6", icon="🌍",
        description="Global events, foreign affairs, diplomacy, geopolitics, UN, wars, treaties.",
        keywords=("international", "global", "diplomatic", "foreign", "united nations",
                  "sanctions", "treaty", "embassy", "conflict", "bilateral"),
    ),
    Category(
        id=1, name="Politics & Governance", short="politics", color="#EF4444", icon="🏛️",
        description="Elections, parliament, government policy, political parties, democracy.",
        keywords=("election", "parliament", "government", "policy", "minister", "vote",
                  "party", "senate", "constitution", "legislation"),
    ),
    Category(
        id=2, name="Business & Finance", short="business", color="#F59E0B", icon="💼",
        description="Stock markets, corporate earnings, mergers, startups, trade, economy.",
        keywords=("stock", "market", "revenue", "profit", "merger", "acquisition",
                  "startup", "investment", "gdp", "inflation"),
    ),
    Category(
        id=3, name="Technology", short="technology", color="#8B5CF6", icon="💻",
        description="AI, gadgets, software, cybersecurity, internet, semiconductors, space tech.",
        keywords=("artificial intelligence", "software", "hardware", "cybersecurity",
                  "smartphone", "chip", "cloud", "algorithm", "data breach", "robotics"),
    ),
    Category(
        id=4, name="Sports", short="sports", color="#10B981", icon="⚽",
        description="Cricket, football, tennis, Olympics, athletes, tournaments, match results.",
        keywords=("match", "tournament", "player", "cricket", "football", "championship",
                  "athlete", "score", "league", "olympic"),
    ),
    Category(
        id=5, name="Health & Medicine", short="health", color="#EC4899", icon="🏥",
        description="Medical breakthroughs, diseases, mental health, hospitals, pharma, fitness.",
        keywords=("hospital", "disease", "vaccine", "treatment", "mental health",
                  "doctor", "drug", "clinical trial", "pandemic", "surgery"),
    ),
    Category(
        id=6, name="Entertainment & Culture", short="entertainment", color="#F97316", icon="🎬",
        description="Movies, music, celebrities, art, festivals, Bollywood, streaming, awards.",
        keywords=("movie", "film", "music", "celebrity", "award", "festival",
                  "bollywood", "series", "concert", "release"),
    ),
    Category(
        id=7, name="Lifestyle & Society", short="lifestyle", color="#D946EF", icon="🏘️",
        description="Social issues, religion, gender, food, travel, fashion, community.",
        keywords=("community", "social", "gender", "religion", "tradition", "festival",
                  "travel", "food", "fashion", "culture"),
    ),
]


# ── Convenience lookups ────────────────────────────────────────────────────────

CATEGORY_BY_ID    = {c.id:    c for c in CATEGORIES}
CATEGORY_BY_SHORT = {c.short: c for c in CATEGORIES}
CATEGORY_BY_NAME  = {c.name:  c for c in CATEGORIES}

NUM_CLASSES   = len(CATEGORIES)                    # ← Now 8
CATEGORY_NAMES = [c.name for c in CATEGORIES]
LABEL_MAP     = {c.id: c.name for c in CATEGORIES}
LABEL_MAP_INV = {c.name: c.id for c in CATEGORIES}


# AG News (4-class) → our new 8-class mapping
AG_NEWS_REMAP = {
    0: 0,   # AG "World"      → World & International
    1: 4,   # AG "Sports"     → Sports
    2: 2,   # AG "Business"   → Business & Finance
    3: 3,   # AG "Technology" → Technology
}


def get_category(id_or_name) -> Category:
    if isinstance(id_or_name, int):
        return CATEGORY_BY_ID[id_or_name]
    return CATEGORY_BY_NAME.get(id_or_name) or CATEGORY_BY_SHORT.get(id_or_name)


if __name__ == "__main__":
    print(f"Total categories: {NUM_CLASSES}\n")
    for c in CATEGORIES:
        print(f"  [{c.id:2d}] {c.icon}  {c.name:<28}  {c.color}")