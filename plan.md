# Plan: Mall Investment Simulation Platform

## Overview

Extend OASIS to simulate a shopping mall's social dynamics before a real estate investment decision. Consumer agents with demographic profiles react to a configured mall (building, tenants, transport, location) on simulated social media. Output: investment-grade analytics comparing scenarios.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHYSICAL REALITY (INPUT)                      │
│                                                                 │
│  MallConfig                                                     │
│  ├── building: floors, type, photo_spots, year_renovated        │
│  ├── tenants: [{name, category, floor, brand_tier}]             │
│  ├── transport: {metro: {distance, lines}, parking, walk_score} │
│  ├── location: rings [{radius_km, population, demographics}]    │
│  └── competitors: [{name, distance_km, positioning}]            │
│                                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              AGENT GENERATION (mall_agents_generator.py)         │
│                                                                 │
│  For each ring in location.rings:                               │
│    Generate N agents proportional to ring.population             │
│    Each agent gets:                                             │
│      - demographics from ring (age, income, type)               │
│      - transport mode (from ring distance + transport config)   │
│      - visit_probability (from accessibility model)             │
│      - mall_context (serialized MallConfig summary)             │
│                                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              SOCIAL SIMULATION (existing OASIS engine)           │
│                                                                 │
│  Each timestep:                                                 │
│    1. Accessibility model filters active agents (visit prob)    │
│    2. Active agents see mall content via recsys (existing)      │
│    3. Agents perform actions:                                   │
│       - EXISTING: create_post, like, comment, follow, share    │
│       - NEW: visit_store, write_review                          │
│    4. All actions recorded in trace table (existing)            │
│    5. Store visits recorded in visit table (new)                │
│                                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              ANALYTICS (MallAnalytics)                           │
│                                                                 │
│  Reads: trace table, visit table, post table, store table       │
│  Outputs:                                                       │
│    - foot_traffic_proxy: visit count per timestep               │
│    - store_popularity: ranked by visits + mentions              │
│    - sentiment_by_segment: positive/negative by demographic     │
│    - viral_moments: posts with high share/comment counts        │
│    - risk_factors: negative sentiment clusters                  │
│    - scenario_comparison: side-by-side metrics across configs   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Files to Create/Modify

### New Files (4)

#### 1. `oasis/social_platform/config/mall.py` — MallConfig dataclass

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TenantConfig:
    name: str
    category: str  # "F&B", "Fashion", "Entertainment", "Service"
    floor: int = 1
    brand_tier: str = "mid"  # "budget", "mid", "premium", "luxury"


@dataclass
class TransportConfig:
    metro_distance_m: int | None = None  # None = no metro
    metro_lines: list[str] = field(default_factory=list)
    bus_lines: int = 0
    parking_spots: int = 0
    parking_price_per_hour: float = 0.0
    walk_score: int = 50  # 0-100


@dataclass
class LocationRing:
    radius_km: float  # 1, 3, or 5
    population: int
    demographics: dict[str, Any] = field(default_factory=dict)
    # demographics keys: age_range, income_level, type (student/family/professional)


@dataclass
class CompetitorMall:
    name: str
    distance_km: float
    positioning: str = "general"  # "budget", "general", "premium", "luxury"


@dataclass
class BuildingConfig:
    name: str = "Mall"
    floors: int = 3
    building_type: str = "modern"  # "modern", "renovated_industrial", "luxury", "community"
    photo_spots: list[str] = field(default_factory=list)
    year_built: int | None = None
    year_renovated: int | None = None


@dataclass
class MallConfig:
    building: BuildingConfig = field(default_factory=BuildingConfig)
    tenants: list[TenantConfig] = field(default_factory=list)
    transport: TransportConfig = field(default_factory=TransportConfig)
    location_rings: list[LocationRing] = field(default_factory=list)
    competitors: list[CompetitorMall] = field(default_factory=list)

    def to_agent_context(self) -> str:
        """Serialize mall info into a text block for agent system prompts."""
        lines = []
        b = self.building
        lines.append(f'"{b.name}" is a {b.floors}-floor {b.building_type} shopping center.')
        if b.year_renovated:
            lines.append(f"Renovated in {b.year_renovated}.")
        if b.photo_spots:
            lines.append(f"Popular photo spots: {', '.join(b.photo_spots)}.")

        # Tenants
        if self.tenants:
            tenant_names = [t.name for t in self.tenants]
            lines.append(f"Stores include: {', '.join(tenant_names)}.")

        # Transport
        t = self.transport
        if t.metro_distance_m is not None:
            lines.append(
                f"Metro ({', '.join(t.metro_lines)}) is {t.metro_distance_m}m away."
            )
        if t.parking_spots > 0:
            lines.append(
                f"Parking: {t.parking_spots} spots at ¥{t.parking_price_per_hour}/hr."
            )
        lines.append(f"Walk score: {t.walk_score}/100.")

        # Competitors
        for c in self.competitors:
            lines.append(
                f"Nearby competitor: \"{c.name}\" ({c.positioning}), {c.distance_km}km away."
            )

        return "\n".join(lines)

    def get_visit_probability(self, ring_radius_km: float) -> float:
        """Simple accessibility model: closer + better transport = higher visit probability.

        ACCESSIBILITY MODEL (visit probability per timestep):
        ┌───────────────┬────────┬────────┬────────┐
        │ Transport      │ 1km    │ 3km    │ 5km    │
        ├───────────────┼────────┼────────┼────────┤
        │ Metro < 500m  │ 0.8    │ 0.5    │ 0.3    │
        │ Metro >= 500m │ 0.6    │ 0.3    │ 0.15   │
        │ No metro      │ 0.4    │ 0.15   │ 0.05   │
        └───────────────┴────────┴────────┴────────┘
        Parking bonus: +0.1 if parking_spots > 200
        """
        base_probs = {
            1: {True: 0.8, False: 0.4},
            3: {True: 0.5, False: 0.15},
            5: {True: 0.3, False: 0.05},
        }
        # Find closest ring key
        closest_ring = min(base_probs.keys(), key=lambda k: abs(k - ring_radius_km))
        has_metro = (
            self.transport.metro_distance_m is not None
            and self.transport.metro_distance_m < 500
        )
        prob = base_probs[closest_ring][has_metro]

        # Far metro penalty
        if (
            self.transport.metro_distance_m is not None
            and self.transport.metro_distance_m >= 500
            and closest_ring > 1
        ):
            prob *= 0.7

        # Parking bonus for drivers
        if self.transport.parking_spots > 200:
            prob = min(prob + 0.1, 1.0)

        return prob
```

#### 2. `oasis/social_agent/mall_agents_generator.py` — Generate agents from MallConfig

```python
import random
from typing import Any

from camel.models import BaseModelBackend

from oasis.social_agent.agent import SocialAgent
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_platform.config.mall import MallConfig, LocationRing
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType


# Default actions for mall simulation agents
MALL_AVAILABLE_ACTIONS = [
    ActionType.CREATE_POST,
    ActionType.LIKE_POST,
    ActionType.DISLIKE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.LIKE_COMMENT,
    ActionType.FOLLOW,
    ActionType.SEARCH_POSTS,
    ActionType.REFRESH,
    ActionType.DO_NOTHING,
    ActionType.VISIT_STORE,      # NEW
    ActionType.WRITE_REVIEW,     # NEW
]


def _generate_agent_profile_from_ring(
    ring: LocationRing, mall_config: MallConfig, agent_index: int
) -> dict[str, Any]:
    """Generate a single agent's profile from a location ring."""
    demo = ring.demographics
    age_range = demo.get("age_range", "20-50")
    age_min, age_max = map(int, age_range.split("-"))
    age = random.randint(age_min, age_max)
    gender = random.choice(["male", "female"])
    income = demo.get("income_level", "mid")
    person_type = demo.get("type", "general")
    mbti_types = [
        "INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP",
    ]
    mbti = random.choice(mbti_types)

    return {
        "age": age,
        "gender": gender,
        "income_level": income,
        "person_type": person_type,
        "mbti": mbti,
        "ring_radius_km": ring.radius_km,
        "visit_probability": mall_config.get_visit_probability(ring.radius_km),
        "mall_context": mall_config.to_agent_context(),
    }


async def generate_mall_agent_graph(
    mall_config: MallConfig,
    model: BaseModelBackend,
    agent_scale: float = 0.01,
    available_actions: list[ActionType] | None = None,
) -> AgentGraph:
    """Generate an AgentGraph from a MallConfig.

    Args:
        mall_config: The mall configuration to simulate.
        model: The LLM model for agents.
        agent_scale: Fraction of ring population to create as agents.
            0.01 = 1% of stated population (e.g., 20000 → 200 agents).
        available_actions: Override default mall actions.

    Returns:
        AgentGraph with all agents added.
    """
    if available_actions is None:
        available_actions = MALL_AVAILABLE_ACTIONS

    agent_graph = AgentGraph()
    agent_id = 0

    for ring in mall_config.location_rings:
        num_agents = max(1, int(ring.population * agent_scale))
        for i in range(num_agents):
            profile = _generate_agent_profile_from_ring(ring, mall_config, agent_id)

            user_info = UserInfo(
                user_name=f"user_{agent_id}",
                name=f"Consumer_{agent_id}",
                description=(
                    f"A {profile['age']}-year-old {profile['gender']} "
                    f"({profile['person_type']}, {profile['income_level']} income, "
                    f"MBTI: {profile['mbti']}) living {profile['ring_radius_km']}km "
                    f"from the mall."
                ),
                profile={
                    "other_info": {
                        "user_profile": profile["mall_context"],
                        "gender": profile["gender"],
                        "age": profile["age"],
                        "mbti": profile["mbti"],
                        "country": "China",
                        "ring_radius_km": profile["ring_radius_km"],
                        "visit_probability": profile["visit_probability"],
                        "income_level": profile["income_level"],
                    }
                },
                recsys_type="reddit",
            )

            agent = SocialAgent(
                agent_id=agent_id,
                user_info=user_info,
                agent_graph=agent_graph,
                model=model,
                available_actions=available_actions,
            )
            agent_graph.add_agent(agent)
            agent_id += 1

    return agent_graph
```

#### 3. `oasis/social_platform/mall_analytics.py` — Analytics engine

```python
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScenarioReport:
    scenario_name: str
    total_visits: int = 0
    total_posts: int = 0
    total_comments: int = 0
    total_likes: int = 0
    total_dislikes: int = 0
    store_popularity: list[dict[str, Any]] = field(default_factory=list)
    sentiment_score: float = 0.0  # -1.0 to 1.0
    sentiment_by_segment: dict[str, float] = field(default_factory=dict)
    viral_posts: list[dict[str, Any]] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    foot_traffic_by_timestep: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "summary": {
                "total_visits": self.total_visits,
                "total_posts": self.total_posts,
                "total_comments": self.total_comments,
                "total_likes": self.total_likes,
                "total_dislikes": self.total_dislikes,
                "sentiment_score": round(self.sentiment_score, 3),
            },
            "store_popularity": self.store_popularity,
            "sentiment_by_segment": {
                k: round(v, 3) for k, v in self.sentiment_by_segment.items()
            },
            "viral_posts": self.viral_posts,
            "risk_factors": self.risk_factors,
            "foot_traffic_by_timestep": self.foot_traffic_by_timestep,
        }


class MallAnalytics:
    """Reads simulation DB and produces investment analytics.

    DATA FLOW:
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  trace   │    │  visit   │    │   post   │
    │  table   │    │  table   │    │  table   │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         └───────┬───────┘───────┬───────┘
                 │               │
                 ▼               ▼
         ┌──────────────┐ ┌──────────────┐
         │ Engagement   │ │  Sentiment   │
         │ Metrics      │ │  Analysis    │
         └──────┬───────┘ └──────┬───────┘
                │                │
                └────────┬───────┘
                         ▼
                 ┌───────────────┐
                 │ ScenarioReport│
                 └───────────────┘
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def generate_report(self, scenario_name: str) -> ScenarioReport:
        report = ScenarioReport(scenario_name=scenario_name)

        # Visit metrics
        visits = self._query("SELECT * FROM visit")
        report.total_visits = len(visits)

        # Store popularity (visits per store)
        store_visits = self._query(
            "SELECT s.store_name, s.category, COUNT(v.visit_id) as visit_count "
            "FROM visit v JOIN store s ON v.store_id = s.store_id "
            "GROUP BY v.store_id ORDER BY visit_count DESC"
        )
        report.store_popularity = store_visits

        # Post and engagement metrics
        posts = self._query("SELECT * FROM post")
        report.total_posts = len(posts)
        report.total_likes = sum(p["num_likes"] for p in posts)
        report.total_dislikes = sum(p["num_dislikes"] for p in posts)

        # Comments
        comments = self._query("SELECT COUNT(*) as cnt FROM comment")
        report.total_comments = comments[0]["cnt"] if comments else 0

        # Sentiment: (likes - dislikes) / (likes + dislikes)
        total_engagement = report.total_likes + report.total_dislikes
        if total_engagement > 0:
            report.sentiment_score = (
                (report.total_likes - report.total_dislikes) / total_engagement
            )

        # Viral posts (top 5 by shares + comments)
        viral = self._query(
            "SELECT post_id, content, num_likes, num_dislikes, num_shares "
            "FROM post ORDER BY (num_likes + num_shares) DESC LIMIT 5"
        )
        report.viral_posts = viral

        # Risk factors
        if report.sentiment_score < 0:
            report.risk_factors.append(
                "Overall negative sentiment — review mall positioning"
            )
        negative_posts = self._query(
            "SELECT COUNT(*) as cnt FROM post WHERE num_dislikes > num_likes"
        )
        if negative_posts and negative_posts[0]["cnt"] > len(posts) * 0.3:
            report.risk_factors.append(
                f"{negative_posts[0]['cnt']}/{len(posts)} posts have more dislikes "
                "than likes — significant negative reception"
            )

        return report

    def compare_scenarios(
        self, reports: list[ScenarioReport]
    ) -> dict[str, Any]:
        """Compare multiple scenario reports side by side."""
        comparison = {"scenarios": []}
        for report in reports:
            comparison["scenarios"].append(report.to_dict())

        # Rank scenarios by composite score
        for scenario in comparison["scenarios"]:
            s = scenario["summary"]
            # Simple composite: visits * sentiment, higher = better
            sentiment = max(s["sentiment_score"], 0.01)
            scenario["composite_score"] = round(
                s["total_visits"] * sentiment, 2
            )

        comparison["scenarios"].sort(
            key=lambda x: x["composite_score"], reverse=True
        )
        comparison["recommendation"] = comparison["scenarios"][0]["scenario_name"]
        return comparison

    def export_json(self, report: ScenarioReport, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
```

#### 4. `oasis/social_platform/schema/store.sql` + `visit.sql` — New DB tables

**`store.sql`:**
```sql
CREATE TABLE store (
    store_id INTEGER PRIMARY KEY AUTOINCREMENT,
    store_name TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    floor INTEGER DEFAULT 1,
    brand_tier TEXT DEFAULT 'mid'
);
```

**`visit.sql`:**
```sql
CREATE TABLE visit (
    visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    store_id INTEGER,
    created_at DATETIME,
    FOREIGN KEY(user_id) REFERENCES user(user_id),
    FOREIGN KEY(store_id) REFERENCES store(store_id)
);
```

### Files to Modify (3)

#### 5. `oasis/social_platform/typing.py` — Add new ActionType values

```python
# Add to ActionType enum:
VISIT_STORE = "visit_store"
WRITE_REVIEW = "write_review"
```

#### 6. `oasis/social_agent/agent_action.py` — Add 2 new action methods

```python
# Add to SocialAction class:

async def visit_store(self, store_name: str) -> str:
    """Visit a store in the mall. Call this when you want to check out
    or shop at a specific store.

    Args:
        store_name (str): The name of the store to visit.

    Returns:
        str: The result of the visit.
    """
    message = (store_name,)
    result = await self.perform_action(message, "visit_store")
    return result

async def write_review(self, store_name: str, content: str) -> str:
    """Write a review about a store you visited.

    Args:
        store_name (str): The name of the store.
        content (str): Your review text.

    Returns:
        str: The result of posting the review.
    """
    message = (store_name, content)
    result = await self.perform_action(message, "write_review")
    return result

# Add to get_openai_function_list:
#   self.visit_store,
#   self.write_review,
```

#### 7. `oasis/social_platform/platform.py` — Add handlers for new actions

```python
# Add to Platform class:

async def visit_store(self, agent_id, visit_message):
    """Record a store visit."""
    (store_name,) = visit_message
    if self.recsys_type == RecsysType.REDDIT:
        current_time = self.sandbox_clock.time_transfer(
            datetime.now(), self.start_time)
    else:
        current_time = self.sandbox_clock.get_time_step()

    user_id = agent_id
    # Look up store
    store_check = "SELECT store_id FROM store WHERE store_name = ?"
    self.pl_utils._execute_db_command(store_check, (store_name,))
    store_row = self.db_cursor.fetchone()
    if not store_row:
        return {"success": False, "error": f"Store '{store_name}' not found."}

    store_id = store_row[0]
    visit_insert = (
        "INSERT INTO visit (user_id, store_id, created_at) VALUES (?, ?, ?)"
    )
    self.pl_utils._execute_db_command(
        visit_insert, (user_id, store_id, current_time), commit=True
    )

    action_info = {"store_name": store_name, "store_id": store_id}
    self.pl_utils._record_trace(
        user_id, ActionType.VISIT_STORE.value, action_info, current_time
    )
    return {"success": True, "store_name": store_name}


async def write_review(self, agent_id, review_message):
    """Write a review as a post tagged with the store."""
    store_name, content = review_message
    if self.recsys_type == RecsysType.REDDIT:
        current_time = self.sandbox_clock.time_transfer(
            datetime.now(), self.start_time)
    else:
        current_time = self.sandbox_clock.get_time_step()

    user_id = agent_id
    # Create a post with review content prefixed by store name
    review_content = f"[Review: {store_name}] {content}"
    post_insert = (
        "INSERT INTO post (user_id, content, created_at) VALUES (?, ?, ?)"
    )
    self.pl_utils._execute_db_command(
        post_insert, (user_id, review_content, current_time), commit=True
    )
    post_id = self.db_cursor.lastrowid

    action_info = {
        "store_name": store_name,
        "content": content,
        "post_id": post_id,
    }
    self.pl_utils._record_trace(
        user_id, ActionType.WRITE_REVIEW.value, action_info, current_time
    )
    return {"success": True, "post_id": post_id}


# Add store registration method (called during env.reset):
async def register_store(self, store_name: str, category: str,
                         floor: int, brand_tier: str):
    """Register a store in the mall."""
    try:
        insert_q = (
            "INSERT INTO store (store_name, category, floor, brand_tier) "
            "VALUES (?, ?, ?, ?)"
        )
        self.pl_utils._execute_db_command(
            insert_q, (store_name, category, floor, brand_tier), commit=True
        )
        return {"success": True, "store_name": store_name}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### New Example (1)

#### 8. `examples/mall_simulation.py` — End-to-end example

```python
import asyncio
import json
import os

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

import oasis
from oasis import LLMAction
from oasis.social_platform.config.mall import (
    BuildingConfig, CompetitorMall, LocationRing, MallConfig,
    TenantConfig, TransportConfig,
)
from oasis.social_agent.mall_agents_generator import generate_mall_agent_graph
from oasis.social_platform.mall_analytics import MallAnalytics


async def run_scenario(scenario_name: str, mall_config: MallConfig,
                       db_path: str, num_timesteps: int = 5):
    """Run a single mall simulation scenario."""
    if os.path.exists(db_path):
        os.remove(db_path)

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    agent_graph = await generate_mall_agent_graph(
        mall_config=mall_config,
        model=model,
        agent_scale=0.01,  # 1% of population → agents
    )

    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
    )

    await env.reset()

    # Register stores from mall config
    for tenant in mall_config.tenants:
        await env.platform.register_store(
            store_name=tenant.name,
            category=tenant.category,
            floor=tenant.floor,
            brand_tier=tenant.brand_tier,
        )

    # Run simulation timesteps
    for step in range(num_timesteps):
        print(f"[{scenario_name}] Timestep {step + 1}/{num_timesteps}")

        # All agents act via LLM (filtered by visit probability internally)
        actions = {
            agent: LLMAction()
            for _, agent in env.agent_graph.get_agents()
        }
        await env.step(actions)

    await env.close()

    # Generate analytics
    analytics = MallAnalytics(db_path)
    report = analytics.generate_report(scenario_name)
    analytics.export_json(report, f"./data/{scenario_name}_report.json")
    return report


async def main():
    # === SCENARIO A: Current mall (before renovation) ===
    config_a = MallConfig(
        building=BuildingConfig(
            name="Sunrise Mall",
            floors=3,
            building_type="dated",
            photo_spots=[],
        ),
        tenants=[
            TenantConfig("Local Supermarket", "Grocery", floor=1, brand_tier="budget"),
            TenantConfig("No-Name Fashion", "Fashion", floor=2, brand_tier="budget"),
            TenantConfig("Food Court", "F&B", floor=3, brand_tier="budget"),
        ],
        transport=TransportConfig(
            metro_distance_m=800,
            metro_lines=["Line 2"],
            parking_spots=100,
            parking_price_per_hour=5,
            walk_score=60,
        ),
        location_rings=[
            LocationRing(1, 20000, {"age_range": "18-35", "income_level": "mid",
                                     "type": "young_professional"}),
            LocationRing(3, 50000, {"age_range": "25-50", "income_level": "mid-high",
                                     "type": "family"}),
            LocationRing(5, 80000, {"age_range": "20-60", "income_level": "mixed",
                                     "type": "general"}),
        ],
        competitors=[
            CompetitorMall("Joy City", distance_km=2.5, positioning="premium"),
        ],
    )

    # === SCENARIO B: Renovated mall ===
    config_b = MallConfig(
        building=BuildingConfig(
            name="Sunrise Mall",
            floors=4,
            building_type="renovated_industrial",
            photo_spots=["rooftop_garden", "art_wall", "central_fountain"],
            year_renovated=2026,
        ),
        tenants=[
            TenantConfig("Starbucks", "F&B", floor=1, brand_tier="mid"),
            TenantConfig("Uniqlo", "Fashion", floor=2, brand_tier="mid"),
            TenantConfig("Haidilao", "F&B", floor=3, brand_tier="mid"),
            TenantConfig("CGV Cinema", "Entertainment", floor=4, brand_tier="mid"),
            TenantConfig("Pop Mart", "Retail", floor=1, brand_tier="mid"),
        ],
        transport=TransportConfig(
            metro_distance_m=200,  # New metro exit built
            metro_lines=["Line 2", "Line 7"],
            parking_spots=500,
            parking_price_per_hour=8,
            walk_score=85,
        ),
        location_rings=config_a.location_rings,  # Same population
        competitors=config_a.competitors,          # Same competitors
    )

    # Run both scenarios
    report_a = await run_scenario("before_renovation", config_a,
                                  "./data/mall_scenario_a.db")
    report_b = await run_scenario("after_renovation", config_b,
                                  "./data/mall_scenario_b.db")

    # Compare scenarios
    analytics = MallAnalytics("./data/mall_scenario_b.db")  # DB doesn't matter for comparison
    comparison = analytics.compare_scenarios([report_a, report_b])

    with open("./data/mall_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("SCENARIO COMPARISON")
    print("=" * 60)
    for s in comparison["scenarios"]:
        print(f"\n{s['scenario_name']}:")
        print(f"  Visits: {s['summary']['total_visits']}")
        print(f"  Posts: {s['summary']['total_posts']}")
        print(f"  Sentiment: {s['summary']['sentiment_score']}")
        print(f"  Composite: {s['composite_score']}")
    print(f"\nRecommendation: {comparison['recommendation']}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Implementation Order

```
Step 1: [COMPLETED] New DB tables (store.sql, visit.sql)
   │     + add to schema/ directory
   │     + registered in database.py create_db()
   │
Step 2: [COMPLETED] MallConfig dataclass (config/mall.py)
   │     + unit tests for to_agent_context() and get_visit_probability()
   │     + exported via config/__init__.py and oasis/__init__.py
   │
Step 3: [COMPLETED] New ActionTypes (typing.py)
   │     + visit_store, write_review in agent_action.py
   │     + handlers in platform.py
   │     + register_store() in platform.py
   │     + DB operation tests for visit_store, write_review
   │
Step 4: [COMPLETED] mall_agents_generator.py
   │     + generates agents from MallConfig location rings
   │     + exported via social_agent/__init__.py and oasis/__init__.py
   │
Step 5: [COMPLETED] MallAnalytics (mall_analytics.py)
   │     + test report generation with mock DB data (12 tests)
   │     + scenario comparison + JSON export
   │
Step 6: [COMPLETED] Example script (mall_simulation.py)
   │     + dual-scenario before/after renovation comparison
   │
Step 7: [COMPLETED] End-to-end verification
   │     + All imports verified
   │     + Platform handlers tested (visit_store, write_review, register_store)
   │     + Analytics pipeline tested (report generation, comparison, export)
   │     + 35 tests pass (23 new + 12 existing), 0 regressions
```

## Phase 2 Roadmap

### 2A. Richer Tenant Model

Current: 7 tenants with basic pricing (name, category, floor, rent, spend).
Target: 20-30 tenants with full P&L modeling.

New TenantConfig fields:
```python
@dataclass
class TenantConfig:
    # ... existing fields ...
    # P&L model
    gross_margin_pct: float = 0.0     # e.g. F&B ~60%, Fashion ~50%, Luxury ~70%
    staff_count: int = 0               # headcount
    avg_staff_cost: float = 0.0        # monthly per person
    inventory_turnover_days: int = 0   # how fast stock rotates
    min_viable_revenue: float = 0.0    # below this, tenant exits
    # Lease terms
    lease_start_date: str = ""
    rent_free_months: int = 0          # incentive period
    rent_escalation_pct: float = 0.0   # annual increase
    break_clause_month: int = 0        # when tenant can exit early
    deposit_months: int = 3
    # Brand characteristics
    brand_strength: float = 0.5        # 0-1, affects foot traffic draw
    target_customer_age: str = ""      # e.g. "18-30"
    online_presence_score: float = 0.0 # 0-1, affects social media buzz
```

Typical mall with 20+ tenants:
- Floor 1: 5-6 tenants (high footfall: coffee, bakery, pharmacy, convenience, fast fashion)
- Floor 2: 5-6 tenants (fashion, lifestyle, sports)
- Floor 3: 4-5 tenants (F&B restaurants, family entertainment)
- Floor 4: 3-4 tenants (cinema, gym, education, co-working)
- B1: 2-3 tenants (supermarket, parking services)

### 2B. Competitor Intelligence

Current: CompetitorMall has name + distance + positioning (3 fields).
Target: Full competitor analysis that affects agent behavior.

```python
@dataclass
class CompetitorMall:
    name: str
    distance_km: float
    positioning: str = "general"
    # NEW
    floors: int = 0
    total_area_sqm: float = 0.0
    anchor_tenants: list[str] = field(default_factory=list)
    monthly_foot_traffic: int = 0       # estimated from public data
    avg_rent_per_sqm: float = 0.0       # market benchmark
    occupancy_rate: float = 0.0         # % of space leased
    year_opened: int = 0
    recent_renovations: str = ""
    strengths: list[str] = field(default_factory=list)   # e.g. ["strong F&B", "cinema"]
    weaknesses: list[str] = field(default_factory=list)  # e.g. ["poor parking", "dated"]
```

How competitors affect simulation:
- Agent system prompt includes competitor context: "Joy City (2.5km away) has Zara, H&M,
  and an IMAX cinema. It recently renovated. How does Sunrise Mall compare?"
- Agents can express preference: "I'd rather go to Joy City for fashion"
- Monte Carlo uses competitor occupancy rate as market benchmark for survival estimates
- Rental prediction uses competitor avg_rent_per_sqm as pricing anchor

### 2C. Richer Agent (Customer) Profiles

Current: age, gender, income, MBTI, ring distance, budget.
Target: Full consumer psychographic profile.

```python
@dataclass
class ConsumerProfile:
    # Demographics (existing)
    age: int
    gender: str
    income_level: str
    mbti: str
    ring_radius_km: float
    monthly_shopping_budget: int
    # NEW: Lifestyle
    household_size: int = 1              # single, couple, family
    has_children: bool = False
    children_ages: list[int] = field(default_factory=list)
    car_owner: bool = False
    # NEW: Shopping behavior
    visit_frequency: str = "weekly"      # daily, weekly, monthly, rarely
    preferred_categories: list[str] = field(default_factory=list)
    price_sensitivity: float = 0.5       # 0=price insensitive, 1=very sensitive
    brand_loyalty: float = 0.5           # 0=switches easily, 1=very loyal
    social_media_activity: float = 0.5   # 0=lurker, 1=influencer
    # NEW: Commute
    primary_transport: str = "metro"     # metro, bus, car, walk, bike
    max_commute_minutes: int = 30
    # NEW: Competitor awareness
    visits_competitor_malls: list[str] = field(default_factory=list)
    competitor_satisfaction: dict[str, float] = field(default_factory=dict)
```

Impact on simulation:
- Agents with children specifically look for kids entertainment, family restaurants
- Car owners care about parking; metro users don't
- High price_sensitivity agents complain more about expensive stores
- High social_media_activity agents post more, driving viral potential
- Competitor-aware agents make direct comparisons in their posts

### 2D. Cap Rate Valuation Model

Cap rate is the key metric for commercial real estate valuation:
  Asset Value = Net Operating Income (NOI) / Cap Rate

```
VALUATION PIPELINE:
┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐
│ Gross Rental     │    │ Operating        │    │ Market Cap Rate   │
│ Income (from MC) │───▶│ Expenses         │───▶│ (from comps)      │
│ P10/P50/P90      │    │ - Management 5%  │    │ e.g. 5-8% for     │
│                  │    │ - Maintenance 3% │    │ neighborhood mall  │
│                  │    │ - Insurance 1%   │    │                   │
│                  │    │ - Property tax   │    │                   │
│                  │    │ - Vacancy reserve│    │                   │
└─────────────────┘    └────────┬─────────┘    └─────────┬─────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐      ┌───────────────┐
                       │ NOI = Gross -    │      │ Asset Value = │
                       │   Expenses       │─────▶│ NOI / CapRate │
                       └─────────────────┘      └───────────────┘
```

New module: `oasis/social_platform/valuation.py`

```python
@dataclass
class ValuationConfig:
    management_fee_pct: float = 0.05      # % of gross income
    maintenance_pct: float = 0.03
    insurance_pct: float = 0.01
    property_tax_pct: float = 0.012
    vacancy_reserve_pct: float = 0.05     # already modeled by MC, but buffer
    cap_rate: float = 0.065               # market cap rate for this area/type
    cap_rate_range: tuple[float, float] = (0.05, 0.08)  # sensitivity range

@dataclass
class ValuationReport:
    # From Monte Carlo
    gross_income_annual_p50: float
    gross_income_annual_p10: float
    gross_income_annual_p90: float
    # Expenses
    total_expenses_pct: float
    noi_annual_p50: float
    noi_annual_p10: float
    noi_annual_p90: float
    # Valuation
    asset_value_p50: float               # NOI_P50 / cap_rate
    asset_value_p10: float               # NOI_P10 / high_cap_rate (conservative)
    asset_value_p90: float               # NOI_P90 / low_cap_rate (optimistic)
    # Investment analysis
    acquisition_cost: float              # user input
    value_add_cost: float                # renovation/refit cost
    total_investment: float
    projected_irr: float                 # internal rate of return
    equity_multiple: float               # total return / equity invested
    payback_years: float

    # Scenario comparison
    # Run for baseline and each fix → which fix maximizes asset value?
```

Example output:
```
VALUATION — SUNRISE MALL
  Gross income (annual, P50):     ¥1,157,267
  Operating expenses (15%):       ¥  173,590
  NOI (P50):                      ¥  983,677
  Cap rate:                       6.5%
  ──────────────────────────────────────────
  Asset value (P50):              ¥15,133,492
  Asset value (P10, conservative):¥10,421,738
  Asset value (P90, optimistic):  ¥22,847,200

  INVESTMENT ANALYSIS
  Acquisition cost:               ¥12,000,000
  Renovation (Gucci→Zara fix):    ¥   500,000
  Total investment:               ¥12,500,000
  Post-fix asset value (P50):     ¥18,900,000
  Value created:                  ¥ 6,400,000
  Equity multiple:                1.51x
  IRR (3yr hold):                 14.8%
```

### 2E. Monte Carlo Optimization Engine — Find the Best Fix Automatically

Current approach: investor manually defines 3-4 fix scenarios, MC evaluates each.
Target: **MC automatically searches hundreds of tenant combinations and finds the
one that maximizes asset value.**

```
OPTIMIZATION PIPELINE:
┌─────────────────────────────────────────────────────────────────────┐
│  1. TENANT CANDIDATE POOL                                          │
│     A library of ~50 possible tenants (brand, category, tier,      │
│     rent, spend, area) that could fit in this mall.                │
│     Sourced from: market data, competitor analysis, broker lists.  │
│                                                                    │
│  2. SLOT DEFINITION                                                │
│     Which spaces are "fixable"? The investor marks slots:          │
│     - Slot A: Floor 2, 150m², currently Gucci (REPLACE)           │
│     - Slot B: Floor 1, 40m², currently Luckin (OPTIONAL)          │
│     - Other tenants: KEEP (not part of optimization)              │
│                                                                    │
│  3. COMBINATION GENERATOR                                          │
│     For each fixable slot, try every candidate that fits:          │
│     - Category constraints (Floor 1 = F&B/Retail only)            │
│     - Size constraints (candidate area ≤ slot area)               │
│     - No duplicate brands (can't have 2 Starbucks)                │
│     → Generates N valid combinations (typically 100-500)           │
│                                                                    │
│  4. MC EVALUATION (per combination)                                │
│     For each combination, run 1000 MC iterations with:             │
│     - Randomized agent demographics (age ±5yr, income ±20%)       │
│     - Randomized visit behavior (frequency ±30%)                  │
│     - Randomized price sensitivity (±25%)                         │
│     - Randomized competitor pull (±20%)                           │
│     → Produces: expected 3yr NOI (P10/P50/P90), tenant survival   │
│                                                                    │
│  5. RANK BY EXPECTED ASSET VALUE                                   │
│     Asset Value = NOI / Cap Rate                                   │
│     Rank all combinations by P50 asset value                      │
│     → Top 10 combinations with confidence intervals               │
│                                                                    │
│  6. VALIDATE TOP 3 WITH LLM AGENTS                                │
│     Run full OASIS simulation on top 3 combinations only          │
│     → Qualitative validation: do agents like this mix?            │
│     → Final recommendation with both signals                      │
└─────────────────────────────────────────────────────────────────────┘
```

Key design: MC models **agent behavior variation**, not just financial noise:

```python
# Per-iteration, each simulated "agent segment" has randomized behavior:
@dataclass
class AgentSegmentSample:
    # Demographics (from LocationRing, with noise)
    population: int              # ring.population × uniform(0.8, 1.2)
    avg_income_rank: float       # ring income ± noise
    price_sensitivity: float     # uniform(0.3, 0.9) — how much price matters

    # Visit behavior (randomized per iteration)
    mall_visit_frequency: float  # visits/month, from transport model ± 30%
    store_enter_rate: float      # base enter rate × uniform(0.7, 1.3)
    purchase_rate: float         # base purchase rate × uniform(0.8, 1.2)

    # Competitor diversion
    competitor_pull: float       # 0-1, what % of visits go to competitor instead
    competitor_overlap: list[str] # categories where competitor is stronger
```

This means the Monte Carlo captures real consumer uncertainty:
- "What if 30% fewer families visit because a new competitor opens?"
- "What if students are more price-sensitive than we assumed?"
- "What if the cinema draws less traffic than expected?"

Output to investor:
```
TOP 5 TENANT COMBINATIONS (ranked by expected 3yr asset value):

  #1: Replace Gucci → Nike, Keep Luckin
      3yr NOI P50: ¥2.8M  |  Asset Value: ¥43M  |  Refit: ¥400K
      P(beats baseline): 98.2%  |  Expected uplift: +¥6.2M

  #2: Replace Gucci → Zara, Replace Luckin → Manner Coffee
      3yr NOI P50: ¥2.6M  |  Asset Value: ¥40M  |  Refit: ¥600K
      P(beats baseline): 95.7%  |  Expected uplift: +¥4.8M

  #3: Replace Gucci → Adidas, Keep Luckin
      3yr NOI P50: ¥2.5M  |  Asset Value: ¥38M  |  Refit: ¥350K
      P(beats baseline): 94.1%  |  Expected uplift: +¥4.1M
  ...
```

### Implementation Order (Phase 2) — REVISED

Priority: ship the product differentiator first, accuracy later.

```
Step 2E: MC Optimization Engine  [2 weeks]  — THE core feature
   │     tenant_pool.py: ~30 real Chinese retail brands with real pricing
   │     optimizer.py: combination generator + MC evaluator + ranker
   │     Uses existing MallConfig + monte_carlo.py
   │     Output: "Top 5 tenant fixes ranked by expected asset value"
   │
Step 2D: Cap Rate Valuation      [1 week]  — speaks fund manager language
   │     valuation.py: NOI = gross - expenses, asset value = NOI / cap rate
   │     Plugs into optimizer ranking (rank by asset value, not raw rent)
   │     Adds IRR and equity multiple to final report
   │
Step 2A: Tenant Pool (simplified) [3 days] — real brands, not richer schema
   │     NOT adding 15 new dataclass fields
   │     Instead: JSON file with ~30 real Chinese retail brands
   │     (Starbucks, Luckin, Heytea, Uniqlo, Zara, UR, Nike, Haidilao, etc.)
   │     Each brand: name, category, tier, avg_spend, typical_rent, typical_area
   │     The optimizer draws candidates from this pool
   │
Step 2F: Tests + polish           [2 days] — verify end-to-end
```

Dependency: 2E uses existing monte_carlo.py. 2D plugs into 2E's ranking.
2A feeds 2E's candidate pool. Total: ~3 weeks.

```
  2A (tenant pool JSON, 3 days)
   │
   └──→ 2E (optimizer, 2 weeks) ──→ 2D (cap rate, 1 week) ──→ 2F (tests)
```

## Deferred to Phase 3

- 2B: Competitor intelligence — competitors already in agent context, enrich later
- 2C: Customer psychographics — current age/income/MBTI is sufficient for v2
- Richer TenantConfig fields (gross_margin, staff_count, lease terms) — only add when
  a real investor says "I need this field"
- Billing/pricing engine
- Web dashboard or API
- PDF/PPTX report generation
- Real geographic data integration (GIS, census API)
- Time-series simulation (monthly tenant churn over 36 months)
- Debt modeling (LTV, DSCR, mortgage payments)
- Multi-asset portfolio optimization
- Real-time market data feeds (rent comps, transaction data)
