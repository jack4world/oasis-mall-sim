# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import asyncio
import json
import os

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

import oasis
from oasis import (BuildingConfig, CompetitorMall, LLMAction, LocationRing,
                   MallConfig, TenantConfig, TransportConfig)
from oasis.social_agent.mall_agents_generator import generate_mall_agent_graph
from oasis.social_platform.mall_analytics import MallAnalytics


async def run_scenario(scenario_name: str, mall_config: MallConfig,
                       db_path: str, num_timesteps: int = 5):
    if os.path.exists(db_path):
        os.remove(db_path)

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    agent_graph = await generate_mall_agent_graph(
        mall_config=mall_config,
        model=model,
        agent_scale=0.01,
    )

    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
    )

    await env.reset()

    for tenant in mall_config.tenants:
        await env.platform.register_store(
            store_name=tenant.name,
            category=tenant.category,
            floor=tenant.floor,
            brand_tier=tenant.brand_tier,
        )

    for step in range(num_timesteps):
        print(f"[{scenario_name}] Timestep {step + 1}/{num_timesteps}")
        actions = {
            agent: LLMAction()
            for _, agent in env.agent_graph.get_agents()
        }
        await env.step(actions)

    await env.close()

    analytics = MallAnalytics(db_path)
    report = analytics.generate_report(scenario_name)
    report_path = f"./data/{scenario_name}_report.json"
    analytics.export_json(report, report_path)
    print(f"[{scenario_name}] Report saved to {report_path}")
    return report


async def main():
    os.makedirs("./data", exist_ok=True)

    config_a = MallConfig(
        building=BuildingConfig(
            name="Sunrise Mall",
            floors=3,
            building_type="dated",
            photo_spots=[],
        ),
        tenants=[
            TenantConfig("Local Supermarket", "Grocery", floor=1,
                         brand_tier="budget"),
            TenantConfig("No-Name Fashion", "Fashion", floor=2,
                         brand_tier="budget"),
            TenantConfig("Food Court", "F&B", floor=3,
                         brand_tier="budget"),
        ],
        transport=TransportConfig(
            metro_distance_m=800,
            metro_lines=["Line 2"],
            parking_spots=100,
            parking_price_per_hour=5,
            walk_score=60,
        ),
        location_rings=[
            LocationRing(1, 20000, {
                "age_range": "18-35",
                "income_level": "mid",
                "type": "young_professional",
            }),
            LocationRing(3, 50000, {
                "age_range": "25-50",
                "income_level": "mid-high",
                "type": "family",
            }),
            LocationRing(5, 80000, {
                "age_range": "20-60",
                "income_level": "mixed",
                "type": "general",
            }),
        ],
        competitors=[
            CompetitorMall("Joy City", distance_km=2.5,
                           positioning="premium"),
        ],
    )

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
            TenantConfig("CGV Cinema", "Entertainment", floor=4,
                         brand_tier="mid"),
            TenantConfig("Pop Mart", "Retail", floor=1, brand_tier="mid"),
        ],
        transport=TransportConfig(
            metro_distance_m=200,
            metro_lines=["Line 2", "Line 7"],
            parking_spots=500,
            parking_price_per_hour=8,
            walk_score=85,
        ),
        location_rings=config_a.location_rings,
        competitors=config_a.competitors,
    )

    report_a = await run_scenario("before_renovation", config_a,
                                  "./data/mall_scenario_a.db",
                                  num_timesteps=3)
    report_b = await run_scenario("after_renovation", config_b,
                                  "./data/mall_scenario_b.db",
                                  num_timesteps=3)

    analytics = MallAnalytics("./data/mall_scenario_b.db")
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
    if "recommendation" in comparison:
        print(f"\nRecommendation: {comparison['recommendation']}")


if __name__ == "__main__":
    asyncio.run(main())
