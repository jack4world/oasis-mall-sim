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
"""Mall Investment Analysis: Fix Probability for Asset Value Improvement.

Runs multiple tenant-mix scenarios against the same population to quantify
the probability and magnitude of each fix improving rental income.

INVESTOR OUTPUT:
  Fix 1: Replace Gucci → Zara     | Rent delta: +¥18/m² | Confidence: HIGH
  Fix 2: Remove duplicate coffee  | Rent delta: +¥5/m²  | Confidence: MED
  Fix 3: Add budget fashion       | Rent delta: +¥12/m² | Confidence: HIGH
"""
import asyncio
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

import oasis
from oasis import (ActionType, BuildingConfig, CompetitorMall, LLMAction,
                   LocationRing, MallConfig, ManualAction, TenantConfig,
                   TransportConfig)
from oasis.social_agent.mall_agents_generator import generate_mall_agent_graph
from oasis.social_platform.mall_analytics import MallAnalytics

NUM_TIMESTEPS = 3
AGENT_SCALE = 0.01
BASE_RENT_PER_SQM = 200.0


@dataclass
class FixScenario:
    name: str
    description: str
    config: MallConfig
    fix_cost_estimate: float = 0.0


@dataclass
class FixResult:
    scenario_name: str
    description: str
    fix_cost: float
    total_leasable_area: float
    monthly_rental_income: float
    avg_rent_per_sqm: float
    sentiment_score: float
    total_engagement: int
    total_visits: int
    total_posts: int
    total_comments: int
    rent_by_floor: dict[int, float] = field(default_factory=dict)
    risk_factors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "description": self.description,
            "fix_cost": self.fix_cost,
            "total_leasable_area_sqm": self.total_leasable_area,
            "predicted_monthly_rental_income": round(
                self.monthly_rental_income, 0),
            "predicted_avg_rent_per_sqm": round(self.avg_rent_per_sqm, 2),
            "sentiment_score": round(self.sentiment_score, 3),
            "total_engagement": self.total_engagement,
            "total_visits": self.total_visits,
            "total_posts": self.total_posts,
            "total_comments": self.total_comments,
            "rent_by_floor": {
                str(k): round(v, 2) for k, v in self.rent_by_floor.items()},
            "risk_factors": self.risk_factors,
        }


async def run_scenario(
    scenario: FixScenario, model: Any, db_dir: str,
) -> FixResult:
    db_name = scenario.name.replace(" ", "_").lower()
    db_path = os.path.join(db_dir, f"{db_name}.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    agent_graph = await generate_mall_agent_graph(
        mall_config=scenario.config, model=model, agent_scale=AGENT_SCALE)

    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
    )
    await env.reset()

    for t in scenario.config.tenants:
        await env.platform.register_store(
            store_name=t.name, category=t.category, floor=t.floor,
            brand_tier=t.brand_tier, avg_spend_per_visit=t.avg_spend_per_visit,
            monthly_rent=t.monthly_rent, area_sqm=t.area_sqm)

    # Seed: neutral prompt that lets agents form their own opinions
    agent_0 = env.agent_graph.get_agent(0)
    tenant_list = ", ".join(t.name for t in scenario.config.tenants)
    seed = {
        agent_0: ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={
                "content": (
                    f"I visited {scenario.config.building.name} today. "
                    f"The stores are: {tenant_list}. "
                    f"What do you all think of the tenant mix and pricing? "
                    f"Would you shop here regularly? What's good value "
                    f"and what's too expensive for this area?"
                ),
            },
        ),
    }
    await env.step(seed)

    for step in range(NUM_TIMESTEPS):
        print(f"  [{scenario.name}] step {step + 1}/{NUM_TIMESTEPS}")
        actions = {
            agent: LLMAction()
            for _, agent in env.agent_graph.get_agents()
        }
        await env.step(actions)

    await env.close()

    analytics = MallAnalytics(db_path)
    report = analytics.generate_full_report(
        scenario.name, scenario.config, NUM_TIMESTEPS, BASE_RENT_PER_SQM)

    total_area = sum(t.area_sqm for t in scenario.config.tenants)
    pred = report.rental_prediction
    if pred is not None:
        avg_rent = pred.predicted_rent_per_sqm
        rent_floors = pred.rent_by_floor
    else:
        avg_rent = 0.0
        rent_floors = {}

    # Compute weighted rental income from floor-level rents and tenant areas
    monthly_income = 0.0
    for t in scenario.config.tenants:
        floor_rent = rent_floors.get(t.floor, avg_rent)
        monthly_income += floor_rent * t.area_sqm

    return FixResult(
        scenario_name=scenario.name,
        description=scenario.description,
        fix_cost=scenario.fix_cost_estimate,
        total_leasable_area=total_area,
        monthly_rental_income=monthly_income,
        avg_rent_per_sqm=avg_rent,
        sentiment_score=report.sentiment_score,
        total_engagement=report.total_likes + report.total_dislikes,
        total_visits=report.total_visits,
        total_posts=report.total_posts,
        total_comments=report.total_comments,
        rent_by_floor=rent_floors,
        risk_factors=report.risk_factors,
    )


def build_scenarios() -> list[FixScenario]:
    # Shared location & transport (same for all scenarios)
    location_rings = [
        LocationRing(1, 500, {"age_range": "20-30", "income_level": "mid",
                               "type": "young_professional"}),
        LocationRing(3, 800, {"age_range": "30-50", "income_level": "mid-high",
                               "type": "family"}),
        LocationRing(5, 500, {"age_range": "18-25", "income_level": "low",
                               "type": "student"}),
    ]
    transport = TransportConfig(
        metro_distance_m=200, metro_lines=["Line 2", "Line 7"],
        bus_lines=4, parking_spots=500,
        parking_price_per_hour=8, walk_score=85)
    competitors = [
        CompetitorMall("Joy City", 2.5, "premium"),
        CompetitorMall("Wanda Plaza", 4.0, "general"),
    ]
    building = BuildingConfig(
        name="Sunrise Mall", floors=4,
        building_type="renovated_industrial",
        photo_spots=["rooftop_garden", "art_wall", "central_fountain"],
        year_renovated=2026)

    # === BASELINE: Current tenant mix (includes Gucci, dual coffee) ===
    baseline_tenants = [
        TenantConfig("Starbucks", "F&B", 1, "mid", 45, 350, 80),
        TenantConfig("Luckin Coffee", "F&B", 1, "budget", 20, 200, 40),
        TenantConfig("Miniso", "Retail", 1, "budget", 30, 150, 60),
        TenantConfig("Uniqlo", "Fashion", 2, "mid", 200, 280, 400),
        TenantConfig("Gucci", "Fashion", 2, "luxury", 5000, 500, 150),
        TenantConfig("Haidilao", "F&B", 3, "mid", 120, 300, 250),
        TenantConfig("CGV Cinema", "Entertainment", 4, "mid", 80, 180, 600),
    ]

    # === FIX 1: Replace Gucci with Zara (mid-tier fashion) ===
    fix1_tenants = deepcopy(baseline_tenants)
    fix1_tenants[4] = TenantConfig(
        "Zara", "Fashion", 2, "mid", 300, 320, 200)

    # === FIX 2: Replace Gucci with UR (budget fashion) + add Naifu bakery ===
    fix2_tenants = [
        TenantConfig("Starbucks", "F&B", 1, "mid", 45, 350, 80),
        TenantConfig("Miniso", "Retail", 1, "budget", 30, 150, 60),
        TenantConfig("Naifu Bakery", "F&B", 1, "budget", 25, 180, 40),
        TenantConfig("Uniqlo", "Fashion", 2, "mid", 200, 280, 400),
        TenantConfig("UR", "Fashion", 2, "budget", 150, 250, 150),
        TenantConfig("Haidilao", "F&B", 3, "mid", 120, 300, 250),
        TenantConfig("CGV Cinema", "Entertainment", 4, "mid", 80, 180, 600),
    ]

    # === FIX 3: Premium repositioning — keep Gucci, add more luxury ===
    fix3_tenants = [
        TenantConfig("Starbucks Reserve", "F&B", 1, "premium", 65, 400, 100),
        TenantConfig("Miniso", "Retail", 1, "budget", 30, 150, 60),
        TenantConfig("Uniqlo", "Fashion", 2, "mid", 200, 280, 400),
        TenantConfig("Gucci", "Fashion", 2, "luxury", 5000, 500, 150),
        TenantConfig("Haidilao", "F&B", 3, "mid", 120, 300, 250),
        TenantConfig("CGV Cinema", "Entertainment", 4, "mid", 80, 180, 600),
    ]

    scenarios = [
        FixScenario(
            "Baseline", "Current tenant mix with Gucci + dual coffee",
            MallConfig(building, baseline_tenants, transport,
                       location_rings, competitors),
            fix_cost_estimate=0),
        FixScenario(
            "Fix1 Gucci to Zara",
            "Replace Gucci (luxury) with Zara (mid-tier fashion)",
            MallConfig(building, fix1_tenants, transport,
                       location_rings, competitors),
            fix_cost_estimate=500000),
        FixScenario(
            "Fix2 Budget Friendly",
            "Remove Gucci + Luckin, add UR (budget fashion) + Naifu Bakery",
            MallConfig(building, fix2_tenants, transport,
                       location_rings, competitors),
            fix_cost_estimate=800000),
        FixScenario(
            "Fix3 Premium Push",
            "Keep Gucci, upgrade Starbucks to Reserve, remove Luckin",
            MallConfig(building, fix3_tenants, transport,
                       location_rings, competitors),
            fix_cost_estimate=300000),
    ]
    return scenarios


def print_investment_report(
    results: list[FixResult],
) -> dict[str, Any]:
    baseline = results[0]
    fixes = results[1:]

    report: dict[str, Any] = {
        "title": "Mall Investment Fix Analysis",
        "baseline": baseline.to_dict(),
        "fixes": [],
        "recommendation": "",
    }

    print()
    print("=" * 74)
    print("  MALL INVESTMENT FIX ANALYSIS — SUNRISE MALL")
    print("=" * 74)
    print()
    print(f"  Population: 1,800 residents | Agents: 18 | Timesteps: "
          f"{NUM_TIMESTEPS}")
    print(f"  Base rent benchmark: ¥{BASE_RENT_PER_SQM}/m²/month")
    print()

    # Baseline
    print("-" * 74)
    print(f"  BASELINE: {baseline.description}")
    print(f"    Avg rent: ¥{baseline.avg_rent_per_sqm:.0f}/m²/month")
    print(f"    Monthly income: ¥{baseline.monthly_rental_income:,.0f}")
    print(f"    Annual income:  ¥{baseline.monthly_rental_income * 12:,.0f}")
    print(f"    Sentiment: {baseline.sentiment_score:.2f}")
    print(f"    Engagement: {baseline.total_engagement} "
          f"(posts: {baseline.total_posts}, "
          f"comments: {baseline.total_comments})")
    if baseline.risk_factors:
        print(f"    Risks: {len(baseline.risk_factors)} factors")
    print()

    best_fix = None
    best_roi = -999.0

    for fix in fixes:
        rent_delta = fix.avg_rent_per_sqm - baseline.avg_rent_per_sqm
        income_delta = fix.monthly_rental_income - baseline.monthly_rental_income
        annual_delta = income_delta * 12
        sentiment_delta = fix.sentiment_score - baseline.sentiment_score
        engagement_delta = fix.total_engagement - baseline.total_engagement

        if fix.fix_cost > 0:
            payback_months = (
                fix.fix_cost / income_delta if income_delta > 0 else float(
                    "inf"))
            roi_annual = (annual_delta / fix.fix_cost * 100
                          if fix.fix_cost > 0 else 0)
        else:
            payback_months = 0
            roi_annual = 0

        # Confidence based on engagement + sentiment direction
        comment_delta = fix.total_comments - baseline.total_comments
        if (rent_delta > 0 and sentiment_delta >= 0
                and comment_delta >= 0):
            confidence = "HIGH"
        elif rent_delta > 0:
            confidence = "MEDIUM"
        elif rent_delta == 0:
            confidence = "LOW"
        else:
            confidence = "NEGATIVE"

        fix_data = {
            **fix.to_dict(),
            "delta_vs_baseline": {
                "rent_per_sqm_delta": round(rent_delta, 2),
                "monthly_income_delta": round(income_delta, 0),
                "annual_income_delta": round(annual_delta, 0),
                "sentiment_delta": round(sentiment_delta, 3),
                "engagement_delta": engagement_delta,
                "payback_months": (round(payback_months, 1)
                                   if payback_months != float("inf")
                                   else "never"),
                "annual_roi_pct": round(roi_annual, 1),
                "confidence": confidence,
            },
        }
        report["fixes"].append(fix_data)

        if roi_annual > best_roi and confidence in ("HIGH", "MEDIUM"):
            best_roi = roi_annual
            best_fix = fix_data

        print("-" * 74)
        print(f"  {fix.scenario_name}: {fix.description}")
        print(f"    Fix cost: ¥{fix.fix_cost:,.0f}")
        print(f"    Avg rent: ¥{fix.avg_rent_per_sqm:.0f}/m²/month "
              f"({'+' if rent_delta >= 0 else ''}{rent_delta:.0f})")
        print(f"    Monthly income: ¥{fix.monthly_rental_income:,.0f} "
              f"({'+' if income_delta >= 0 else ''}¥{income_delta:,.0f})")
        print(f"    Annual income:  ¥{fix.monthly_rental_income * 12:,.0f} "
              f"({'+' if annual_delta >= 0 else ''}¥{annual_delta:,.0f})")
        print(f"    Sentiment: {fix.sentiment_score:.2f} "
              f"({'+' if sentiment_delta >= 0 else ''}{sentiment_delta:.2f})")
        if payback_months != float("inf") and income_delta > 0:
            print(f"    Payback: {payback_months:.1f} months")
            print(f"    Annual ROI: {roi_annual:.1f}%")
        elif income_delta <= 0:
            print(f"    Payback: NEVER (negative return)")
        print(f"    Confidence: {confidence}")
        if fix.risk_factors:
            for r in fix.risk_factors[:3]:
                print(f"    ⚠ {r}")
        print()

    if best_fix:
        report["recommendation"] = best_fix["scenario"]
        print("=" * 74)
        print(f"  RECOMMENDATION: {best_fix['scenario']}")
        d = best_fix["delta_vs_baseline"]
        print(f"    Expected annual income uplift: "
              f"¥{d['annual_income_delta']:,.0f}")
        print(f"    ROI: {d['annual_roi_pct']}% annually")
        pb = d["payback_months"]
        if isinstance(pb, (int, float)):
            print(f"    Payback period: {pb:.1f} months")
        print(f"    Confidence: {d['confidence']}")
        print("=" * 74)

    return report


async def main():
    os.makedirs("./data", exist_ok=True)

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    scenarios = build_scenarios()
    results = []

    for scenario in scenarios:
        print(f"\nRunning: {scenario.name}...")
        result = await run_scenario(scenario, model, "./data")
        results.append(result)

    report = print_investment_report(results)

    with open("./data/investment_analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report: ./data/investment_analysis.json")


if __name__ == "__main__":
    asyncio.run(main())
