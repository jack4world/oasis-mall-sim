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
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oasis.social_platform.config.mall import MallConfig


@dataclass
class TenantMixAnalysis:
    category_distribution: dict[str, int] = field(default_factory=dict)
    category_visit_share: dict[str, float] = field(default_factory=dict)
    category_mention_share: dict[str, float] = field(default_factory=dict)
    underserved_categories: list[str] = field(default_factory=list)
    overrepresented_categories: list[str] = field(default_factory=list)
    floor_traffic: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category_distribution": self.category_distribution,
            "category_visit_share": {
                k: round(v, 3) for k, v in self.category_visit_share.items()},
            "category_mention_share": {
                k: round(v, 3)
                for k, v in self.category_mention_share.items()},
            "underserved_categories": self.underserved_categories,
            "overrepresented_categories": self.overrepresented_categories,
            "floor_traffic": self.floor_traffic,
        }


@dataclass
class AffordabilityAnalysis:
    price_mentions: int = 0
    positive_price_mentions: int = 0
    negative_price_mentions: int = 0
    affordability_score: float = 0.0
    expensive_stores: list[str] = field(default_factory=list)
    affordable_stores: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "price_mentions": self.price_mentions,
            "positive_price_mentions": self.positive_price_mentions,
            "negative_price_mentions": self.negative_price_mentions,
            "affordability_score": round(self.affordability_score, 3),
            "stores_perceived_expensive": self.expensive_stores,
            "stores_perceived_affordable": self.affordable_stores,
        }


@dataclass
class RentalPrediction:
    estimated_daily_foot_traffic: int = 0
    estimated_monthly_foot_traffic: int = 0
    avg_sentiment: float = 0.0
    predicted_rent_per_sqm: float = 0.0
    rent_by_floor: dict[int, float] = field(default_factory=dict)
    rent_by_category: dict[str, float] = field(default_factory=dict)
    methodology: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_daily_foot_traffic": self.estimated_daily_foot_traffic,
            "estimated_monthly_foot_traffic":
                self.estimated_monthly_foot_traffic,
            "avg_sentiment": round(self.avg_sentiment, 3),
            "predicted_rent_per_sqm_per_month":
                round(self.predicted_rent_per_sqm, 2),
            "rent_by_floor": {
                str(k): round(v, 2) for k, v in self.rent_by_floor.items()},
            "rent_by_category": {
                k: round(v, 2) for k, v in self.rent_by_category.items()},
            "methodology": self.methodology,
        }


@dataclass
class ScenarioReport:
    scenario_name: str
    total_visits: int = 0
    total_posts: int = 0
    total_comments: int = 0
    total_likes: int = 0
    total_dislikes: int = 0
    store_popularity: list[dict[str, Any]] = field(default_factory=list)
    sentiment_score: float = 0.0
    sentiment_by_segment: dict[str, float] = field(default_factory=dict)
    viral_posts: list[dict[str, Any]] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    tenant_mix: TenantMixAnalysis | None = None
    affordability: AffordabilityAnalysis | None = None
    rental_prediction: RentalPrediction | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
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
                k: round(v, 3)
                for k, v in self.sentiment_by_segment.items()
            },
            "viral_posts": self.viral_posts,
            "risk_factors": self.risk_factors,
        }
        if self.tenant_mix is not None:
            result["tenant_mix_analysis"] = self.tenant_mix.to_dict()
        if self.affordability is not None:
            result["affordability_analysis"] = self.affordability.to_dict()
        if self.rental_prediction is not None:
            result["rental_prediction"] = self.rental_prediction.to_dict()
        return result


class MallAnalytics:
    r"""Reads simulation DB and produces investment analytics.

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

    def _query(self, sql: str,
               params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def generate_report(self, scenario_name: str) -> ScenarioReport:
        report = ScenarioReport(scenario_name=scenario_name)

        visits = self._query("SELECT * FROM visit")
        report.total_visits = len(visits)

        store_visits = self._query(
            "SELECT s.store_name, s.category, COUNT(v.visit_id) as "
            "visit_count FROM visit v JOIN store s ON v.store_id = "
            "s.store_id GROUP BY v.store_id ORDER BY visit_count DESC")
        report.store_popularity = store_visits

        posts = self._query("SELECT * FROM post")
        report.total_posts = len(posts)
        report.total_likes = sum(p["num_likes"] for p in posts)
        report.total_dislikes = sum(p["num_dislikes"] for p in posts)

        comments = self._query("SELECT COUNT(*) as cnt FROM comment")
        report.total_comments = comments[0]["cnt"] if comments else 0

        total_engagement = report.total_likes + report.total_dislikes
        if total_engagement > 0:
            report.sentiment_score = (
                (report.total_likes - report.total_dislikes) /
                total_engagement)

        viral = self._query(
            "SELECT post_id, content, num_likes, num_dislikes, num_shares "
            "FROM post ORDER BY (num_likes + num_shares) DESC LIMIT 5")
        report.viral_posts = viral

        if report.sentiment_score < 0:
            report.risk_factors.append(
                "Overall negative sentiment — review mall positioning")
        if report.total_posts > 0:
            negative_posts = self._query(
                "SELECT COUNT(*) as cnt FROM post "
                "WHERE num_dislikes > num_likes")
            neg_count = negative_posts[0]["cnt"] if negative_posts else 0
            if neg_count > report.total_posts * 0.3:
                report.risk_factors.append(
                    f"{neg_count}/{report.total_posts} posts have more "
                    "dislikes than likes — significant negative reception")

        return report

    def analyze_tenant_mix(
        self, mall_config: MallConfig
    ) -> TenantMixAnalysis:
        analysis = TenantMixAnalysis()

        for t in mall_config.tenants:
            analysis.category_distribution[t.category] = (
                analysis.category_distribution.get(t.category, 0) + 1)

        store_visits = self._query(
            "SELECT s.store_name, s.category, s.floor, "
            "COUNT(v.visit_id) as cnt "
            "FROM store s LEFT JOIN visit v ON s.store_id = v.store_id "
            "GROUP BY s.store_id")
        total_visits = sum(r["cnt"] for r in store_visits)
        if total_visits > 0:
            cat_visits: dict[str, int] = {}
            for r in store_visits:
                cat_visits[r["category"]] = (
                    cat_visits.get(r["category"], 0) + r["cnt"])
            for cat, cnt in cat_visits.items():
                analysis.category_visit_share[cat] = cnt / total_visits

        for r in store_visits:
            floor = r["floor"]
            analysis.floor_traffic[floor] = (
                analysis.floor_traffic.get(floor, 0) + r["cnt"])

        all_text = self._query(
            "SELECT content FROM post UNION ALL "
            "SELECT content FROM comment")
        store_names = {t.name: t.category for t in mall_config.tenants}
        cat_mentions: dict[str, int] = {}
        total_mentions = 0
        for row in all_text:
            content = (row["content"] or "").lower()
            for store_name, cat in store_names.items():
                if store_name.lower() in content:
                    cat_mentions[cat] = cat_mentions.get(cat, 0) + 1
                    total_mentions += 1
        if total_mentions > 0:
            for cat, cnt in cat_mentions.items():
                analysis.category_mention_share[cat] = cnt / total_mentions

        cat_count = analysis.category_distribution
        total_tenants = sum(cat_count.values()) if cat_count else 1
        for cat, share in analysis.category_mention_share.items():
            tenant_share = cat_count.get(cat, 0) / total_tenants
            if share > tenant_share * 1.5 and tenant_share < 0.3:
                analysis.underserved_categories.append(cat)
        for cat, count in cat_count.items():
            tenant_share = count / total_tenants
            mention_share = analysis.category_mention_share.get(cat, 0)
            if tenant_share > 0.3 and mention_share < tenant_share * 0.5:
                analysis.overrepresented_categories.append(cat)

        return analysis

    def analyze_affordability(
        self, mall_config: MallConfig
    ) -> AffordabilityAnalysis:
        analysis = AffordabilityAnalysis()

        price_keywords_pos = [
            "affordable", "cheap", "good price", "good value",
            "worth it", "reasonable", "bargain", "budget-friendly",
            "great deal", "fair price", "value for money",
        ]
        price_keywords_neg = [
            "expensive", "overpriced", "pricey", "too much",
            "can't afford", "costly", "not worth", "rip off",
            "too expensive", "out of budget", "unaffordable",
        ]

        all_text = self._query(
            "SELECT content FROM post UNION ALL "
            "SELECT content FROM comment")
        store_names = [t.name for t in mall_config.tenants]

        for row in all_text:
            content = (row["content"] or "").lower()
            has_price_ref = False
            is_positive = False
            is_negative = False

            for kw in price_keywords_pos:
                if kw in content:
                    has_price_ref = True
                    is_positive = True
            for kw in price_keywords_neg:
                if kw in content:
                    has_price_ref = True
                    is_negative = True

            if has_price_ref:
                analysis.price_mentions += 1
                if is_positive:
                    analysis.positive_price_mentions += 1
                if is_negative:
                    analysis.negative_price_mentions += 1

                for name in store_names:
                    if name.lower() in content:
                        if is_negative and name not in analysis.expensive_stores:
                            analysis.expensive_stores.append(name)
                        if is_positive and name not in analysis.affordable_stores:
                            analysis.affordable_stores.append(name)

        if analysis.price_mentions > 0:
            analysis.affordability_score = (
                (analysis.positive_price_mentions
                 - analysis.negative_price_mentions)
                / analysis.price_mentions)

        return analysis

    def predict_rental_price(
        self,
        mall_config: MallConfig,
        simulation_timesteps: int,
        base_rent_per_sqm: float = 150.0,
    ) -> RentalPrediction:
        """Predict rental price based on simulation results.

        RENTAL PREDICTION MODEL:
        ┌────────────────┐   ┌──────────────┐   ┌───────────────┐
        │  Foot Traffic   │   │  Sentiment   │   │  Store Visits  │
        │  (from agents   │   │  (likes vs   │   │  (per tenant)  │
        │   × population  │   │   dislikes)  │   │               │
        │   scale factor) │   │              │   │               │
        └───────┬────────┘   └──────┬───────┘   └───────┬───────┘
                │                    │                    │
                └────────────┬───────┘────────────┬──────┘
                             │                    │
                             ▼                    ▼
                    ┌─────────────────┐  ┌──────────────────┐
                    │ Traffic Multiplier│ │ Popularity Premium│
                    │ base × traffic   │ │ per-store adjust  │
                    └────────┬────────┘  └────────┬─────────┘
                             │                    │
                             └──────────┬─────────┘
                                        ▼
                               ┌─────────────────┐
                               │ Floor Discount   │
                               │ F1=1.0, F2=0.85 │
                               │ F3=0.7, F4+=0.6 │
                               └────────┬────────┘
                                        ▼
                               ┌─────────────────┐
                               │ ¥/m²/month      │
                               └─────────────────┘

        Args:
            mall_config: The mall configuration.
            simulation_timesteps: Number of timesteps run.
            base_rent_per_sqm: Base rent for the area (¥/m²/month).
        """
        pred = RentalPrediction()

        total_pop = sum(r.population for r in mall_config.location_rings)
        visits = self._query("SELECT COUNT(*) as cnt FROM visit")
        sim_visits = visits[0]["cnt"] if visits else 0
        agents = self._query("SELECT COUNT(*) as cnt FROM user")
        num_agents = agents[0]["cnt"] if agents else 1

        scale_factor = total_pop / max(num_agents, 1)
        if simulation_timesteps > 0:
            daily_visits = (
                sim_visits * scale_factor / simulation_timesteps)
        else:
            daily_visits = 0
        pred.estimated_daily_foot_traffic = int(daily_visits)
        pred.estimated_monthly_foot_traffic = int(daily_visits * 30)

        posts = self._query(
            "SELECT SUM(num_likes) as likes, SUM(num_dislikes) as dislikes "
            "FROM post")
        total_likes = posts[0]["likes"] or 0 if posts else 0
        total_dislikes = posts[0]["dislikes"] or 0 if posts else 0
        total_eng = total_likes + total_dislikes
        pred.avg_sentiment = (
            (total_likes - total_dislikes) / total_eng
            if total_eng > 0 else 0.5)

        traffic_multiplier = min(daily_visits / 5000, 2.0)
        sentiment_multiplier = 0.7 + (pred.avg_sentiment * 0.6)

        pred.predicted_rent_per_sqm = (
            base_rent_per_sqm * max(traffic_multiplier, 0.3)
            * sentiment_multiplier)

        floor_discount = {1: 1.0, 2: 0.85, 3: 0.70}
        for floor_num in range(1, mall_config.building.floors + 1):
            discount = floor_discount.get(floor_num, 0.60)
            pred.rent_by_floor[floor_num] = (
                pred.predicted_rent_per_sqm * discount)

        store_visits = self._query(
            "SELECT s.category, COUNT(v.visit_id) as cnt "
            "FROM store s LEFT JOIN visit v ON s.store_id = v.store_id "
            "GROUP BY s.category")
        total_sv = sum(r["cnt"] for r in store_visits)
        cat_premium = {"Entertainment": 0.8, "F&B": 1.1, "Fashion": 1.0,
                       "Service": 0.9, "Retail": 0.95, "Grocery": 0.75}
        for row in store_visits:
            cat = row["category"]
            base = pred.predicted_rent_per_sqm * cat_premium.get(cat, 1.0)
            if total_sv > 0 and row["cnt"] > 0:
                popularity_boost = 1 + (row["cnt"] / total_sv) * 0.3
                base *= popularity_boost
            pred.rent_by_category[cat] = base

        pred.methodology = (
            f"Based on {num_agents} simulated agents representing "
            f"{total_pop:,} residents over {simulation_timesteps} timesteps. "
            f"Traffic multiplier: {traffic_multiplier:.2f}, "
            f"sentiment multiplier: {sentiment_multiplier:.2f}. "
            f"Base rent: ¥{base_rent_per_sqm}/m²/month for this area.")

        return pred

    def generate_full_report(
        self,
        scenario_name: str,
        mall_config: MallConfig,
        simulation_timesteps: int,
        base_rent_per_sqm: float = 150.0,
    ) -> ScenarioReport:
        report = self.generate_report(scenario_name)
        report.tenant_mix = self.analyze_tenant_mix(mall_config)
        report.affordability = self.analyze_affordability(mall_config)
        report.rental_prediction = self.predict_rental_price(
            mall_config, simulation_timesteps, base_rent_per_sqm)

        if report.tenant_mix.underserved_categories:
            cats = ", ".join(report.tenant_mix.underserved_categories)
            report.risk_factors.append(
                f"Underserved categories (high demand, few tenants): {cats}")
        if report.tenant_mix.overrepresented_categories:
            cats = ", ".join(report.tenant_mix.overrepresented_categories)
            report.risk_factors.append(
                f"Overrepresented categories (many tenants, low interest): "
                f"{cats}")
        if (report.affordability is not None
                and report.affordability.affordability_score < -0.3):
            report.risk_factors.append(
                "Significant affordability concerns — consumers perceive "
                "prices as too high for this demographic")
        if report.affordability is not None:
            for store in report.affordability.expensive_stores:
                report.risk_factors.append(
                    f'"{store}" perceived as expensive by consumers')

        return report

    def compare_scenarios(
        self, reports: list[ScenarioReport]
    ) -> dict[str, Any]:
        comparison: dict[str, Any] = {"scenarios": []}
        for report in reports:
            scenario = report.to_dict()
            s = scenario["summary"]
            sentiment = max(s["sentiment_score"], 0.01)
            scenario["composite_score"] = round(
                s["total_visits"] * sentiment, 2)
            comparison["scenarios"].append(scenario)

        comparison["scenarios"].sort(key=lambda x: x["composite_score"],
                                     reverse=True)
        if comparison["scenarios"]:
            comparison["recommendation"] = (
                comparison["scenarios"][0]["scenario_name"])
        return comparison

    def export_json(self, report: ScenarioReport, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
