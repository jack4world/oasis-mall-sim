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
"""Monte Carlo simulation for tenant viability and rental income prediction.

Runs N iterations with randomized parameters to produce probability
distributions instead of single-point estimates.

RANDOMIZED PARAMETERS PER ITERATION:
  - Population per ring: ±20% uniform noise
  - Visit probability: ±15% noise
  - Enter rate: ±25% noise (foot traffic is volatile)
  - Purchase rate: ±20% noise
  - Avg spend per visit: ±30% noise (seasonal, promotional variation)
  - Operating cost multiplier: ±15% noise
  - Vacancy duration: ±50% noise (market-dependent)
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Any

from oasis.social_platform.config.mall import MallConfig, TenantConfig

INCOME_RANK = {
    "low": 1, "budget": 1, "mid": 2, "mixed": 2,
    "mid-high": 3, "high": 4,
}
TIER_TARGET_INCOME = {
    "budget": "low", "mid": "mid", "premium": "mid-high", "luxury": "high",
}


@dataclass
class TenantMCResult:
    tenant: str
    brand_tier: str
    monthly_rent: float
    survival_rates: list[float] = field(default_factory=list)
    monthly_revenues: list[float] = field(default_factory=list)
    profit_ratios: list[float] = field(default_factory=list)
    expected_3yr_rents: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        def pct(data: list[float], p: float) -> float:
            s = sorted(data)
            idx = int(len(s) * p)
            return s[min(idx, len(s) - 1)]

        survive_pcts = [1 if r >= 1.0 else 0 for r in self.profit_ratios]
        return {
            "tenant": self.tenant,
            "brand_tier": self.brand_tier,
            "monthly_rent": self.monthly_rent,
            "revenue_p10": round(pct(self.monthly_revenues, 0.10), 0),
            "revenue_p50": round(pct(self.monthly_revenues, 0.50), 0),
            "revenue_p90": round(pct(self.monthly_revenues, 0.90), 0),
            "profit_ratio_p10": round(pct(self.profit_ratios, 0.10), 2),
            "profit_ratio_p50": round(pct(self.profit_ratios, 0.50), 2),
            "profit_ratio_p90": round(pct(self.profit_ratios, 0.90), 2),
            "prob_profitable": round(
                sum(1 for r in self.profit_ratios if r >= 1.0)
                / len(self.profit_ratios) * 100, 1),
            "prob_survive_3yr": round(
                statistics.mean(self.survival_rates) * 100, 1),
            "expected_3yr_rent_p10": round(
                pct(self.expected_3yr_rents, 0.10), 0),
            "expected_3yr_rent_p50": round(
                pct(self.expected_3yr_rents, 0.50), 0),
            "expected_3yr_rent_p90": round(
                pct(self.expected_3yr_rents, 0.90), 0),
        }


@dataclass
class MallMCResult:
    iterations: int
    tenant_results: list[TenantMCResult] = field(default_factory=list)
    total_3yr_rents: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        def pct(data: list[float], p: float) -> float:
            s = sorted(data)
            idx = int(len(s) * p)
            return s[min(idx, len(s) - 1)]

        return {
            "iterations": self.iterations,
            "total_3yr_rent_p10": round(
                pct(self.total_3yr_rents, 0.10), 0),
            "total_3yr_rent_p50": round(
                pct(self.total_3yr_rents, 0.50), 0),
            "total_3yr_rent_p90": round(
                pct(self.total_3yr_rents, 0.90), 0),
            "total_3yr_rent_mean": round(
                statistics.mean(self.total_3yr_rents), 0),
            "total_3yr_rent_std": round(
                statistics.stdev(self.total_3yr_rents)
                if len(self.total_3yr_rents) > 1 else 0, 0),
            "tenants": [t.summary() for t in self.tenant_results],
        }


def _noise(base: float, pct: float) -> float:
    return base * (1 + random.uniform(-pct, pct))


def _run_single_iteration(
    config: MallConfig,
    base_rent_share: dict[str, float],
    enter_rate_by_gap: dict[int, float],
    purchase_rate_by_tier: dict[str, float],
) -> list[dict[str, float]]:
    tier_rank_map = {
        t: INCOME_RANK.get(TIER_TARGET_INCOME.get(t, "mid"), 2)
        for t in ["budget", "mid", "premium", "luxury"]
    }

    results = []
    for tenant in config.tenants:
        monthly_rent_total = tenant.monthly_rent * tenant.area_sqm
        tier_rank = tier_rank_map.get(tenant.brand_tier, 2)
        purchase_rate = _noise(
            purchase_rate_by_tier.get(tenant.brand_tier, 0.30), 0.20)

        total_paying = 0.0
        for ring in config.location_rings:
            ring_pop = _noise(ring.population, 0.20)
            ring_income = ring.demographics.get("income_level", "mid")
            area_rank = INCOME_RANK.get(ring_income, 2)
            gap = max(0, tier_rank - area_rank)

            base_enter = enter_rate_by_gap.get(gap, 0.005)
            enter_rate = _noise(base_enter, 0.25)
            visit_prob = _noise(
                config.get_visit_probability(ring.radius_km), 0.15)

            monthly_mall_visitors = ring_pop * visit_prob * 30
            store_traffic = monthly_mall_visitors * enter_rate
            paying = store_traffic * purchase_rate
            total_paying += paying

        avg_spend = _noise(tenant.avg_spend_per_visit, 0.30)
        monthly_revenue = total_paying * avg_spend

        rent_share = base_rent_share.get(tenant.brand_tier, 0.15)
        opcost_multiplier = _noise(1.0, 0.15)
        total_opcost = (monthly_rent_total / rent_share) * opcost_multiplier

        profit_ratio = (monthly_revenue / total_opcost
                        if total_opcost > 0 else 99.0)

        if profit_ratio >= 1.3:
            survival = 0.90
        elif profit_ratio >= 1.0:
            survival = 0.70
        elif profit_ratio >= 0.7:
            survival = 0.40
        elif profit_ratio >= 0.4:
            survival = 0.15
        else:
            survival = 0.05

        lease_months = int(tenant.lease_years * 12)
        if profit_ratio < 0.7:
            months_active = max(3, int(lease_months * profit_ratio / 2))
            vacancy_months = _noise(
                max(3, (1 - profit_ratio) * 12), 0.50)
            expected_rent = (monthly_rent_total * months_active)
        else:
            expected_rent = monthly_rent_total * lease_months * survival
            vacancy_months = 0 if survival > 0.8 else _noise(6, 0.50)
            expected_rent -= monthly_rent_total * vacancy_months * (
                1 - survival)

        results.append({
            "monthly_revenue": monthly_revenue,
            "profit_ratio": profit_ratio,
            "survival": survival,
            "expected_3yr_rent": max(0, expected_rent),
        })

    return results


def run_monte_carlo(
    config: MallConfig,
    iterations: int = 10000,
    seed: int | None = None,
) -> MallMCResult:
    if seed is not None:
        random.seed(seed)

    base_rent_share = {
        "budget": 0.20, "mid": 0.15, "premium": 0.10, "luxury": 0.06}
    enter_rate_by_gap = {0: 0.25, 1: 0.10, 2: 0.03, 3: 0.005}
    purchase_rate_by_tier = {
        "budget": 0.50, "mid": 0.30, "premium": 0.15, "luxury": 0.05}

    tenant_results = [
        TenantMCResult(
            tenant=t.name,
            brand_tier=t.brand_tier,
            monthly_rent=t.monthly_rent * t.area_sqm,
        )
        for t in config.tenants
    ]
    total_3yr_rents: list[float] = []

    for _ in range(iterations):
        iter_results = _run_single_iteration(
            config, base_rent_share, enter_rate_by_gap,
            purchase_rate_by_tier)

        iter_total = 0.0
        for i, r in enumerate(iter_results):
            tenant_results[i].monthly_revenues.append(r["monthly_revenue"])
            tenant_results[i].profit_ratios.append(r["profit_ratio"])
            tenant_results[i].survival_rates.append(r["survival"])
            tenant_results[i].expected_3yr_rents.append(
                r["expected_3yr_rent"])
            iter_total += r["expected_3yr_rent"]
        total_3yr_rents.append(iter_total)

    return MallMCResult(
        iterations=iterations,
        tenant_results=tenant_results,
        total_3yr_rents=total_3yr_rents,
    )
