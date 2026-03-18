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
from dataclasses import dataclass, field
from typing import Any


TIER_TARGET_INCOME: dict[str, str] = {
    "budget": "low",
    "mid": "mid",
    "premium": "mid-high",
    "luxury": "high",
}

INCOME_RANK: dict[str, int] = {
    "low": 1, "budget": 1, "mid": 2, "mixed": 2,
    "mid-high": 3, "high": 4,
}


@dataclass
class TenantConfig:
    name: str
    category: str  # "F&B", "Fashion", "Entertainment", "Service"
    floor: int = 1
    brand_tier: str = "mid"  # "budget", "mid", "premium", "luxury"
    avg_spend_per_visit: float = 0.0  # average customer spend (¥)
    monthly_rent: float = 0.0  # monthly rent (¥/m²)
    area_sqm: float = 0.0  # leased area (m²)
    lease_years: float = 3.0  # lease term


@dataclass
class TransportConfig:
    metro_distance_m: int | None = None
    metro_lines: list[str] = field(default_factory=list)
    bus_lines: int = 0
    parking_spots: int = 0
    parking_price_per_hour: float = 0.0
    walk_score: int = 50


@dataclass
class LocationRing:
    radius_km: float
    population: int
    demographics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompetitorMall:
    name: str
    distance_km: float
    positioning: str = "general"


@dataclass
class BuildingConfig:
    name: str = "Mall"
    floors: int = 3
    building_type: str = "modern"
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
        lines = []
        b = self.building
        lines.append(
            f'"{b.name}" is a {b.floors}-floor {b.building_type} '
            f"shopping center.")
        if b.year_renovated:
            lines.append(f"Renovated in {b.year_renovated}.")
        if b.photo_spots:
            lines.append(
                f"Popular photo spots: {', '.join(b.photo_spots)}.")

        if self.tenants:
            lines.append("")
            lines.append("Tenant directory (floor by floor):")
            by_floor: dict[int, list[TenantConfig]] = {}
            for t in self.tenants:
                by_floor.setdefault(t.floor, []).append(t)
            for floor_num in sorted(by_floor.keys()):
                floor_tenants = by_floor[floor_num]
                tenant_strs = []
                for t in floor_tenants:
                    desc = f"{t.name} ({t.category}, {t.brand_tier})"
                    if t.avg_spend_per_visit > 0:
                        desc += f" ~¥{t.avg_spend_per_visit:.0f}/visit"
                    tenant_strs.append(desc)
                lines.append(
                    f"  Floor {floor_num}: {', '.join(tenant_strs)}")

            categories = {}
            for t in self.tenants:
                categories[t.category] = categories.get(t.category, 0) + 1
            total = len(self.tenants)
            mix_parts = [f"{cat} {n}/{total}" for cat, n in categories.items()]
            lines.append(f"  Tenant mix: {', '.join(mix_parts)}")

        lines.append("")
        t = self.transport
        if t.metro_distance_m is not None:
            lines.append(
                f"Metro ({', '.join(t.metro_lines)}) is "
                f"{t.metro_distance_m}m away.")
        if t.bus_lines > 0:
            lines.append(f"{t.bus_lines} bus lines nearby.")
        if t.parking_spots > 0:
            lines.append(
                f"Parking: {t.parking_spots} spots, "
                f"¥{t.parking_price_per_hour}/hr.")
        lines.append(f"Walk score: {t.walk_score}/100.")

        for c in self.competitors:
            lines.append(
                f'Nearby competitor: "{c.name}" ({c.positioning}), '
                f"{c.distance_km}km away.")

        lines.append("")
        lines.append(
            "When you post or comment about this mall, share your honest "
            "opinion: Is the tenant mix right for people like you? Are the "
            "prices affordable? Which stores would you actually visit, and "
            "which feel out of place or too expensive? What's missing?")

        return "\n".join(lines)

    def get_visit_probability(self, ring_radius_km: float) -> float:
        """Accessibility model mapping distance + transport to visit probability.

        ┌───────────────┬────────┬────────┬────────┐
        │ Transport      │ 1km    │ 3km    │ 5km    │
        ├───────────────┼────────┼────────┼────────┤
        │ Metro < 500m  │ 0.8    │ 0.5    │ 0.3    │
        │ Metro >= 500m │ 0.6    │ 0.3    │ 0.15   │
        │ No metro      │ 0.4    │ 0.15   │ 0.05   │
        └───────────────┴────────┴────────┴────────┘
        Parking bonus: +0.1 if parking_spots > 200
        """
        base_probs: dict[int, dict[bool, float]] = {
            1: {True: 0.8, False: 0.4},
            3: {True: 0.5, False: 0.15},
            5: {True: 0.3, False: 0.05},
        }
        closest_ring = min(
            base_probs.keys(), key=lambda k: abs(k - ring_radius_km))
        has_close_metro = (
            self.transport.metro_distance_m is not None
            and self.transport.metro_distance_m < 500)
        prob = base_probs[closest_ring][has_close_metro]

        if (self.transport.metro_distance_m is not None
                and self.transport.metro_distance_m >= 500
                and closest_ring > 1):
            prob *= 0.7

        if self.transport.parking_spots > 200:
            prob = min(prob + 0.1, 1.0)

        return prob

    def get_dominant_income_level(self) -> str:
        if not self.location_rings:
            return "mid"
        best_ring = max(self.location_rings, key=lambda r: r.population)
        return best_ring.demographics.get("income_level", "mid")

    def _estimate_monthly_paying_customers(
        self, tenant: TenantConfig,
    ) -> float:
        """Estimate monthly PAYING customers for a tenant.

        CONVERSION FUNNEL:
        ┌─────────────────┐
        │  Ring Population  │
        └────────┬────────┘
                 │ × visit_prob (distance + transport)
                 ▼
        ┌─────────────────┐
        │  Mall Visitors    │  (people who come to the mall)
        └────────┬────────┘
                 │ × enter_rate (who walks into this store?)
                 ▼
        ┌─────────────────┐
        │  Store Foot      │  (people browsing)
        │  Traffic          │
        └────────┬────────┘
                 │ × purchase_rate (who actually buys?)
                 ▼
        ┌─────────────────┐
        │  Paying Customers │  → × avg_spend = REVENUE
        └─────────────────┘

        Enter rate (walk into the store):
          gap ≤ 0 → 25%    (affordable, many enter)
          gap = 1 → 10%    (slightly above budget, some enter)
          gap = 2 → 3%     (expensive, few enter)
          gap ≥ 3 → 0.5%   (way too expensive, almost nobody enters)

        Purchase rate (actually buy something):
          budget  → 50%  (cheap impulse buys)
          mid     → 30%  (considered purchase)
          premium → 15%  (selective buying)
          luxury  → 5%   (very few actually buy)
        """
        enter_rate_by_gap = {0: 0.25, 1: 0.10, 2: 0.03, 3: 0.005}
        purchase_rate_by_tier = {
            "budget": 0.50, "mid": 0.30, "premium": 0.15, "luxury": 0.05}

        tier_rank = INCOME_RANK.get(
            TIER_TARGET_INCOME.get(tenant.brand_tier, "mid"), 2)
        purchase_rate = purchase_rate_by_tier.get(tenant.brand_tier, 0.30)

        total_paying = 0.0
        for ring in self.location_rings:
            ring_income = ring.demographics.get("income_level", "mid")
            area_rank = INCOME_RANK.get(ring_income, 2)
            gap = max(0, tier_rank - area_rank)
            enter_rate = enter_rate_by_gap.get(gap, 0.005)
            visit_prob = self.get_visit_probability(ring.radius_km)
            monthly_mall_visitors = ring.population * visit_prob * 30
            store_foot_traffic = monthly_mall_visitors * enter_rate
            paying_customers = store_foot_traffic * purchase_rate
            total_paying += paying_customers

        return total_paying

    def compute_tenant_viability(self) -> list[dict[str, Any]]:
        """Revenue-driven tenant viability: can this store make enough
        money from local customers to cover its rent?

        TENANT VIABILITY MODEL:
        ┌──────────────────┐    ┌────────────────┐
        │ Local Population  │    │ Brand Tier vs   │
        │ × Visit Prob      │    │ Area Income     │
        └────────┬─────────┘    └───────┬────────┘
                 │                       │
                 ▼                       ▼
        ┌─────────────────────────────────────────┐
        │  Est. Monthly Customers                  │
        │  = Σ(ring_pop × visit_prob × afford_rate)│
        └────────────────────┬────────────────────┘
                             │ × avg_spend_per_visit
                             ▼
        ┌─────────────────────────────────────────┐
        │  Est. Monthly Revenue                    │
        └────────────────────┬────────────────────┘
                             │ vs monthly_rent
                             ▼
        ┌─────────────────────────────────────────┐
        │  Rent Coverage Ratio = Revenue / Rent    │
        │                                         │
        │  ratio > 3.0  → SAFE (90% survive)      │
        │  ratio 2-3    → HEALTHY (80% survive)   │
        │  ratio 1-2    → STRESSED (50% survive)  │
        │  ratio 0.5-1  → DANGER (20% survive)    │
        │  ratio < 0.5  → WILL LEAVE (5% survive) │
        └─────────────────────────────────────────┘

        A store's total operating cost is much higher than rent alone.
        Rent-to-total-cost ratio by tier:
          budget  → rent is ~20% of total cost (low staff, cheap inventory)
          mid     → rent is ~15% of total cost
          premium → rent is ~10% of total cost (more staff, displays)
          luxury  → rent is ~6% of total cost (high staff, huge inventory,
                    brand standards, visual merchandising)

        So total breakeven = rent / rent_share_of_cost.
        A store is viable when revenue ≥ total_operating_cost.
        """
        rent_share_of_cost = {
            "budget": 0.20, "mid": 0.15, "premium": 0.10, "luxury": 0.06}

        results = []
        for t in self.tenants:
            monthly_rent_total = t.monthly_rent * t.area_sqm
            monthly_customers = self._estimate_monthly_paying_customers(t)
            monthly_revenue = monthly_customers * t.avg_spend_per_visit

            rent_share = rent_share_of_cost.get(t.brand_tier, 0.15)
            total_operating_cost = monthly_rent_total / rent_share
            if total_operating_cost > 0:
                profit_ratio = monthly_revenue / total_operating_cost
            else:
                profit_ratio = 99.0

            if profit_ratio >= 1.3:
                survival = 0.90
                status = "SAFE"
            elif profit_ratio >= 1.0:
                survival = 0.70
                status = "MARGINAL"
            elif profit_ratio >= 0.7:
                survival = 0.40
                status = "LOSING_MONEY"
            elif profit_ratio >= 0.4:
                survival = 0.15
                status = "WILL_LEAVE"
            else:
                survival = 0.05
                status = "DEAD"

            vacancy_months = (
                3 if survival >= 0.7 else 6 if survival >= 0.4 else 9)
            vacancy_cost = monthly_rent_total * vacancy_months
            expected_loss = (1 - survival) * vacancy_cost

            lease_months = int(t.lease_years * 12)
            lease_total = monthly_rent_total * lease_months
            if status in ("WILL_LEAVE", "DEAD"):
                months_before_exit = max(
                    3, int(lease_months * profit_ratio / 2))
                expected_rent_collected = (
                    monthly_rent_total * months_before_exit)
            elif status == "LOSING_MONEY":
                months_before_exit = max(
                    6, int(lease_months * profit_ratio))
                expected_rent_collected = (
                    monthly_rent_total * months_before_exit)
            else:
                expected_rent_collected = lease_total * survival

            results.append({
                "tenant": t.name,
                "brand_tier": t.brand_tier,
                "category": t.category,
                "floor": t.floor,
                "area_sqm": t.area_sqm,
                "monthly_rent_total": round(monthly_rent_total, 0),
                "est_monthly_customers": round(monthly_customers, 0),
                "est_monthly_revenue": round(monthly_revenue, 0),
                "total_operating_cost": round(total_operating_cost, 0),
                "profit_ratio": round(profit_ratio, 2),
                "status": status,
                "survival_3yr_pct": round(survival * 100, 1),
                "vacancy_months_if_leave": vacancy_months,
                "vacancy_cost": round(vacancy_cost, 0),
                "expected_vacancy_loss": round(expected_loss, 0),
                "expected_rent_collected_3yr": round(
                    expected_rent_collected, 0),
            })

        return results
