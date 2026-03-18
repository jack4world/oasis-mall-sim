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
"""Integrated Analysis: Monte Carlo (quantitative) + LLM Agents (qualitative).

PIPELINE:
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1: Monte Carlo (no API cost, runs in seconds)               │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────────┐ │
│  │ MallConfig    │───▶│ 10K iterations │───▶│ Probability dists    │ │
│  └──────────────┘    └───────────────┘    │ per-tenant viability  │ │
│                                           │ total rental income   │ │
│                                           └──────────┬───────────┘ │
├──────────────────────────────────────────────────────┼─────────────┤
│  Phase 2: LLM Simulation (uses API, takes minutes)   │             │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────▼───────────┐ │
│  │ MallConfig    │───▶│ OASIS agents   │───▶│ Agent opinions       │ │
│  │ + MC results  │    │ discuss mall   │    │ sentiment, quotes    │ │
│  └──────────────┘    └───────────────┘    └──────────┬───────────┘ │
├──────────────────────────────────────────────────────┼─────────────┤
│  Phase 3: Merge into Investment Report               │             │
│  ┌──────────────────────────────────────────────────▼───────────┐ │
│  │ Per-tenant:                                                   │ │
│  │   - Survival probability (MC)                                 │ │
│  │   - Profit ratio distribution (MC)                            │ │
│  │   - Consumer sentiment quotes (LLM)                           │ │
│  │   - Affordability perception (LLM)                            │ │
│  │   - VERDICT: quantitative + qualitative alignment             │ │
│  │                                                               │ │
│  │ Mall-level:                                                   │ │
│  │   - 3yr rental income P10/P50/P90 (MC)                       │ │
│  │   - Overall sentiment score (LLM)                             │ │
│  │   - Top consumer complaints (LLM)                             │ │
│  │   - Risk factors (both)                                       │ │
│  │   - Fix recommendations with probability of improvement       │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from oasis.social_platform.config.mall import MallConfig
from oasis.social_platform.monte_carlo import MallMCResult, run_monte_carlo


@dataclass
class TenantVerdict:
    name: str
    brand_tier: str
    monthly_rent: float
    # Monte Carlo
    prob_profitable: float
    prob_survive_3yr: float
    profit_ratio_p50: float
    revenue_p50: float
    expected_3yr_rent_p50: float
    # LLM
    mention_count: int = 0
    positive_mentions: int = 0
    negative_mentions: int = 0
    sentiment: float = 0.0
    sample_quotes: list[str] = field(default_factory=list)
    perceived_expensive: bool = False
    perceived_affordable: bool = False
    # Combined
    verdict: str = ""
    confidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant": self.name,
            "brand_tier": self.brand_tier,
            "monthly_rent": self.monthly_rent,
            "quantitative": {
                "prob_profitable": f"{self.prob_profitable:.1f}%",
                "prob_survive_3yr": f"{self.prob_survive_3yr:.1f}%",
                "profit_ratio_p50": f"{self.profit_ratio_p50:.2f}x",
                "revenue_p50": round(self.revenue_p50, 0),
                "expected_3yr_rent_p50": round(self.expected_3yr_rent_p50, 0),
            },
            "qualitative": {
                "mention_count": self.mention_count,
                "positive_mentions": self.positive_mentions,
                "negative_mentions": self.negative_mentions,
                "sentiment": round(self.sentiment, 2),
                "perceived_expensive": self.perceived_expensive,
                "perceived_affordable": self.perceived_affordable,
                "sample_quotes": self.sample_quotes[:3],
            },
            "verdict": self.verdict,
            "confidence": self.confidence,
        }


@dataclass
class IntegratedReport:
    mall_name: str
    population: int
    num_agents: int
    mc_iterations: int
    llm_timesteps: int
    # MC totals
    total_3yr_rent_p10: float = 0.0
    total_3yr_rent_p50: float = 0.0
    total_3yr_rent_p90: float = 0.0
    total_3yr_if_all_stay: float = 0.0
    # LLM totals
    overall_sentiment: float = 0.0
    total_posts: int = 0
    total_comments: int = 0
    # Combined
    tenant_verdicts: list[TenantVerdict] = field(default_factory=list)
    top_complaints: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    recommendations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mall": self.mall_name,
            "population": self.population,
            "methodology": {
                "monte_carlo_iterations": self.mc_iterations,
                "llm_agents": self.num_agents,
                "llm_timesteps": self.llm_timesteps,
            },
            "financial_summary": {
                "3yr_rent_if_all_stay": round(self.total_3yr_if_all_stay, 0),
                "3yr_rent_P10_worst": round(self.total_3yr_rent_p10, 0),
                "3yr_rent_P50_median": round(self.total_3yr_rent_p50, 0),
                "3yr_rent_P90_best": round(self.total_3yr_rent_p90, 0),
                "collection_rate_p50": (
                    round(self.total_3yr_rent_p50
                          / self.total_3yr_if_all_stay * 100, 1)
                    if self.total_3yr_if_all_stay > 0 else 0),
            },
            "consumer_sentiment": {
                "overall_score": round(self.overall_sentiment, 2),
                "total_posts": self.total_posts,
                "total_comments": self.total_comments,
                "top_complaints": self.top_complaints,
            },
            "tenant_verdicts": [t.to_dict() for t in self.tenant_verdicts],
            "risk_factors": self.risk_factors,
            "recommendations": self.recommendations,
        }


PRICE_POSITIVE = [
    "affordable", "cheap", "good price", "good value", "worth it",
    "reasonable", "bargain", "budget-friendly", "great deal",
    "fair price", "value for money",
]
PRICE_NEGATIVE = [
    "expensive", "overpriced", "pricey", "too much", "can't afford",
    "costly", "not worth", "rip off", "too expensive", "out of budget",
    "out of place", "out of reach", "steep",
]


def _extract_tenant_quotes(
    all_text: list[str], tenant_name: str, max_quotes: int = 5,
) -> list[str]:
    quotes = []
    name_lower = tenant_name.lower()
    for text in all_text:
        if name_lower in text.lower():
            # Extract the sentence containing the tenant name
            sentences = re.split(r'[.!?]', text)
            for s in sentences:
                if name_lower in s.lower() and len(s.strip()) > 20:
                    quote = s.strip()
                    if len(quote) > 150:
                        quote = quote[:147] + "..."
                    quotes.append(quote)
                    if len(quotes) >= max_quotes:
                        return quotes
    return quotes


def _classify_sentiment_for_tenant(
    all_text: list[str], tenant_name: str,
) -> tuple[int, int, int, bool, bool]:
    """Returns (mentions, positive, negative, perceived_expensive, perceived_affordable)."""
    name_lower = tenant_name.lower()
    mentions = 0
    positive = 0
    negative = 0
    seen_expensive = False
    seen_affordable = False

    for text in all_text:
        lower = text.lower()
        if name_lower not in lower:
            continue
        mentions += 1

        # Find sentences mentioning this tenant
        sentences = re.split(r'[.!?,;]', lower)
        for s in sentences:
            if name_lower not in s:
                continue
            has_pos = any(kw in s for kw in PRICE_POSITIVE)
            has_neg = any(kw in s for kw in PRICE_NEGATIVE)
            if has_pos:
                positive += 1
                seen_affordable = True
            if has_neg:
                negative += 1
                seen_expensive = True

    return mentions, positive, negative, seen_expensive, seen_affordable


def generate_integrated_report(
    mall_config: MallConfig,
    db_path: str,
    llm_timesteps: int,
    mc_iterations: int = 10000,
    mc_seed: int = 42,
) -> IntegratedReport:
    # Phase 1: Monte Carlo
    mc_result = run_monte_carlo(mall_config, mc_iterations, mc_seed)
    mc_summary = mc_result.summary()

    # Phase 2: Read LLM simulation DB
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as cnt FROM user")
    num_agents = cursor.fetchone()["cnt"]

    cursor.execute(
        "SELECT SUM(num_likes) as l, SUM(num_dislikes) as d, "
        "COUNT(*) as cnt FROM post")
    post_row = cursor.fetchone()
    total_posts = post_row["cnt"]
    total_likes = post_row["l"] or 0
    total_dislikes = post_row["d"] or 0

    cursor.execute("SELECT COUNT(*) as cnt FROM comment")
    total_comments = cursor.fetchone()["cnt"]

    total_eng = total_likes + total_dislikes
    overall_sentiment = (
        (total_likes - total_dislikes) / total_eng
        if total_eng > 0 else 0.0)

    # Gather all text for qualitative analysis
    cursor.execute("SELECT content FROM post")
    post_texts = [r["content"] for r in cursor.fetchall() if r["content"]]
    cursor.execute("SELECT content FROM comment")
    comment_texts = [r["content"] for r in cursor.fetchall() if r["content"]]
    all_text = post_texts + comment_texts

    conn.close()

    # Phase 3: Merge
    population = sum(r.population for r in mall_config.location_rings)
    total_rent_3yr = sum(
        t.monthly_rent * t.area_sqm for t in mall_config.tenants) * 36

    report = IntegratedReport(
        mall_name=mall_config.building.name,
        population=population,
        num_agents=num_agents,
        mc_iterations=mc_iterations,
        llm_timesteps=llm_timesteps,
        total_3yr_rent_p10=mc_summary["total_3yr_rent_p10"],
        total_3yr_rent_p50=mc_summary["total_3yr_rent_p50"],
        total_3yr_rent_p90=mc_summary["total_3yr_rent_p90"],
        total_3yr_if_all_stay=total_rent_3yr,
        overall_sentiment=overall_sentiment,
        total_posts=total_posts,
        total_comments=total_comments,
    )

    # Per-tenant verdicts
    for i, tenant in enumerate(mall_config.tenants):
        mc_t = mc_summary["tenants"][i]
        mentions, pos, neg, expensive, affordable = (
            _classify_sentiment_for_tenant(all_text, tenant.name))
        quotes = _extract_tenant_quotes(all_text, tenant.name)

        t_sentiment = (
            (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0)

        verdict_obj = TenantVerdict(
            name=tenant.name,
            brand_tier=tenant.brand_tier,
            monthly_rent=tenant.monthly_rent * tenant.area_sqm,
            prob_profitable=mc_t["prob_profitable"],
            prob_survive_3yr=mc_t["prob_survive_3yr"],
            profit_ratio_p50=mc_t["profit_ratio_p50"],
            revenue_p50=mc_t["revenue_p50"],
            expected_3yr_rent_p50=mc_t["expected_3yr_rent_p50"],
            mention_count=mentions,
            positive_mentions=pos,
            negative_mentions=neg,
            sentiment=t_sentiment,
            sample_quotes=quotes,
            perceived_expensive=expensive,
            perceived_affordable=affordable,
        )

        # Verdict logic: combine quantitative + qualitative
        mc_dead = mc_t["prob_profitable"] < 5
        mc_risky = mc_t["prob_profitable"] < 50
        mc_safe = mc_t["prob_profitable"] >= 70
        llm_negative = neg > pos and mentions >= 3
        llm_positive = pos > neg or (mentions >= 3 and neg == 0)

        if mc_dead and llm_negative:
            verdict_obj.verdict = "REPLACE IMMEDIATELY"
            verdict_obj.confidence = "VERY HIGH"
        elif mc_dead and not llm_negative:
            verdict_obj.verdict = "REPLACE (financials unsustainable)"
            verdict_obj.confidence = "HIGH"
        elif mc_risky and llm_negative:
            verdict_obj.verdict = "AT RISK — monitor closely"
            verdict_obj.confidence = "HIGH"
        elif mc_risky:
            verdict_obj.verdict = "AT RISK — financials marginal"
            verdict_obj.confidence = "MEDIUM"
        elif mc_safe and llm_positive:
            verdict_obj.verdict = "KEEP — strong fit"
            verdict_obj.confidence = "VERY HIGH"
        elif mc_safe:
            verdict_obj.verdict = "KEEP — financially viable"
            verdict_obj.confidence = "HIGH"
        else:
            verdict_obj.verdict = "MONITOR"
            verdict_obj.confidence = "LOW"

        if mc_dead:
            report.risk_factors.append(
                f"{tenant.name}: {mc_t['prob_profitable']:.0f}% profitable "
                f"(MC), {neg} negative mentions (LLM) → "
                f"{verdict_obj.verdict}")

        report.tenant_verdicts.append(verdict_obj)

    # Top complaints from LLM
    complaint_counts: dict[str, int] = {}
    complaint_patterns = [
        (r"out of place", "Tenant feels out of place for area"),
        (r"too expensive|overpriced|can't afford", "Prices too high"),
        (r"redundant|unnecessary|both .* on the same",
         "Duplicate tenants on same floor"),
        (r"missing|wish there|more .* options",
         "Missing categories wanted by consumers"),
        (r"parking|traffic|crowded", "Access/congestion concerns"),
    ]
    for text in all_text:
        lower = text.lower()
        for pattern, label in complaint_patterns:
            if re.search(pattern, lower):
                complaint_counts[label] = complaint_counts.get(label, 0) + 1

    report.top_complaints = [
        f"{label} ({count} mentions)"
        for label, count in sorted(
            complaint_counts.items(), key=lambda x: -x[1])
    ]

    return report


def print_integrated_report(report: IntegratedReport) -> None:
    print()
    print("=" * 80)
    print(f"  INVESTMENT ANALYSIS — {report.mall_name}")
    print(f"  Monte Carlo ({report.mc_iterations:,} iterations) + "
          f"LLM Simulation ({report.num_agents} agents, "
          f"{report.llm_timesteps} timesteps)")
    print("=" * 80)

    # Financial summary
    coll_rate = (report.total_3yr_rent_p50 / report.total_3yr_if_all_stay
                 * 100 if report.total_3yr_if_all_stay > 0 else 0)
    print()
    print("  FINANCIAL FORECAST (3-year)")
    print(f"    If all tenants stay:   ¥{report.total_3yr_if_all_stay:>12,.0f}")
    print(f"    P10 (worst case):      ¥{report.total_3yr_rent_p10:>12,.0f}")
    print(f"    P50 (median):          ¥{report.total_3yr_rent_p50:>12,.0f}"
          f"  ({coll_rate:.0f}% collection)")
    print(f"    P90 (best case):       ¥{report.total_3yr_rent_p90:>12,.0f}")

    # Consumer sentiment
    print()
    print(f"  CONSUMER SENTIMENT ({report.total_posts} posts, "
          f"{report.total_comments} comments)")
    print(f"    Overall: {report.overall_sentiment:+.2f} "
          f"({'positive' if report.overall_sentiment > 0 else 'negative'})")
    if report.top_complaints:
        print("    Top complaints:")
        for c in report.top_complaints[:5]:
            print(f"      - {c}")

    # Per-tenant verdicts
    print()
    print("  TENANT VERDICTS")
    print(f"  {'Tenant':<14} {'P(profit)':<10} {'P(surv)':<9} "
          f"{'Sentiment':<11} {'Verdict':<28} {'Confidence'}")
    print(f"  {'-'*14} {'-'*10} {'-'*9} {'-'*11} {'-'*28} {'-'*10}")

    for v in sorted(report.tenant_verdicts,
                    key=lambda x: x.prob_profitable):
        sent_str = (
            f"+{v.positive_mentions}/-{v.negative_mentions}"
            if v.mention_count > 0 else "no data")
        icon = {"VERY HIGH": "★", "HIGH": "●", "MEDIUM": "○",
                "LOW": "?"}.get(v.confidence, "?")
        print(f"  {v.name:<14} {v.prob_profitable:>7.1f}%  "
              f"{v.prob_survive_3yr:>6.1f}%  "
              f"{sent_str:<11} {v.verdict:<28} {icon} {v.confidence}")

        if v.sample_quotes:
            print(f"    └ \"{v.sample_quotes[0][:75]}\"")

    # Risk factors
    if report.risk_factors:
        print()
        print("  RISK FACTORS")
        for r in report.risk_factors:
            print(f"    ⚠ {r}")

    print()
    print("=" * 80)


def export_integrated_report(
    report: IntegratedReport, output_path: str,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False,
                  default=str)
