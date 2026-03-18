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
import json
import os
import sqlite3

import pytest

from oasis.social_platform.config.mall import (BuildingConfig, CompetitorMall,
                                                LocationRing, MallConfig,
                                                TenantConfig, TransportConfig)
from oasis.social_platform.database import create_db
from oasis.social_platform.mall_analytics import MallAnalytics, ScenarioReport


class TestMallConfig:

    def test_to_agent_context_basic(self):
        config = MallConfig(
            building=BuildingConfig(name="Test Mall", floors=3),
            tenants=[TenantConfig("Shop A", "Retail", floor=1)],
            transport=TransportConfig(metro_distance_m=200,
                                     metro_lines=["Line 2"]),
        )
        context = config.to_agent_context()
        assert "Test Mall" in context
        assert "3-floor" in context
        assert "Shop A" in context
        assert "Metro" in context
        assert "200m" in context

    def test_to_agent_context_with_all_fields(self):
        config = MallConfig(
            building=BuildingConfig(name="Grand Mall", floors=5,
                                   building_type="luxury",
                                   photo_spots=["fountain", "sky_garden"],
                                   year_renovated=2025),
            tenants=[
                TenantConfig("Starbucks", "F&B", floor=1),
                TenantConfig("Uniqlo", "Fashion", floor=2),
                TenantConfig("Haidilao", "F&B", floor=3),
            ],
            transport=TransportConfig(metro_distance_m=100,
                                     metro_lines=["L1", "L3"],
                                     bus_lines=5, parking_spots=800,
                                     parking_price_per_hour=10,
                                     walk_score=90),
            competitors=[CompetitorMall("Rival", 1.5, "premium")],
        )
        context = config.to_agent_context()
        assert "Renovated in 2025" in context
        assert "fountain" in context
        assert "sky_garden" in context
        assert "Starbucks" in context
        assert "Haidilao" in context
        assert "5 bus lines" in context
        assert "800 spots" in context
        assert "Rival" in context
        assert "1.5km" in context

    def test_to_agent_context_empty(self):
        config = MallConfig()
        context = config.to_agent_context()
        assert "Mall" in context
        assert "Walk score" in context

    def test_visit_probability_close_metro(self):
        config = MallConfig(
            transport=TransportConfig(metro_distance_m=200,
                                     metro_lines=["L1"]),
        )
        assert config.get_visit_probability(1) == pytest.approx(0.8)
        assert config.get_visit_probability(3) == pytest.approx(0.5)
        assert config.get_visit_probability(5) == pytest.approx(0.3)

    def test_visit_probability_no_metro(self):
        config = MallConfig(
            transport=TransportConfig(metro_distance_m=None),
        )
        assert config.get_visit_probability(1) == pytest.approx(0.4)
        assert config.get_visit_probability(3) == pytest.approx(0.15)
        assert config.get_visit_probability(5) == pytest.approx(0.05)

    def test_visit_probability_far_metro(self):
        config = MallConfig(
            transport=TransportConfig(metro_distance_m=800,
                                     metro_lines=["L1"]),
        )
        prob_3km = config.get_visit_probability(3)
        assert prob_3km < 0.15  # far metro penalty

    def test_visit_probability_parking_bonus(self):
        config_no_parking = MallConfig(
            transport=TransportConfig(metro_distance_m=200,
                                     metro_lines=["L1"],
                                     parking_spots=50),
        )
        config_parking = MallConfig(
            transport=TransportConfig(metro_distance_m=200,
                                     metro_lines=["L1"],
                                     parking_spots=500),
        )
        prob_no = config_no_parking.get_visit_probability(3)
        prob_yes = config_parking.get_visit_probability(3)
        assert prob_yes == prob_no + 0.1


class TestDBSchema:

    def setup_method(self):
        self.db_path = "/tmp/test_mall_schema.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.conn, self.cursor = create_db(self.db_path)

    def teardown_method(self):
        self.conn.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_store_table_exists(self):
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND "
            "name='store'")
        assert self.cursor.fetchone() is not None

    def test_visit_table_exists(self):
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND "
            "name='visit'")
        assert self.cursor.fetchone() is not None

    def test_store_insert_and_query(self):
        self.cursor.execute(
            "INSERT INTO store (store_name, category, floor, brand_tier) "
            "VALUES (?, ?, ?, ?)", ("TestStore", "Retail", 2, "premium"))
        self.conn.commit()
        self.cursor.execute("SELECT * FROM store WHERE store_name='TestStore'")
        row = self.cursor.fetchone()
        assert row is not None
        assert row[1] == "TestStore"
        assert row[2] == "Retail"
        assert row[3] == 2
        assert row[4] == "premium"

    def test_visit_insert_and_query(self):
        self.cursor.execute(
            "INSERT INTO store (store_name) VALUES (?)", ("S1", ))
        self.cursor.execute(
            "INSERT INTO user (user_id, agent_id, user_name, name, bio, "
            "created_at, num_followings, num_followers) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?)",
            (1, 1, "u1", "User 1", "bio", "2026-01-01", 0, 0))
        self.conn.commit()
        self.cursor.execute(
            "INSERT INTO visit (user_id, store_id, created_at) "
            "VALUES (?, ?, ?)", (1, 1, "2026-01-01 10:00:00"))
        self.conn.commit()
        self.cursor.execute("SELECT * FROM visit")
        row = self.cursor.fetchone()
        assert row is not None
        assert row[1] == 1  # user_id
        assert row[2] == 1  # store_id


class TestMallAnalytics:

    def setup_method(self):
        self.db_path = "/tmp/test_mall_analytics.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.conn, self.cursor = create_db(self.db_path)
        self._setup_test_data()

    def _setup_test_data(self):
        self.cursor.execute(
            "INSERT INTO store (store_name, category, floor, brand_tier) "
            "VALUES (?, ?, ?, ?)", ("StoreA", "F&B", 1, "mid"))
        self.cursor.execute(
            "INSERT INTO store (store_name, category, floor, brand_tier) "
            "VALUES (?, ?, ?, ?)", ("StoreB", "Fashion", 2, "mid"))
        for i in range(3):
            self.cursor.execute(
                "INSERT INTO user (user_id, agent_id, user_name, name, bio, "
                "created_at, num_followings, num_followers) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?)",
                (i, i, f"u{i}", f"User{i}", f"bio{i}", "2026-01-01", 0, 0))
        self.cursor.execute(
            "INSERT INTO visit (user_id, store_id, created_at) "
            "VALUES (?, ?, ?)", (0, 1, "2026-01-01 10:00"))
        self.cursor.execute(
            "INSERT INTO visit (user_id, store_id, created_at) "
            "VALUES (?, ?, ?)", (1, 1, "2026-01-01 11:00"))
        self.cursor.execute(
            "INSERT INTO visit (user_id, store_id, created_at) "
            "VALUES (?, ?, ?)", (2, 2, "2026-01-01 12:00"))
        self.cursor.execute(
            "INSERT INTO post (user_id, content, created_at, num_likes, "
            "num_dislikes, num_shares) VALUES (?, ?, ?, ?, ?, ?)",
            (0, "[Review: StoreA] Great!", "2026-01-01 10:30", 5, 1, 2))
        self.cursor.execute(
            "INSERT INTO post (user_id, content, created_at, num_likes, "
            "num_dislikes, num_shares) VALUES (?, ?, ?, ?, ?, ?)",
            (1, "[Review: StoreA] Bad service", "2026-01-01 11:30", 1, 4, 0))
        self.conn.commit()

    def teardown_method(self):
        self.conn.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_generate_report_visits(self):
        analytics = MallAnalytics(self.db_path)
        report = analytics.generate_report("test")
        assert report.total_visits == 3

    def test_generate_report_posts(self):
        analytics = MallAnalytics(self.db_path)
        report = analytics.generate_report("test")
        assert report.total_posts == 2

    def test_generate_report_engagement(self):
        analytics = MallAnalytics(self.db_path)
        report = analytics.generate_report("test")
        assert report.total_likes == 6
        assert report.total_dislikes == 5

    def test_generate_report_sentiment(self):
        analytics = MallAnalytics(self.db_path)
        report = analytics.generate_report("test")
        expected = (6 - 5) / (6 + 5)
        assert report.sentiment_score == pytest.approx(expected, abs=0.01)

    def test_generate_report_store_popularity(self):
        analytics = MallAnalytics(self.db_path)
        report = analytics.generate_report("test")
        assert len(report.store_popularity) == 2
        assert report.store_popularity[0]["store_name"] == "StoreA"
        assert report.store_popularity[0]["visit_count"] == 2

    def test_generate_report_viral_posts(self):
        analytics = MallAnalytics(self.db_path)
        report = analytics.generate_report("test")
        assert len(report.viral_posts) == 2
        assert report.viral_posts[0]["num_likes"] == 5

    def test_generate_report_risk_factors(self):
        analytics = MallAnalytics(self.db_path)
        report = analytics.generate_report("test")
        assert any("dislikes" in r for r in report.risk_factors)

    def test_compare_scenarios(self):
        analytics = MallAnalytics(self.db_path)
        report_a = ScenarioReport(scenario_name="a", total_visits=10,
                                  sentiment_score=0.8)
        report_b = ScenarioReport(scenario_name="b", total_visits=5,
                                  sentiment_score=0.3)
        comparison = analytics.compare_scenarios([report_a, report_b])
        assert comparison["recommendation"] == "a"
        assert len(comparison["scenarios"]) == 2

    def test_compare_scenarios_ordering(self):
        analytics = MallAnalytics(self.db_path)
        report_a = ScenarioReport(scenario_name="low", total_visits=2,
                                  sentiment_score=0.1)
        report_b = ScenarioReport(scenario_name="high", total_visits=20,
                                  sentiment_score=0.9)
        comparison = analytics.compare_scenarios([report_a, report_b])
        assert comparison["scenarios"][0]["scenario_name"] == "high"

    def test_export_json(self):
        analytics = MallAnalytics(self.db_path)
        report = analytics.generate_report("export_test")
        export_path = "/tmp/test_export.json"
        analytics.export_json(report, export_path)
        with open(export_path) as f:
            data = json.load(f)
        assert data["scenario_name"] == "export_test"
        assert data["summary"]["total_visits"] == 3
        os.remove(export_path)

    def test_to_dict_roundtrip(self):
        report = ScenarioReport(
            scenario_name="rt", total_visits=7, total_posts=3,
            total_likes=10, total_dislikes=2, sentiment_score=0.667,
            risk_factors=["test risk"])
        d = report.to_dict()
        assert d["scenario_name"] == "rt"
        assert d["summary"]["total_visits"] == 7
        assert d["risk_factors"] == ["test risk"]

    def test_empty_db_report(self):
        empty_db = "/tmp/test_empty_analytics.db"
        if os.path.exists(empty_db):
            os.remove(empty_db)
        conn, _ = create_db(empty_db)
        conn.close()
        analytics = MallAnalytics(empty_db)
        report = analytics.generate_report("empty")
        assert report.total_visits == 0
        assert report.total_posts == 0
        assert report.sentiment_score == 0.0
        os.remove(empty_db)
