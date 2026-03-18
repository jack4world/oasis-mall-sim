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

import random
from typing import Any, List, Optional, Union

from camel.models import BaseModelBackend, ModelManager

from oasis.social_agent.agent import SocialAgent
from oasis.social_agent.agent_graph import AgentGraph
from oasis.social_platform.config.mall import LocationRing, MallConfig
from oasis.social_platform.config.user import UserInfo
from oasis.social_platform.typing import ActionType

MALL_DEFAULT_ACTIONS = [
    ActionType.CREATE_POST,
    ActionType.LIKE_POST,
    ActionType.DISLIKE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.LIKE_COMMENT,
    ActionType.FOLLOW,
    ActionType.SEARCH_POSTS,
    ActionType.REFRESH,
    ActionType.DO_NOTHING,
    ActionType.VISIT_STORE,
    ActionType.WRITE_REVIEW,
]

MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP",
]


MONTHLY_BUDGET_BY_INCOME: dict[str, tuple[int, int]] = {
    "low": (1000, 3000),
    "budget": (1500, 4000),
    "mid": (3000, 8000),
    "mid-high": (5000, 15000),
    "high": (10000, 30000),
    "mixed": (2000, 12000),
}


def _generate_profile_from_ring(
    ring: LocationRing,
    mall_config: MallConfig,
) -> dict[str, Any]:
    demo = ring.demographics
    age_range = demo.get("age_range", "20-50")
    age_min, age_max = map(int, age_range.split("-"))
    age = random.randint(age_min, age_max)
    gender = random.choice(["male", "female"])
    income = demo.get("income_level", "mid")
    person_type = demo.get("type", "general")
    mbti = random.choice(MBTI_TYPES)
    budget_range = MONTHLY_BUDGET_BY_INCOME.get(income, (3000, 8000))
    monthly_shopping_budget = random.randint(*budget_range)

    return {
        "age": age,
        "gender": gender,
        "income_level": income,
        "person_type": person_type,
        "mbti": mbti,
        "ring_radius_km": ring.radius_km,
        "visit_probability": mall_config.get_visit_probability(
            ring.radius_km),
        "mall_context": mall_config.to_agent_context(),
        "monthly_shopping_budget": monthly_shopping_budget,
    }


async def generate_mall_agent_graph(
    mall_config: MallConfig,
    model: Optional[Union[BaseModelBackend, List[BaseModelBackend],
                          ModelManager]],
    agent_scale: float = 0.01,
    available_actions: list[ActionType] | None = None,
) -> AgentGraph:
    r"""Generate an AgentGraph from a MallConfig.

    Args:
        mall_config: The mall configuration to simulate.
        model: The LLM model for agents.
        agent_scale: Fraction of ring population to create as agents.
            0.01 means 1% of stated population (e.g., 20000 -> 200 agents).
        available_actions: Override default mall actions.

    Returns:
        AgentGraph with all agents added.
    """
    if available_actions is None:
        available_actions = MALL_DEFAULT_ACTIONS

    agent_graph = AgentGraph()
    agent_id = 0

    for ring in mall_config.location_rings:
        num_agents = max(1, int(ring.population * agent_scale))
        for _ in range(num_agents):
            profile = _generate_profile_from_ring(ring, mall_config)

            budget = profile["monthly_shopping_budget"]
            user_info = UserInfo(
                user_name=f"consumer_{agent_id}",
                name=f"Consumer_{agent_id}",
                description=(
                    f"A {profile['age']}-year-old {profile['gender']} "
                    f"({profile['person_type']}, "
                    f"{profile['income_level']} income, "
                    f"MBTI: {profile['mbti']}) living "
                    f"{profile['ring_radius_km']}km from the mall. "
                    f"Monthly shopping budget: ~¥{budget}."),
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
                        "monthly_shopping_budget": budget,
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
