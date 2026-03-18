# OASIS Codebase Research Report

## 1. What Is OASIS?

**OASIS (Open Agent Social Interaction Simulations with One Million Agents)** is a scalable, open-source social media simulator built by [CAMEL-AI.org](https://www.camel-ai.org/). It uses LLM-powered agents to realistically mimic user behavior on platforms like **Twitter** and **Reddit** at massive scale (up to 1 million agents).

- **Paper**: [arXiv:2411.11581](https://arxiv.org/abs/2411.11581)
- **Package**: `camel-oasis` on PyPI (v0.2.5)
- **License**: Apache 2.0
- **Python**: 3.10–3.11

It is designed for research into complex social phenomena: information spread, group polarization, herd behavior, misinformation propagation, and more.

---

## 2. Project Structure

```
oasis/
├── oasis/                           # Core library (~6,000 LOC)
│   ├── environment/                 # Simulation environment (env, actions, factory)
│   ├── social_agent/                # Agent logic, actions, graph, generation
│   ├── social_platform/             # Platform engine, DB, recsys, channel, config
│   ├── clock/                       # Simulated time management
│   └── testing/                     # DB inspection utilities
├── examples/                        # 25 example scripts
│   ├── quick_start.py               # Minimal working example
│   ├── twitter_simulation_openai.py
│   ├── reddit_simulation_openai.py
│   ├── group_chat_simulation.py
│   ├── twitter_interview.py
│   ├── twitter_misinforeport.py
│   └── experiment/                  # Research experiments
│       ├── twitter_simulation/      # Alignment, polarization
│       ├── twitter_simulation_1M_agents/
│       ├── reddit_simulation_align_with_human/
│       ├── reddit_simulation_counterfactual/
│       └── reddit_emall_demo/       # E-commerce simulation
├── data/                            # Agent profile datasets
│   ├── twitter/                     # Twitter propagation data (depth/breadth/scale variants)
│   └── reddit/                      # Reddit user profiles (user_data_36.json, etc.)
├── test/                            # pytest test suite (30+ modules)
│   ├── agent/                       # Agent-level tests
│   └── infra/database/              # 20+ DB operation tests
├── docs/                            # Mintlify documentation site
├── generator/                       # Dataset generation scripts
├── visualization/                   # Visualization utilities
├── deploy.py                        # vLLM multi-GPU deployment script
├── pyproject.toml                   # Poetry config
└── .github/workflows/               # CI/CD pipeline
```

---

## 3. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10–3.11 |
| Multi-agent framework | CAMEL-AI 0.2.78 |
| Database | SQLite3 (primary), Neo4j 5.23.0 (optional graph DB) |
| LLM models | OpenAI GPT-4o-mini (default), vLLM-compatible (Llama-3, etc.) |
| Embeddings | Sentence Transformers (Twhin-Bert, all-MiniLM-L6-v2), OpenAI Embeddings |
| ML/NLP | scikit-learn (TF-IDF, cosine similarity), PyTorch |
| Graph | igraph 0.11.6 (in-memory), Neo4j (distributed) |
| Visualization | cairocffi 1.7.1, igraph layout algorithms |
| Package management | Poetry |
| CI/CD | GitHub Actions |
| Code quality | pre-commit (Ruff, isort, Flake8, Mdformat) |
| Testing | pytest 8.2.0, pytest-asyncio 0.23.6 |

---

## 4. Architecture Deep Dive

### 4.1 High-Level Flow

```
User Code
    │
    ▼
OasisEnv.make()  ──→  OasisEnv
    │
    ├── env.reset()   ──→  Platform.running() (async task)
    │                       + Agent sign-up via Channel
    │
    ├── env.step()    ──→  Update rec table
    │                       + Execute actions (Manual or LLM) concurrently
    │
    └── env.close()   ──→  Send EXIT via Channel, await platform task
```

### 4.2 Core Components

#### OasisEnv (`oasis/environment/env.py`)
The top-level simulation controller following a **Gymnasium/PettingZoo-style interface**:
- `make(agent_graph, platform, database_path)` — factory function
- `reset()` — starts the platform event loop as an async task, signs up all agents
- `step(actions)` — updates recommendation table, then executes all actions concurrently via `asyncio.gather()`
- `close()` — sends EXIT signal, awaits platform task completion

Key design choices:
- **Semaphore-based throttling** (default 128) to limit concurrent LLM API calls
- Actions can be **ManualAction** (pre-scripted) or **LLMAction** (agent decides autonomously)
- Platform type determines defaults: Twitter uses Twhin-Bert recsys; Reddit uses TF-IDF recsys

#### SocialAgent (`oasis/social_agent/agent.py`)
Extends CAMEL's `ChatAgent` class. Each agent has:
- **UserInfo** — profile, personality, description → generates a system prompt
- **SocialAction** — the set of tool functions the agent can call
- **SocialEnvironment** — provides context (recommended posts, followers, follows) before each LLM call
- **Available actions** — configurable subset of all 29 possible actions
- **Custom model/tools/prompts** — fully customizable per agent

The LLM action flow:
1. Agent's `SocialEnvironment` fetches current posts/followers from platform
2. Environment info is formatted as a user message
3. CAMEL's `ChatAgent.step()` calls the LLM with tool-use
4. LLM selects an action (tool call), which writes to the Channel
5. Platform processes the action and writes the result back

#### SocialAction (`oasis/social_agent/agent_action.py`, 758 LOC)
Defines **29 action methods**, each following the same pattern:
1. Package action arguments into a message
2. Write to `Channel.receive_queue` with a UUID
3. Await response from `Channel.send_dict`

Actions: `create_post`, `like_post`, `repost`, `quote_post`, `unlike_post`, `dislike_post`, `undo_dislike_post`, `report_post`, `follow`, `unfollow`, `mute`, `unmute`, `search_posts`, `search_user`, `trend`, `refresh`, `do_nothing`, `create_comment`, `like_comment`, `unlike_comment`, `dislike_comment`, `undo_dislike_comment`, `purchase_product`, `interview`, `join_group`, `leave_group`, `send_to_group`, `create_group`, `listen_from_group`

#### Channel (`oasis/social_platform/channel.py`)
The **message bus** between agents and the platform:
- `AsyncSafeDict` — lock-protected dictionary for thread-safe concurrent access
- `receive_queue` — asyncio.Queue for incoming agent actions
- `send_dict` — UUID-keyed responses from platform
- Agents write to queue → Platform reads from queue → Platform writes response → Agent reads response
- Polling with 0.1s sleep to avoid tight loops

#### Platform (`oasis/social_platform/platform.py`, 1,642 LOC — the largest file)
The central simulation engine running as a continuous async event loop:
- **`running()`** — main loop that reads from Channel and dispatches to action handlers
- Manages the **SQLite database** with 16 tables
- Handles all CRUD operations (sign_up, create_post, like, follow, search, etc.)
- Implements **content filtering** (muted users, self-rating rules)
- Updates **recommendation tables** before each step
- Tracks all actions in a **trace table** for post-hoc analysis

Platform configuration options:
- `show_score` — Reddit-style (likes - dislikes) vs separate counts
- `allow_self_rating` — whether users can like/dislike own posts
- `refresh_rec_post_count` — posts returned per refresh
- `max_rec_post_len` — maximum recommended posts per user
- `following_post_count` — posts from followed users per refresh
- `use_openai_embedding` — toggle between local and API embeddings

#### AgentGraph (`oasis/social_agent/agent_graph.py`)
Manages the social network topology:
- **igraph backend** — in-memory directed graph for small/medium simulations
- **Neo4j backend** — distributed graph for large-scale (1M agent) scenarios
- Operations: `add_agent`, `add_edge`, `remove_edge`, `get_agents`, `get_edges`
- Visualization via igraph layout algorithms
- Serializable for saving/loading experiment states

#### Clock (`oasis/clock/clock.py`)
Simulated time management:
- `k` factor (default 60) — magnification of real time
- `time_transfer(start_time)` — converts real elapsed time to simulated time
- `time_step` — discrete step counter incremented each `env.step()`

### 4.3 Recommendation Systems (`oasis/social_platform/recsys.py`)

Four algorithms available:

| Algorithm | Use Case | Method |
|---|---|---|
| `reddit` | Reddit platform | TF-IDF vectorization + cosine similarity + hot score (recency + engagement) |
| `twhin-bert` | Twitter platform | Twitter's Twhin-BERT embeddings + follower graph + engagement metrics |
| `random` | Baseline | Random post selection |
| `twitter` | Legacy | Basic content-based filtering |

The Reddit recsys combines:
- **Content similarity** — TF-IDF vectors of post text vs. user interest profile
- **Hot score** — `log10(max(|score|, 1)) + (created_seconds - reference_epoch) / 45000`
- **Temporal decay** — recent posts scored higher
- **Engagement signals** — likes, dislikes, comments

The Twhin-BERT recsys adds:
- **Embedding similarity** — sentence-transformer embeddings of posts and user profiles
- **Social graph signals** — posts from followed users weighted higher
- **Popularity weighting** — follower count and engagement metrics

---

## 5. Database Schema (16 tables)

Located in `oasis/social_platform/schema/`:

| Table | Purpose | Key Fields |
|---|---|---|
| `user` | User accounts | user_id, agent_id, user_name, name, bio, created_at, num_followings, num_followers |
| `post` | Posts/tweets | post_id, user_id, original_post_id, content, quote_content, created_at, num_likes, num_dislikes, num_shares, num_reports |
| `follow` | Follow relationships | user_id, followee_id, created_at |
| `mute` | Muted users | user_id, mutee_id, created_at |
| `like` | Post likes | user_id, post_id, created_at |
| `dislike` | Post dislikes | user_id, post_id, created_at |
| `report` | Post reports | user_id, post_id, reason, created_at |
| `trace` | Action audit log | agent_id, action, timestamp, details |
| `rec` | Recommendation cache | user_id, post_id_list |
| `comment` | Comments | comment_id, post_id, user_id, content, created_at, num_likes, num_dislikes |
| `comment_like` | Comment likes | user_id, comment_id, created_at |
| `comment_dislike` | Comment dislikes | user_id, comment_id, created_at |
| `product` | E-mall products | product_id, name, description |
| `chat_group` | Group chats | group_id, group_name, created_at |
| `group_member` | Group membership | group_id, user_id |
| `group_message` | Group messages | message_id, group_id, user_id, content, created_at |

The `trace` table is particularly important — it stores every action performed by every agent, enabling post-hoc analysis of social dynamics.

---

## 6. Agent Profiles and Generation

### Profile Data Format
Agent profiles are loaded from JSON/CSV files in `data/`:

**Reddit profiles** (`user_data_36.json`): Include name, description, and `other_info` with gender, age, MBTI, country, and `user_profile` text.

**Twitter profiles**: Include propagation data (who retweets whom) with depth/breadth/scale variants for different experiment designs.

### Generation Pipeline (`oasis/social_agent/agents_generator.py`)

Three main generators:
1. **`generate_reddit_agent_graph()`** — loads Reddit JSON profiles, creates agents with Reddit system prompts
2. **`generate_twitter_agent_graph()`** — loads Twitter CSV profiles, establishes follow relationships from propagation data
3. **`generate_agents_100w()`** — large-scale (1M) agent generation using Neo4j backend for relationship storage

The `generate_custom_agents()` function handles the sign-up process: iterates through all agents in the graph and registers them with the platform via the Channel.

---

## 7. Platform-Specific Behaviors

### Twitter Mode
- Recommendation: Twhin-BERT embeddings
- Actions: create_post, like, repost, quote_post, follow, do_nothing
- Scores shown separately (likes count, repost count)
- Clock advances by timestep

### Reddit Mode
- Recommendation: TF-IDF + hot score
- Actions: full action set including dislike, search, trend
- Score = likes - dislikes (shown as single number via `show_score=True`)
- Self-rating allowed by default (`allow_self_rating=True`)
- Richer profile system (gender, age, MBTI, country)

### E-Mall Extension
- Product table and purchase_product action
- Simulates e-commerce behavior on Reddit-like platform
- Used in `reddit_emall_demo` experiment

---

## 8. Key Design Patterns

| Pattern | Where | Why |
|---|---|---|
| **Actor Model** | Agent ↔ Channel ↔ Platform | Decouples agents from platform, enables concurrent execution |
| **Async/Await** | Entire codebase | Non-blocking I/O for 1M agent scalability |
| **Factory** | `oasis.make()` | Platform-agnostic environment creation |
| **Strategy** | RecsysType enum | Pluggable recommendation algorithms |
| **PettingZoo-style API** | `reset()`, `step()`, `close()` | Familiar RL environment interface |
| **Message Queue** | Channel with UUID tracking | Reliable async communication with response correlation |
| **Semaphore** | `llm_semaphore` (default 128) | Rate-limiting concurrent LLM API calls |

---

## 9. Example Experiments

The `examples/experiment/` directory contains research-grade simulations:

| Experiment | Description |
|---|---|
| `twitter_simulation/align_with_real_world/` | Validates simulation against real Twitter data |
| `twitter_simulation/group_polarization/` | Studies opinion polarization dynamics |
| `twitter_simulation_1M_agents/` | Million-agent scalability demonstration |
| `reddit_simulation_align_with_human/` | Validates Reddit agent behavior against human data |
| `reddit_simulation_counterfactual/` | Counterfactual experiments on Reddit |
| `reddit_emall_demo/` | E-commerce behavior simulation |
| `twitter_gpt_example/` | Standard Twitter simulation with GPT |
| `twitter_gpt_example_openai_embedding/` | Twitter with OpenAI embeddings |

---

## 10. Testing Infrastructure

- **Framework**: pytest + pytest-asyncio
- **30+ test modules** covering database operations, agent actions, platform flows
- **MockChannel** pattern for isolating platform tests
- **CI/CD**: GitHub Actions on push/PR to main — runs pre-commit + full test suite
- **Secrets**: OPENAI_API_KEY and OPENAI_BASE_URL from GitHub secrets for LLM-dependent tests

Test organization:
```
test/
├── agent/            # Agent-level integration tests
└── infra/
    └── database/     # Fine-grained DB operation tests
        ├── test_signup.py
        ├── test_create_post.py
        ├── test_like_post.py
        ├── test_follow.py
        ├── test_comment.py
        ├── test_search.py
        ├── test_trend.py
        ├── test_group_chat.py
        └── ... (20+ files)
```

---

## 11. Deployment

### Local Development
```bash
pip install camel-oasis
export OPENAI_API_KEY=<key>
python examples/quick_start.py
```

### Large-Scale (vLLM)
`deploy.py` launches multiple vLLM API servers across GPUs for distributed LLM inference, enabling the 1M agent experiments.

### Environment Variables
- `OPENAI_API_KEY` — Required for OpenAI models
- `OPENAI_BASE_URL` — Custom API endpoint (e.g., vLLM)
- `OASIS_DB_PATH` — Override database path

---

## 12. Notable Specificities and Quirks

1. **PRAGMA synchronous = OFF** — Platform disables SQLite synchronous writes for performance (`platform.py:84`). This trades durability for speed, which is acceptable since simulation data can be regenerated.

2. **Agent ID = User ID - 1** — The codebase assumes sequential agent registration where `user_id = agent_id + 1`. This is noted in `recsys.py:56-57` and breaking this invariant causes recommendation system errors.

3. **Channel polling** — `read_from_send_queue` uses a `0.1s sleep` polling loop (`channel.py:70-71`). This is a deliberate trade-off to reduce CPU load, but introduces latency.

4. **Global mutable state in recsys** — `recsys.py` uses module-level global dictionaries (`user_previous_post_all`, `user_previous_post`, `t_items`, `u_items`, etc.) for caching. This means **only one simulation can run per process**.

5. **Lazy model loading** — Twhin-BERT tokenizer and model are loaded on first use (`get_twhin_tokenizer()`, `get_twhin_model()`), avoiding unnecessary GPU memory allocation for Reddit-only experiments.

6. **Debug artifacts** — `platform.py:72` contains a commented-out `import pdb; pdb.set_trace()` line.

7. **Sphinx guard** — Logger initialization is wrapped in `if "sphinx" not in sys.modules` checks to prevent file handler creation during documentation builds.

8. **Clock magnification** — The simulated clock runs at 60x real time by default. This is used for Twitter mode to give posts temporal context, but Reddit mode primarily uses the discrete `time_step` counter.

9. **Self-contained DB creation** — The database schema is split across 16 separate `.sql` files in `schema/`, all executed at platform startup. The DB file is typically deleted and recreated for each experiment run.

10. **Follow-up research ecosystem** — OASIS has spawned derivative projects: MultiAgent4Collusion (collusion simulation), CUBE (Unity3D environments), and MultiAgent4Fraud (financial fraud).

---

## 13. Data Flow Summary

```
┌─────────────┐     ManualAction/LLMAction     ┌──────────────┐
│  User Code   │ ──────────────────────────────→│   OasisEnv   │
└─────────────┘                                 └──────┬───────┘
                                                       │
                                        ┌──────────────┼──────────────┐
                                        │              │              │
                                        ▼              ▼              ▼
                                   ┌─────────┐  ┌─────────┐   ┌─────────┐
                                   │ Agent 0  │  │ Agent 1  │   │ Agent N │
                                   └────┬─────┘  └────┬─────┘   └────┬────┘
                                        │              │              │
                                   LLM call or    LLM call or   LLM call or
                                   manual action  manual action  manual action
                                        │              │              │
                                        ▼              ▼              ▼
                                   ┌──────────────────────────────────────┐
                                   │           Channel (async queue)       │
                                   │    receive_queue ←→ send_dict        │
                                   └──────────────────┬───────────────────┘
                                                      │
                                                      ▼
                                   ┌──────────────────────────────────────┐
                                   │         Platform (event loop)         │
                                   │                                      │
                                   │  ┌────────────┐  ┌───────────────┐  │
                                   │  │  SQLite DB  │  │  Recsys Engine│  │
                                   │  │  (16 tables)│  │  (4 algos)   │  │
                                   │  └────────────┘  └───────────────┘  │
                                   │                                      │
                                   │  ┌────────────┐  ┌───────────────┐  │
                                   │  │    Clock    │  │ Trace Logger  │  │
                                   │  └────────────┘  └───────────────┘  │
                                   └──────────────────────────────────────┘
```

---

## 14. Conclusion

OASIS is a well-architected multi-agent social simulation platform with clear separation of concerns across four layers: **environment** (orchestration), **agent** (LLM-powered decision-making), **platform** (social media engine), and **data** (SQLite + graph DB). Its async-first design with Channel-based message passing enables scaling to 1M agents, while the pluggable recommendation system and dual-platform support (Twitter/Reddit) make it versatile for diverse social science research.

The codebase is approximately 6,000 lines of core library code, well-tested with 30+ test modules, and comes with 25 example scripts covering everything from minimal quickstarts to million-agent experiments. It represents a research-grade tool that balances simulation fidelity with computational scalability.
