# OASIS Mall Investment Simulation

Shopping-mall investment analysis powered by [OASIS](https://github.com/camel-ai/oasis) agent simulation.

Consumer agents with demographic profiles react to a configured mall (building, tenants, transport, location) on simulated social media. Output: investment-grade analytics comparing tenant-mix scenarios.

## Quick Start

```bash
# Install
pip install camel-oasis

# Set API key
export OPENAI_API_KEY=<your-key>

# Run the simulation
python examples/mall_investment_analysis.py
```

## Key Components

| File | Purpose |
|------|---------|
| `oasis/social_platform/config/mall.py` | Mall configuration (building, tenants, transport, location) |
| `oasis/social_agent/mall_agents_generator.py` | Generate consumer agents from demographic rings |
| `oasis/social_platform/mall_analytics.py` | Analytics: foot traffic, spending, tenant revenue |
| `examples/mall_simulation.py` | Basic mall simulation example |
| `examples/mall_investment_analysis.py` | Monte Carlo investment analysis with scenario comparison |

## How It Works

1. **Configure** a mall: floors, tenant mix, transport links, surrounding population
2. **Generate** consumer agents with demographics (age, income, preferences) from location rings
3. **Simulate** social media interactions — agents discuss, recommend, and react to the mall
4. **Analyze** results: foot traffic, spending patterns, rental income delta per scenario
5. **Compare** tenant-mix fixes with confidence scores for investment decisions

## Upstream

Built on [OASIS](https://github.com/camel-ai/oasis) — Open Agent Social Interaction Simulations with One Million Agents ([paper](https://arxiv.org/abs/2411.11581)).

## License

Apache 2.0
