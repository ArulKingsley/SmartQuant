# SmartQuant
RL + LSTM Hybrid Trading Agent
# SmartQuant

RL + LSTM Hybrid Trading Agent  
A research and deployment project for a reinforcement learning-based trading bot using Deep RL, LSTM predictive models, and market simulation.

## Structure

- `agent/` – RL and meta-RL agent code
- `environment/` – Trading environment & rewards
- `models/` – Forecasting models (e.g., LSTM)
- `data/` – Market data (historical + live)
- `notebooks/` – Prototyping & research notebooks
- `deployment/` – Scripts for running/deploying
- `utils/` – Tools, configs, helpers
- `virtualization/` – Docker/env setup

## Phase Goals

```mermaid
flowchart TD
    A[Phase 1: Data Preprocessing] --> B[Phase 2: LSTM Forecasting]
    B --> C[Phase 3: RL Trading Environment]
    C --> D[Phase 4: Meta-RL + Multi-Agent System]
    D --> E[Phase 5: Backtesting & Visualization]
    E --> F[Phase 6: Deployment - Paper Trading]
    F --> G[Phase 7: Extensions - News, Portfolio, Explainability]
