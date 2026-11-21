# Reinforcement Learning for Inventory Replenishment
## A Comparative Study of Q-learning and SARSA under Three Reward Designs

This project applies **tabular Reinforcement Learning (RL)** to a simplified single-product inventory replenishment problem. It compares **Q-learning** (off-policy) and **SARSA** (on-policy) under three reward designs inspired by Demand Driven Material Requirements Planning (DDMRP).

The goal is to learn when and how much to reorder while balancing product availability and inventory levels.

> **Project status and scope:** This is a reproducible simulation inspired by Cuartas and Aguilar (2023). It is not an exact reproduction of the paper or a validated real-world supply-chain deployment. The demand series is partly approximated and the training/evaluation demand is synthetic.

---

## Table of Contents
- [Problem Overview](#problem-overview)
- [Objectives](#objectives)
- [Project Workflow](#project-workflow)
- [Inventory Environment](#inventory-environment)
- [Reinforcement Learning Algorithms](#reinforcement-learning-algorithms)
- [Reward Designs](#reward-designs)
- [Experimental Setup](#experimental-setup)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Technology Stack](#technology-stack)
- [Running the Project](#running-the-project)
- [Reference](#reference)

---

## Problem Overview

Inventory replenishment is a sequential decision-making problem. Ordering too late can cause stockouts and lost sales, while ordering too early or in excessive quantities can create unnecessary inventory.

In this project, an RL agent observes the current inventory situation and selects an order action. The simulator then processes incoming deliveries and daily demand, calculates a reward, and provides the next state.

The problem is formulated as a **Markov Decision Process (MDP)**:

- **State:** Discretised on-hand stock, target stock level, and days until the next delivery.
- **Action:** Select an order quantity.
- **Transition:** Inventory changes according to deliveries, orders, and daily demand.
- **Reward:** Scores the resulting inventory position and penalises unmet demand.
- **Policy:** Maps a state to an action.

The project compares two learning algorithms across three reward functions, producing six learned policies.

## Objectives

1. Formulate inventory replenishment as an MDP.
2. Build a reproducible inventory simulation environment with lead time, minimum order quantity, and storage capacity.
3. Implement Q-learning and SARSA from scratch using NumPy.
4. Implement three reward designs inspired by the base paper.
5. Train and evaluate all six algorithm/reward combinations on shared held-out demand scenarios.
6. Compare learned policies against a simple fixed reorder-point baseline.
7. Assess policies using operational metrics rather than relying only on cumulative reward.
8. Document the assumptions and limitations of the simulator.

## Project Workflow

1. Define the P1 inventory parameters and an approximate 21-day demand sequence from the base paper's plot.
2. Generate synthetic demand scenarios for training and evaluation.
3. Build the inventory environment with discretised states, five order actions, and three reward functions.
4. Train Q-learning and SARSA using epsilon-greedy exploration.
5. Extract a greedy policy from each trained Q-table.
6. Evaluate the six policies on the same 25 held-out scenarios.
7. Compare the results with a fixed reorder-point baseline.
8. Export result tables and inspect training and evaluation behaviour.

## Inventory Environment

### Inventory parameters

| Parameter | Value |
|---|---:|
| Lead time | 7 days |
| Minimum order quantity (MOQ) | 20 units |
| Maximum inventory capacity | 100 units |
| Target inventory | 98 units |
| Top of red zone (TOR) | 49 units |
| Top of yellow zone (TOY) | 83 units |
| Top of green zone (TOG) | 122 units |
| Initial inventory | 98 units |

The target is calculated using the approximate mean demand from the P1 series:

<p align="center">
  <strong>TARGET = round(mean demand × lead time + MOQ) = 98</strong>
</p>

The target is a simplified stand-in for the optimal inventory level used in the full DDMRP calculation. The green-zone boundary is above the simulator's capacity, so inventory cannot reach it.

### Demand generation

The report uses the P1 case from the base paper as a reference. The exact daily series was not available as a downloadable table, so a 21-day sequence was read approximately from a figure.

For the experiments, synthetic demand is generated reproducibly:

- Normal-day demand: integer from 2 to 15 units.
- Peak-day probability: 10%.
- Peak-day demand: integer from 16 to 30 units.
- Training: 2,000 episodes, each containing 120 days.
- Evaluation: 25 scenarios, each containing 180 days.
- Training and evaluation use different random seeds.

The evaluation therefore measures performance on new random demand sequences from the same assumed distribution, not on a different demand pattern or real company data.

### State representation

The state is a tuple:

<p align="center">
  <strong>s = (stock bin, target bin, days to next arrival)</strong>
</p>

| State component | Definition | Range |
|---|---|---:|
| Stock bin | Rounded stock level in 10% capacity bins | 0–10 |
| Target bin | Rounded target level in 10% capacity bins | 10 |
| Days to next arrival | Days until the earliest pending delivery | 0–7 |

Since the target bin is constant, the environment has at most 88 distinct states.

### Action space

The agent selects one of five requested order quantities:

| Action index | Requested quantity | Actual quantity ordered |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 10 | 20 |
| 2 | 20 | 20 |
| 3 | 30 | 30 |
| 4 | 40 | 40 |

Any positive order is raised to the MOQ of 20 units.

### Environment dynamics

Each simulated day follows this sequence:

1. Receive orders whose arrival day is today or earlier. Inventory is capped at 100 units; excess stock is discarded.
2. Place a new order if the selected action is positive. It arrives after the seven-day lead time.
3. Serve demand from available stock.
4. Calculate the reward using the selected reward design and subtract the stockout penalty.
5. Advance to the next day.

Unmet demand is treated as lost sales and is not back-ordered.

## Reinforcement Learning Algorithms

Both algorithms maintain a Q-table containing estimated future rewards for state-action pairs.

### Q-learning

Q-learning is an **off-policy** algorithm. It updates its estimate using the maximum Q-value in the next state.

<p align="center">
  <strong>Q(s, a) ← Q(s, a) + α [r + γ max<sub>a′</sub> Q(s′, a′) − Q(s, a)]</strong>
</p>

### SARSA

SARSA is an **on-policy** algorithm. It updates its estimate using the next action actually selected by the policy.

<p align="center">
  <strong>Q(s, a) ← Q(s, a) + α [r + γ Q(s′, a′) − Q(s, a)]</strong>
</p>

Where:

- <code>α</code> is the learning rate.
- <code>γ</code> is the discount factor.
- <code>r</code> is the immediate reward.
- <code>s′</code> is the next state.
- <code>a′</code> is the next action.

Both algorithms use epsilon-greedy exploration. During training, the agent explores random actions with probability ε and otherwise selects an action with the highest estimated Q-value.

## Reward Designs

Three reward designs inspired by the base paper are implemented.

### R1 — Inventory-zone reward

The reward depends on the inventory zone:

- +1 when stock is in the yellow zone (50–83 units).
- −1 when stock is in the green zone (84–100 units).
- 0 when stock is in the red zone (0–49 units).

This reward does not directly encourage inventory to reach the target of 98 units. It penalises stock above 83 units.

### R2 — Distance-to-target reward

This reward encourages the agent to keep inventory close to the target.

<p align="center">
  <strong>R2 = 1 / (1 + |stock − TARGET|)</strong>
</p>

Adding 1 avoids division by zero. The reward is bounded between 0 and 1 before the stockout penalty.

### R3 — Zone reward plus bounded shaping

R3 combines the zone reward with an additional term that increases as stock approaches the target.

<p align="center">
  <strong>R3 = zone reward + 0.5 × (1 − min(distance / TARGET, 1))</strong>
</p>

For all three reward designs, the following stockout penalty is subtracted:

<p align="center">
  <strong>Stockout penalty = 0.10 × unmet demand units</strong>
</p>

The reward scales differ between R1, R2, and R3. Therefore, raw cumulative rewards should not be used to rank policies across different reward designs. Common operational metrics provide a fairer comparison.

## Experimental Setup

| Parameter | Value |
|---|---:|
| Learning rate (α) | 0.15 |
| Discount factor (γ) | 0.95 |
| Initial exploration rate (ε) | 1.0 |
| Minimum exploration rate | 0.05 |
| Epsilon decay per episode | × 0.997 |
| Training episodes | 2,000 |
| Days per training episode | 120 |
| Evaluation scenarios | 25 |
| Days per evaluation scenario | 180 |
| Random seed | 42 |

Six policies are trained:

| Algorithm | Reward designs |
|---|---|
| Q-learning | R1, R2, R3 |
| SARSA | R1, R2, R3 |

A fixed reorder-point baseline is also evaluated. It orders the MOQ of 20 units whenever current stock falls below 49 units. This is an illustrative baseline, not an optimised reorder policy.

## Evaluation Metrics

The learned policies are evaluated using:

- **Fill rate:** Units sold divided by units demanded. Higher is better.
- **Stockout days:** Number of days on which demand exceeds available stock. Lower is better.
- **Stockout units:** Total unmet demand. Lower is better.
- **Mean inventory:** Average ending inventory.
- **Order count:** Number of days on which an order is placed.
- **Ordered units:** Total quantity ordered.
- **Mean target distance:** Average absolute difference between ending inventory and the target of 98 units. Lower means closer to target.

All policies are tested on the same 25 evaluation scenarios, allowing paired scenario-level comparisons.

## Results

The following values are means across the 25 evaluation scenarios.

### Overall policy comparison

| Policy | Fill rate | Stockout units | Stockout days | Mean inventory | Target distance | Ordered units / demand |
|---|---:|---:|---:|---:|---:|---:|
| Fixed baseline | 89.29% | 190.44 | 21.76 | 46.52 | 51.48 | 1.03 |
| Q-learning R1 | 99.94% | 1.16 | 0.12 | 75.80 | 22.20 | 1.57 |
| SARSA R1 | 97.55% | 43.64 | 5.16 | 68.71 | 29.29 | 1.44 |
| Q-learning R2 | 97.20% | 49.80 | 6.16 | 64.15 | 33.85 | 1.45 |
| SARSA R2 | **99.99%** | **0.12** | **0.04** | 84.42 | **13.58** | 2.30 |
| Q-learning R3 | 99.80% | 3.48 | 0.48 | 74.83 | 23.17 | 1.55 |
| SARSA R3 | 99.72% | 5.08 | 0.64 | 75.28 | 22.72 | 1.60 |

### Results by reward design

| Reward design | Q-learning fill rate | SARSA fill rate | Main observation |
|---|---:|---:|---|
| R1 | 99.94% | 97.55% | Q-learning provides better service. |
| R2 | 97.20% | 99.99% | SARSA performs substantially better and stays closer to the target. |
| R3 | 99.80% | 99.72% | Both algorithms produce similar results. |

### Training behaviour

Training reward curves are smoothed using a 75-episode moving average. R1 and R3 show noisy increases during training. R2 curves remain relatively close, even though the greedy policies show a clear difference during evaluation. The curves are based on exploratory training behaviour and do not prove convergence.

## Key Findings

1. **All six learned policies outperform the simple baseline on fill rate.** The baseline reaches 89.29%, while learned policies reach between 97.20% and 99.99%.
2. **There is no single algorithm that wins under every reward design.** Q-learning performs better under R1, SARSA performs better under R2, and both are similar under R3.
3. **SARSA with R2 gives the best service result** in these experiments, with a 99.99% fill rate and an average target distance of 13.58 units.
4. **Reward design strongly affects learned behaviour.** The reward function changes what the agent is encouraged to optimise.
5. **Higher service comes with increased ordering.** Learned policies order between 1.44 and 2.30 times the total demand, compared with 1.03 times for the baseline.
6. **Raw rewards cannot be compared across reward designs** because R1, R2, and R3 use different reward scales.

The results should be interpreted within the assumptions of this simulator. In particular, the high fill rates are partly achieved through frequent ordering, and the simulator does not charge for excess inventory or discarded deliveries.

## Limitations

- The 21-day P1 demand series is approximated visually from a figure; its mean differs from the paper-reported mean.
- Training and evaluation use synthetic demand from one assumed distribution.
- The simulator does not implement the full DDMRP net-flow equation, average daily usage updates, or demand-qualification rules.
- Unmet demand is lost rather than back-ordered.
- The target inventory is a heuristic approximation of the paper's optimal inventory level.
- Holding, ordering, and stockout costs are not modelled.
- Each algorithm/reward configuration is trained only once.
- The evaluation set contains 25 scenarios from the same distribution as training.
- Reward weights and RL hyperparameters are not tuned through sensitivity analysis.
- The fixed reorder-point baseline is illustrative and not optimised.

## Future Enhancements

- Add holding, ordering, and stockout costs to evaluate total inventory cost.
- Use the exact P1 demand series and test real, intermittent, seasonal, and high-peak demand patterns.
- Implement the full DDMRP buffer calculation and net-flow position.
- Train each configuration across multiple random seeds and report confidence intervals.
- Compare against a tuned (s, S) policy and the DDMRP replenishment rule.
- Track the quantity of inventory discarded due to the capacity limit.
- Extend the state representation and explore function approximation, such as a Deep Q-Network (DQN).
- Perform sensitivity analysis on reward weights, learning rate, discount factor, and epsilon decay.
- Save Q-tables and learned policies alongside the exported evaluation results.

## Technology Stack

- **Language:** Python
- **Notebook environment:** Google Colab
- **Numerical computation:** NumPy
- **Data handling:** Pandas
- **Visualisation:** Matplotlib
- **Algorithms:** Tabular Q-learning and SARSA
- **Approach:** Markov Decision Process, epsilon-greedy exploration, DDMRP-inspired reward designs

## Running the Project

The report identifies the notebook as:

`RL_Inventory_Qlearning_vs_SARSA_Results.ipynb`

1. Open the notebook in Google Colab or a local Jupyter environment.
2. Install the required Python libraries if they are not already available:

   ```bash
   pip install numpy pandas matplotlib
   ```

3. Run the notebook cells in order.
4. The notebook generates synthetic training and evaluation demand, trains the six policies, evaluates the fixed baseline, and produces result tables and charts.

**Reproducibility note:** The notebook uses random seed 42 for its main setup and derives separate seeds for training and evaluation scenarios. The report recommends restarting the kernel and running all cells in order before saving the Q-tables, policies, and CSV results.

## Reference

Cuartas, C., & Aguilar, J. (2023). *Hybrid algorithm based on reinforcement learning for smart inventory management*. Journal of Intelligent Manufacturing, 34, 123–149.

https://doi.org/10.1007/s10845-022-01982-5
