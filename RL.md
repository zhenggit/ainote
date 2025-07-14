# RL notebook

From SFT, PPO to GRPO and some following RL algorithms.

## SFT

### Objective

```math
J_{SFT}(\theta) = \mathbb{E}_{x, y \sim P_{\mathrm{sft}} (X, Y)} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta (y_t | x, y_{< t}) \right],
```
where $`x`$ is the prompt and $`y = (y_1, y_2, \cdots)`$ is the answer. $`P_{\mathrm{sft}} (X, Y)`$ is the distribution of SFT trainning data. This is the cross entropy for the next token prediction.

### Gradient

```math
\nabla_{\theta} J_{SFT}(\theta) = \mathbb{E}_{x, y \sim P_{\mathrm{sft}} (X, Y)} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} \log \nabla_\theta \pi_\theta (y_t | x, y_{< t}) \right]
```

## PPO

### Objective

```math
J_{\mathrm{PPO}} (\theta) = \mathbb{E}_{x \sim P_{\mathrm{sft}} (X), y \sim \pi_{\mathrm{old} (Y | x)}} \left[
  \frac{1}{|y|} \sum_{t=1}^{|y|} \min \left\{
    A_t \cdot \frac{\pi_\theta (y_t | x, y_{< t})}{\pi_{\mathrm{old}} (y_t | x, y_{< t})},
    A_t \mathrm{clip} \left(
       \frac{\pi_\theta (y_t | x, y_{< t})}{\pi_{\mathrm{old}} (y_t | x, y_{< t})}, 1 - \epsilon, 1 + \epsilon
    \right)
  \right\}
\right]
```

where the advantage (from GAE alg.)

```math
A_t = \delta_t + \gamma \lambda \delta_{t+1} + \cdots + (\gamma \lambda)^{T-t+1} \delta_{T-1},
```
with $`\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi (s_t)`$ and $`V_\phi (s)`$ is the value function trained alongside. Here, $`r_t`$ is the per-token reward model

```math
r_t = r (x, y_{\le t})- \beta \log \frac{\pi_\theta (y_t | x, y_{< t})}{\pi_\mathrm{ref} (y_t | x, y_{< t})}
```
where $`r(x,y_{\le t})`$ is a (raw) reward model.


### Gradient

Assume $`\pi_\theta = \pi_\mathrm{old}`$.
```math
\nabla_\theta J_{\mathrm{PPO}} (\theta) = \mathbb{E}_{x \sim P_{\mathrm{sft}} (X), y \sim \pi_{\mathrm{old} (Y | x)}} \left[
  \frac{1}{|y|} \sum_{t=1}^{|y|} 
    A_t \nabla_\theta \log \pi_\theta (y_t | x, y_{< t}).
\right]
```
* THIS IS FROM GRPO PAPER, WHY???

## DPO

### Objective

### Gradient

## RFT (Rejection sampling Fine Tunning)

## GRPO

## DAPO
