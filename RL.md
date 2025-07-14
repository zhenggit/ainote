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
J_{\mathrm{PPO}} (\theta) = \mathbb{E}_{x \sim P_{\mathrm{sft}}, y \sim \pi_{\mathrm{old} (Y | x)}} \left[
  \frac{1}{|y|} \sum_{t=1}^{|y|} \min \left\{
    A_t \cdot \frac{\pi_\theta (y_t | x, y_{< t})}{\pi_{\mathrm{old}} (y_t | x, y_{< t})},
    A_t \mathrm{clip} \left(
       \frac{\pi_\theta (y_t | x, y_{< t})}{\pi_{\mathrm{old}} (y_t | x, y_{< t})}, 1 - \epsilon, 1 + \epsilon
    \right)
  \right\}
\right]
```



reward model(per-token)

```math
r_t = r_\phi (x, y_{\le t})- \beta \log \frac{\pi_\theta (y_t | x, y_{< t})}{\pi_\mathrm{ref} (y_t | x, y_{< t})}
```
where $`r_\phi(x,y_{\le t})`$ is a (raw) reward model with parameter $`\phi`$.

GAE: Advantage $`A_t = GAE(r_{\ge t}, V_{\varphi})`$ where $`V_{\varphi}`$ is a value function trained alongside the policy model.

### Gradient

## DPO

### Objective

### Gradient

## RFT (Rejection sampling Fine Tunning)

## GRPO

## DAPO
