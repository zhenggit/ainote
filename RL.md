# RL notebook

From SFT, RLHF to GRPO and some following RL algorithms.

## SFT

### Objective

```math
J_{SFT}(\theta) = \mathbb{E}_{x, y \sim P_{\mathrm{sft}} (X, Y)} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta  (y_t | x, y_{< t}) \right],
```

```math
J_{SFT}(\theta) = \mathbb{E}_{x, y \sim P_{\mathrm{sft}} (X, Y)} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta (y_t | x, y_{< t}) \right],
```
where $`x`$ is the prompt and $`y = (y_1, y_2, \cdots)`$ is the answer (supervised label). $`P_{\mathrm{sft}} (X, Y)`$ is the distribution of SFT trainning data.

### Gradient

```math
\nabla_{\theta} J_{SFT}(\theta) = \mathbb{E}_{x, y \sim P_{\mathrm{sft}} (X, Y)} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} \log \nabla_\theta \pi_\theta (y_t | x, y_{<t}) \right]
```

## RLHF

### Objective

```math
J_{RLHF}(\theta) = \mathbb{E}_{x \sim P_{\mathrm{sft}} (X), y \sim \pi_{\mathrm{old}} (Y| x)} \left[ r_\phi (x, y) \right]- \beta \mathrm{KL} ( \pi_\theta \| \pi_\mathrm{old} ),
```
where
```math
\mathrm{KL}(\pi_\theta \| \pi_{\mathrm{old}}) = \mathbb{E}_{x \sim P_{\mathrm{sft}} (X), y \sim \pi_{\mathrm{old}} (Y| x)} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} \log \frac{\pi_\theta (y_t | x, y_{<t})}{\pi_\mathrm{old} (y_t | x, y_{<t})}\right]
```
and $`r_\phi(x,y)`$ is learned value function with parameter $`\phi`$.

## PPO

### Objective

```math
J_{\mathrm{PPO}} (\theta) = 
```

### Gradient

## DPO

### Objective

### Gradient

## RFT (Rejection sampling Fine Tunning)

## GRPO

## DAPO
