# RL notebook

From SFT, RLHF to GRPO and some following RL algorithms.

## SFT

### Objective

$$
J_{SFT}(\theta) = \mathbb{E}_{x, y \sim P_{\mathrm{sft}} (X, Y)} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta (y_t | x, y_{<t}) \right]
$$

### Gradient

$$
\nabla_{\theta} J_{SFT}(\theta) = \mathbb{E}_{x, y \sim P_{\mathrm{sft}} (X, Y)} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} \log \nabla_\theta \pi_\theta (y_t | x, y_{<t}) \right]
$$

## RLHF

### Objective

$$
J_{RLHF}(\theta) = \mathbb{E}_{x \sim P_{\mathrm{sft}} (X), y \sim \pi_{\mathrm{sft}} (Y| x)} \left[ \right]
$$
