from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            lr = group["lr"]
            correct_bias = group["correct_bias"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                # State should be stored in this dictionary
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # First moment
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Second moment
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # Update first and second moments of the gradients
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction（efficient version, Adam 論文通り）
                if correct_bias:
                    bias_correction1 = 1 - beta1 ** t
                    bias_correction2 = 1 - beta2 ** t
                    step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
                else:
                    step_size = lr

                # denom = √(v_hat) + eps
                denom = exp_avg_sq.sqrt().add_(eps)

                # Update parameters (Adam 本体)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Add weight decay after the main gradient-based updates
                if weight_decay != 0.0:
                    # lr を組み込んだ decoupled weight decay (AdamW)
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss

