import torch

EXP_PARAM_SCALE = 10.0
KERNEL_BETA_UPBOUND = 5.0
VALUE_BETA_INIT = -0.5


def lag_coefficients(block_size: int) -> torch.Tensor:
    """Матрица лагов: элемент на k шагов ниже диагонали равен -k."""
    coef = torch.zeros(block_size, block_size)
    for lag in range(block_size):
        coef = torch.diagonal_scatter(
            coef, -torch.ones(block_size - lag) * lag, -lag
        )
    return coef


def future_mask(block_size: int) -> torch.Tensor:
    """True на позициях строго в будущем. Диагональ остаётся видимой."""
    eye = torch.tril(torch.ones(block_size, block_size))
    return eye.view(1, 1, block_size, block_size) == 0


def attention_mask(block_size: int, v_shift: int, att_shift: int = 0) -> torch.Tensor:
    """True там, куда запросу смотреть нельзя: на себя и в будущее.

    Срез со столбца v_shift повторяет маску Library: значение заглядывает
    на v_shift токенов вперёд, а внимание этот сдвиг компенсирует.
    """
    mask_size = block_size + v_shift + att_shift
    causal = torch.tril(torch.ones(mask_size, mask_size))
    visible = causal[:block_size, v_shift + att_shift :]
    return visible.view(1, 1, block_size, block_size) == 0


def apply_leaky_keys(
    keys: torch.Tensor,
    leakyavg_coef: torch.Tensor,
    leaky_attn_mask: torch.Tensor,
    leaky_key_beta: torch.Tensor,
    kernel_beta: torch.Tensor,
    *,
    scale_power: int,
) -> torch.Tensor:
    """Геометрическое среднее ключей по прошлому, затем L2 и масштаб exp(β).

    scale_power=2 используется persistent memory: один и тот же β задаёт
    и ширину ядра слотов, поэтому масштаб возводится в квадрат.
    """
    batch, n_head, length, head_dim = keys.shape
    beta = leaky_key_beta.abs() * EXP_PARAM_SCALE
    window = leakyavg_coef[:length, :length].view(1, 1, length, length)
    window = window.expand(1, n_head, length, length)
    beta_view = beta.view(1, n_head, 1, 1).expand(1, n_head, length, length)
    coef = torch.exp(window * beta_view)
    coef = coef * (1 - torch.exp(-beta)).view(1, n_head, 1, 1).expand(
        1, n_head, length, length
    )
    coef = coef.masked_fill(leaky_attn_mask[:, :, :length, :length], 0)
    mixed = coef.expand(batch, n_head, length, length) @ keys
    mixed = mixed / mixed.norm(dim=-1, keepdim=True)
    scale = torch.exp(
        torch.clamp(kernel_beta * EXP_PARAM_SCALE, max=KERNEL_BETA_UPBOUND)
    )
    if scale_power != 1:
        scale = scale**scale_power
    return mixed * scale.view(1, n_head, 1, 1).expand(batch, n_head, length, head_dim)


def value_scale(value_beta: torch.Tensor, n_head: int, length: int, head_dim: int) -> torch.Tensor:
    """Множитель exp(βᵥ) формы (1, головы, 1, 1), растянутый на батч позже."""
    return torch.exp(value_beta * EXP_PARAM_SCALE).view(1, n_head, 1, 1).expand(
        1, n_head, length, head_dim
    )
