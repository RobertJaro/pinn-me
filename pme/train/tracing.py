import torch


def cumsum_exclusive(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumulative_trapezoid(y=y, x=x, dim=-1)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumsum = torch.roll(cumsum, 1, -1)
    # Replace the first element by "1" as this is what tf.cumsum(..., exclusive=True) does.
    cumsum[..., 0] = 1.
    return cumsum
