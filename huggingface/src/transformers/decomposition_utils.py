import torch, math


def _apply_decomposed_ln(dec_input, orig_input, ln_module):
    """
    Applies LayerNorm to target-aligned decomposed input by centering the input,
    but dividing by the original undecomposed variance.
    This function does not add the learned bias of LayerNorm.
    """
    ln_weight = ln_module.weight
    ln_eps = ln_module.eps
    dec_means = torch.mean(dec_input, dim=-1, keepdim=True)
    ln_std = (torch.var(orig_input, dim=-1, unbiased=False, keepdim=True) + ln_eps) ** (1/2)  # seq_len, 1

    # subtract means
    dec_output = dec_input - dec_means
    dec_output = torch.div(dec_output, ln_std)
    dec_output = dec_output * ln_weight

    return dec_output


def _align_source(dec_input):
    """
    Takes a tensor of decomposed representations where columns are target positions (useful for LN, FF) and
    returns a tensor of decomposed representations where columns are source positions (useful for ATTN).
    Rolls representations from right to left:
    [1, 2, 3]    [1, 2, 3]
    [0, 4, 5] -> [4, 5, 0]
    [0, 0, 6]    [6, 0, 0]
    """
    for i in range(1, dec_input.size(0)):
        dec_input[i] = torch.roll(dec_input[i], shifts=-i, dims=0)

    return dec_input


def _align_target(dec_input):
    """
    Takes a tensor of decomposed representations where columns are source positions (useful for ATTN) and
    returns a tensor of decomposed representations where columns are target positions (useful for LN, FF).
    Rolls representations from left to right:
    [1, 2, 3]    [1, 2, 3]
    [4, 5, 0] -> [0, 4, 5]
    [6, 0, 0]    [0, 0, 6]
    """
    for i in range(1, dec_input.size(0)):
        dec_input[i] = torch.roll(dec_input[i], shifts=i, dims=0)

    return dec_input
