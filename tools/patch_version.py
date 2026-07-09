from __future__ import annotations

import importlib
import math
import warnings
from collections.abc import Callable
from typing import Any as _Any, Optional, TYPE_CHECKING

import torch
from torch import Tensor


def _set_obj_state(obj, state):
    if isinstance(state, tuple):
        if not len(state) == 2:
            raise RuntimeError(f"Invalid serialized state: {state}")
        dict_state = state[0]
        slots_state = state[1]
    else:
        dict_state = state
        slots_state = None

    # Starting with Python 3.11, the __dict__ attribute is lazily created
    # and is serialized as None when not needed.
    if dict_state:
        for k, v in dict_state.items():
            setattr(obj, k, v)

    if slots_state:
        for k, v in slots_state.items():
            setattr(obj, k, v)
    return obj

if not hasattr(torch._utils, "_set_obj_state"):
    torch._utils._set_obj_state = _set_obj_state


def _rebuild_from_type_v2(func, new_type, args, state):
    ret = func(*args)
    if type(ret) is not new_type:
        ret = ret.as_subclass(new_type)
    # Tensor does define __setstate__ even though it doesn't define
    # __getstate__. So only use __setstate__ if it is NOT the one defined
    # on Tensor
    if (
        getattr(ret.__class__, "__setstate__", Tensor.__setstate__)
        is not Tensor.__setstate__
    ):
        ret.__setstate__(state)
    else:
        ret = torch._utils._set_obj_state(ret, state)
    return ret

if not hasattr(torch._tensor, "_rebuild_from_type_v2"):
    torch._tensor._rebuild_from_type_v2 = _rebuild_from_type_v2


import torch.nn.functional as F

# 1. Grab the original function
original_interpolate = F.interpolate

# 2. Define a wrapper that "clips" the bad argument
def patched_interpolate(*args, **kwargs):
    kwargs.pop('antialias', None)  # Removes 'antialias' if present, does nothing if not
    return original_interpolate(*args, **kwargs)

# 3. Overwrite the library function with your wrapper
F.interpolate = patched_interpolate



def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention