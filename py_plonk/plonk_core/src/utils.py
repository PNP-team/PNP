# Linear combination of a series of values
# For values [v_0, v_1,... v_k] returns:
# v_0 + challenge * v_1 + ... + challenge^k  * v_k
import torch.nn.functional as F

def lc(values:list, challenge):
    kth_val = values[-1]
    for val in reversed(values[:-1]):
        kth_val = F.mul_mod_scalar(kth_val, challenge)
        kth_val = F.add_mod(kth_val, val)

    return kth_val

