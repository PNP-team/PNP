from ....bls12_381 import fr
import torch

def into_dense_poly(public_inputs, pi_pos, n, INTT):
    evals_tensor = torch.zeros(n, fr.LIMBS(), dtype = fr.TYPE())
    evals_tensor[pi_pos] = public_inputs
    pi_coeffs = INTT(evals_tensor.to('cuda'))
    return pi_coeffs
