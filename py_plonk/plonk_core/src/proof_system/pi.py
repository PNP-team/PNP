from ....bls12_381 import fr
import torch

def as_evals(public_inputs,pi_pos,n):
    pi = torch.zeros(n, fr.LIMBS(), dtype = fr.TYPE())
    pi[pi_pos] = public_inputs
    return pi

def into_dense_poly(public_inputs, pi_pos, n, INTT):
    evals_tensor = as_evals(public_inputs, pi_pos, n)
    pi_coeffs = INTT(evals_tensor.to('cuda'))
    return pi_coeffs
