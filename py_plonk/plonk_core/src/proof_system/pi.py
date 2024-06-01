from ....bls12_381 import fr
from ....arithmetic import INTT,from_coeff_vec
import torch

def as_evals(public_inputs,pi_pos,n):
    pi = torch.zeros(n, fr.LIMBS(), dtype = fr.TYPE())
    pi[pi_pos] = public_inputs
    return pi

def into_dense_poly(public_inputs,pi_pos,n):
    evals_tensor = as_evals(public_inputs,pi_pos,n)
    pi_coeffs = INTT(n,evals_tensor.to('cuda'))
    pi_poly = from_coeff_vec(pi_coeffs)
    return pi_poly
