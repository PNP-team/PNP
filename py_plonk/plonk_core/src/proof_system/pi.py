from ....bls12_381 import fr
from ....domain import Radix2EvaluationDomain
from ....arithmetic import INTT_new,from_coeff_vec

import torch
import torch.nn as nn

def as_evals(public_inputs,pi_pos,n):
    pi = torch.zeros(n,4,dtype=torch.BLS12_381_Fr_G1_Mont)
    for pos in pi_pos:
        pi[pos] = public_inputs
    return pi

def into_dense_poly(public_inputs,pi_pos,n,params):
    domain = Radix2EvaluationDomain.new(n,params)
    evals_tensor = as_evals(public_inputs,pi_pos,n)
    pi_coeffs = INTT_new(domain,evals_tensor.to('cuda'))
    pi_poly = from_coeff_vec(pi_coeffs)
    return pi_poly
