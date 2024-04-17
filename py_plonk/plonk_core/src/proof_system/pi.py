from ....bls12_381 import fr
from ....domain import Radix2EvaluationDomain
from ....arithmetic import INTT,from_coeff_vec
from ....arithmetic import INTT,from_coeff_vec,resize,\
                        from_gmpy_list,from_list_gmpy,from_list_tensor,from_tensor_list,domian_trans_tensor

import torch
import torch.nn as nn
# def as_evals(public_inputs,pi_pos,n):
#     pi = [fr.Fr.zero() for _ in range(n)]
#     for pos in pi_pos:
#         pi[pos] = public_inputs
#     return pi


def as_evals(public_inputs,pi_pos,n):
    pi = torch.zeros(n,4,dtype=torch.BLS12_381_Fr_G1_Mont)
    for pos in pi_pos:
        pi[pos] = public_inputs
    return pi

def into_dense_poly(public_inputs,pi_pos,n,params):
    domain = Radix2EvaluationDomain.new(n,params)
    evals_tensor = as_evals(public_inputs,pi_pos,n)
    # domian_trans_tensor(domain.group_gen_inv)
    # domian_trans_tensor(domain.size_inv)
    # domian_trans_tensor(domain.group_gen)

    inttclass = nn.Intt(domain.size, torch.BLS12_381_Fr_G1_Mont)
    pi_coeffs = inttclass.forward(evals_tensor.to('cuda'))
    pi_poly = from_coeff_vec(pi_coeffs)
    return pi_poly
