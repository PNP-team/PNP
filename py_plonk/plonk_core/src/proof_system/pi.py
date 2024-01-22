from ....bls12_381 import fr
from ....domain import Radix2EvaluationDomain
from ....arithmetic import INTT,from_coeff_vec
from ....arithmetic import INTT,from_coeff_vec,resize,\
                        from_gmpy_list,from_list_gmpy,from_list_tensor,from_tensor_list


def as_evals(public_inputs,pi_pos,n):
    pi = [fr.Fr.zero() for _ in range(n)]
    for pos in pi_pos:
        pi[pos] = public_inputs
    return pi

def into_dense_poly(public_inputs,pi_pos,n,params):
    domain = Radix2EvaluationDomain.new(n,params)
    evals = as_evals(public_inputs,pi_pos,n)
    
    from_gmpy_list(evals)
    evals_tensor=from_list_tensor(evals)
    pi_coeffs = INTT(domain,evals_tensor)
    pi_poly = from_coeff_vec(pi_coeffs)
    return pi_poly
