from ....bls12_381 import fr
from ....arithmetic import INTT,from_coeff_vec

def as_evals(public_inputs,pi_pos,n):
    pi = [fr.Fr.zero() for _ in range(n)]
    for pos in pi_pos:
        pi[pos] = public_inputs
    return pi

def into_dense_poly(public_inputs,pi_pos,n):
    evals = as_evals(public_inputs,pi_pos,n)
    pi_coeffs = INTT(evals)
    pi_poly = from_coeff_vec(pi_coeffs)
    return pi_poly
