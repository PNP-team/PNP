from ....plonk_core.src.permutation import constants
from ....arithmetic import calculate_execution_time
from ....bls12_381 import fr
import torch
import torch.nn as nn
import torch.nn.functional as F

def _numerator_irreducible(root, w, k, beta, gamma):
    mid1 = F.mul_mod(beta,k)
    mid2 = F.mul_mod(mid1,root)
    mid3 = F.add_mod(w,mid2)
    numerator = F.add_mod(mid3,gamma)
    return numerator

def _denominator_irreducible(w, sigma, beta, gamma):
    mid1 = F.mul_mod_scalar(sigma, beta)
    mid2 = F.add_mod(w, mid1)
    denominator = F.add_mod(mid2, gamma)
    return denominator

def _lookup_ratio(one ,delta, epsilon, f, t, t_next,
                h_1, h_1_next, h_2):
   
    one_plus_delta = F.add_mod(delta,one)
    epsilon_one_plus_delta = F.mul_mod(epsilon,one_plus_delta)
    
    mid1 = F.add_mod(epsilon,f)
    mid2 = F.add_mod(epsilon_one_plus_delta,t)
    mid3 = F.mul_mod(delta,t_next)
    mid4 = F.add_mod(mid2,mid3)
    mid5 = F.mul_mod(one_plus_delta,mid1)
    result = F.mul_mod(mid4,mid5)

    mid6 = F.mul_mod(h_2,delta)
    mid7 = F.add_mod(epsilon_one_plus_delta,h_1)
    mid8 = F.add_mod(mid6,mid7)
    mid9 = F.add_mod(epsilon_one_plus_delta,h_2)
    mid10= F.mul_mod(h_1_next,delta)
    mid11 = F.add_mod(mid9,mid10)
    mid12 = F.mul_mod(mid8,mid11)
    mid12 = F.div_mod(one,mid12)
    result= F.mul_mod(result,mid12)

    return result

@calculate_execution_time
def compute_permutation_poly(domain, wires, beta, gamma, sigma_polys: torch.Tensor):
    n = domain.size
    one = fr.one().to("cuda")
    # Constants defining cosets H, k1H, k2H, etc
    ks = [[],[],[],[]]
    ks[0] = fr.one().to("cuda")
    ks[1] = constants.K1().to("cuda")
    ks[2] = constants.K2().to("cuda")
    ks[3] = constants.K3().to("cuda")

    NTT = nn.Ntt(fr.TWO_ADICITY(), fr.TYPE())
    sigma_mappings = [[],[],[],[]]
    sigma_mappings[0] = NTT(sigma_polys[0].to("cuda"))
    sigma_mappings[1] = NTT(sigma_polys[1].to("cuda"))
    sigma_mappings[2] = NTT(sigma_polys[2].to("cuda"))
    sigma_mappings[3] = NTT(sigma_polys[3].to("cuda"))


    # Transpose wires and sigma values to get "rows" in the form [wl_i,
    # wr_i, wo_i, ... ] where each row contains the wire and sigma
    # values for a single gate
    # Compute all roots, same as calculating twiddles, but doubled in size
    roots = F.gen_sequence(n, domain.group_gen.to("cuda"))

    numerator_product = F.repeat_to_poly(one, n)
    denominator_product = F.repeat_to_poly(one, n)
    numerator_product = numerator_product.to("cuda")
    denominator_product = denominator_product.to("cuda")

    beta = beta.to("cuda")
    gamma = gamma.to("cuda")

    extend_beta = F.repeat_to_poly(beta, n)
    extend_gamma = F.repeat_to_poly(gamma, n)
    extend_one = F.repeat_to_poly(one, n)

    for index in range(len(ks)):
        # Initialize numerator and denominator products
        # Now the ith element represents gate i and will have the form:
        # (root_i, ((w0_i, s0_i, k0), (w1_i, s1_i, k1), ..., (wm_i, sm_i,
        # km)))   for m different wires, which is all the
        # information   needed for a single product coefficient
        # for a single gate Multiply up the numerator and
        # denominator irreducibles for each gate and pair the results 
        extend_ks = F.repeat_to_poly(ks[index], n)
        
        numerator_temps = _numerator_irreducible(roots, wires[index], extend_ks, extend_beta, extend_gamma)
        numerator_product = F.mul_mod(numerator_temps, numerator_product)
        denominator_temps = _denominator_irreducible(wires[index], sigma_mappings[index], beta, extend_gamma)
        denominator_product = F.mul_mod(denominator_temps, denominator_product)

    denominator_product_under = F.div_mod(extend_one, denominator_product)
    gate_coefficient = F.mul_mod(numerator_product, denominator_product_under)

    z = F.accumulate_mul_poly(gate_coefficient)
    INTT = nn.Intt(fr.TWO_ADICITY(), fr.TYPE())
    #Compute z poly
    z_poly = INTT(z)
    return z_poly

@calculate_execution_time
def compute_lookup_permutation_poly(n, f, t, h_1, h_2, delta, epsilon):  
    assert f.size(0) == n
    assert t.size(0) == n
    assert h_1.size(0) == n
    assert h_2.size(0) == n

    t_next = torch.zeros(n, 4, dtype=torch.BLS12_381_Fr_G1_Mont)
    t_next[:n-1] = t[1:].clone()
    t_next[-1] = t[0].clone()

    h_1_next = torch.zeros(n, 4, dtype=torch.BLS12_381_Fr_G1_Mont)
    h_1_next[:n-1] = h_1[1:].clone()
    h_1_next[-1] = h_1[0].clone()

    one = fr.one().to("cuda")
    extend_one = F.repeat_to_poly(one, n)
    extend_delta = F.repeat_to_poly(delta, n)
    extend_epsilon = F.repeat_to_poly(epsilon, n)

    product_arguments = _lookup_ratio(extend_one ,extend_delta, extend_epsilon, f, t, t_next.to('cuda'), h_1, h_1_next.to('cuda'), h_2)

    p = F.accumulate_mul_poly(product_arguments)
    INTT = nn.Intt(fr.TWO_ADICITY(), fr.TYPE())
    p_poly = INTT(p)
    
    return p_poly

