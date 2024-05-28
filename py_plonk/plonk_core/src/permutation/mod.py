from ....plonk_core.src.permutation import constants
from ....arithmetic import NTT,INTT,from_coeff_vec
from ....bls12_381 import fr
import copy
import math
import torch
import torch.nn.functional as F
from ....arithmetic import from_coeff_vec, calculate_execution_time,INTT,NTT

import torch.nn as nn
def extend_tensor(input:torch.tensor,size):
    if input.dim()==2:
        res = torch.zeros(size, 4,len(input), dtype=torch.BLS12_381_Fr_G1_Mont)
        for i in range(len(res)):
            res[i] = input
        return res.to('cuda')
    else:
        res = torch.zeros(size, 4, dtype=torch.BLS12_381_Fr_G1_Mont)
        for i in range(len(res)):
            res[i] = input
        return res.to('cuda')

def numerator_irreducible(root, w, k, beta, gamma):
    mid1 = F.mul_mod(beta,k)
    mid2 = F.mul_mod(mid1,root)
    mid3 = F.add_mod(w,mid2)
    numerator = F.add_mod(mid3,gamma)
    return numerator

def denominator_irreducible(w, sigma, beta, gamma):
    mid1 = F.mul_mod_scalar(sigma, beta)
    mid2 = F.add_mod(w, mid1)
    denominator = F.add_mod(mid2, gamma)
    return denominator

def lookup_ratio(one ,delta, epsilon, f, t, t_next,
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
    zero = fr.Fr.zero()
    one = fr.Fr.one()
    # Constants defining cosets H, k1H, k2H, etc
    ks = [[],[],[],[]]
    ks[0] = fr.Fr.one()
    ks[1] = constants.K1()
    ks[2] = constants.K2()
    ks[3] = constants.K3()
    sigma_mappings = [[],[],[],[]]
    sigma_mappings[0] = NTT(n, sigma_polys[0].to("cuda"))
    sigma_mappings[1] = NTT(n, sigma_polys[1].to("cuda"))
    sigma_mappings[2] = NTT(n, sigma_polys[2].to("cuda"))
    sigma_mappings[3] = NTT(n, sigma_polys[3].to("cuda"))


    # Transpose wires and sigma values to get "rows" in the form [wl_i,
    # wr_i, wo_i, ... ] where each row contains the wire and sigma
    # values for a single gate
    # Compute all roots, same as calculating twiddles, but doubled in size
    roots = zero.repeat(n,1)
    roots[0] = fr.Fr.one()
    for idx in range(1, roots.size(0)):
        roots[idx] = F.mul_mod(roots[idx - 1], domain.group_gen)
    roots = roots.to('cuda')

    numerator_product = one.clone()
    denominator_product = one.clone()
    numerator_product = numerator_product.repeat(n,1)
    denominator_product = denominator_product.repeat(n,1)
    numerator_product = numerator_product.to("cuda")
    denominator_product = denominator_product.to("cuda")

    extend_beta = beta.repeat(n,1)
    extend_gamma = gamma.repeat(n,1)
    extend_one = one.repeat(n,1)
    beta = beta.to("cuda")
    extend_beta = extend_beta.to("cuda")
    extend_gamma = extend_gamma.to("cuda")
    extend_one = extend_one.to("cuda")

    for index in range(len(ks)):
        # Initialize numerator and denominator products
        # Now the ith element represents gate i and will have the form:
        # (root_i, ((w0_i, s0_i, k0), (w1_i, s1_i, k1), ..., (wm_i, sm_i,
        # km)))   for m different wires, which is all the
        # information   needed for a single product coefficient
        # for a single gate Multiply up the numerator and
        # denominator irreducibles for each gate and pair the results 
        extend_ks = ks[index].repeat(n,1)
        extend_ks = extend_ks.to("cuda")
        
        numerator_temps = numerator_irreducible(roots, wires[index], extend_ks, extend_beta, extend_gamma)
        numerator_product = F.mul_mod(numerator_temps, numerator_product)
        denominator_temps = denominator_irreducible(wires[index], sigma_mappings[index], beta, extend_gamma)
        denominator_product = F.mul_mod(denominator_temps, denominator_product)

    denominator_product_under = F.div_mod(extend_one, denominator_product)
    gate_coefficient = F.mul_mod(numerator_product, denominator_product_under)
    z = torch.tensor([], dtype = fr.Fr.Dtype).to("cuda")

    # First element is one
    state = one.clone().to("cuda")
    z = torch.cat((z, state))
    # Accumulate by successively multiplying the scalars   
    for s in gate_coefficient:
        state = F.mul_mod(state, s)
        z = torch.cat((z, state))
    z = z.reshape(-1, fr.Fr.Limbs)   
    # Remove the last(n+1'th) element
    z = z[:-1] 

    #Compute z poly
    z_poly = INTT(n,z)
    z_poly = from_coeff_vec(z_poly)
    return z_poly

@calculate_execution_time
# Define a Python function that mirrors the Rust function
def compute_lookup_permutation_poly(n, f, t, h_1, h_2, delta, epsilon):  ####输出为Tensor
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

    one = fr.Fr.one()
    extend_one = one.repeat(n,1)
    extend_delta = delta.repeat(n,1)
    extend_epsilon = epsilon.repeat(n,1)

    product_arguments = lookup_ratio(extend_one.to("cuda") ,extend_delta.to("cuda"), extend_epsilon.to("cuda"), f.to('cuda'), t.to('cuda'), t_next.to('cuda'), h_1, h_1_next.to('cuda'), h_2)

    state = one.clone().to("cuda")
    p = torch.tensor([], dtype = fr.Fr.Dtype).to('cuda')
    p = torch.cat((p,state))
    for s in product_arguments:
        state = F.mul_mod(state,s)
        p = torch.cat((p,state))
    p = p.reshape(-1, fr.Fr.Limbs)
    p = p[:-1]

    p_poly = INTT(n, p)
    p_poly = from_coeff_vec(p_poly)
    
    return p_poly

