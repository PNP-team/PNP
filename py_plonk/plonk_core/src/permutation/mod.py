from ....plonk_core.src.permutation import constants
from ....arithmetic import NTT,INTT,from_coeff_vec
from ....bls12_381 import fr
import copy
import math
import torch
import torch.nn.functional as F
from ....arithmetic import from_coeff_vec,\
                        from_gmpy_list,from_list_tensor,from_gmpy_list_1,domian_trans_tensor,calculate_execution_time,\
                        INTT_new

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
    
    mid1=F.mul_mod(beta,k)
    mid2=F.mul_mod(mid1,root)
    mid3=F.add_mod(w,mid2)
    mid4=F.add_mod(mid3,gamma)
    return mid4

def denominator_irreducible(w, sigma, beta, gamma):
    mid1=F.mul_mod(beta,sigma)
    mid2=F.add_mod(w,mid1)
    mid3=F.add_mod(mid2,gamma)
    return mid3

def lookup_ratio(one ,delta, epsilon, f, t, t_next,
                h_1, h_1_next, h_2):
   
    one_plus_delta=F.add_mod(delta,one)
    epsilon_one_plus_delta =F.mul_mod(epsilon,one_plus_delta)

    
    mid1= F.add_mod(epsilon,f)
    mid2= F.add_mod(epsilon_one_plus_delta,t)
    mid3= F.mul_mod(delta,t_next)
    mid4= F.add_mod(mid2,mid3)
    mid4= F.add_mod(mid2,mid3)
    mid5= F.mul_mod(one_plus_delta,mid1)
    result= F.mul_mod(mid4,mid5)

  
    mid6 = F.mul_mod(h_2,delta)
    mid7 = F.add_mod(epsilon_one_plus_delta,h_1)
    mid8 = F.add_mod(mid6,mid7)
    mid9 = F.add_mod(epsilon_one_plus_delta,h_2)
    mid10= F.mul_mod(h_1_next,delta)
    mid11 = F.add_mod(mid9,mid10)
    mid12 = F.mul_mod(mid8,mid11)
    # multielement on cuda
    mid12 = F.div_mod(one,mid12)
    result= F.mul_mod(result,mid12)


    return result

@calculate_execution_time
def compute_permutation_poly(domain, wires, beta, gamma, sigma_polys):
    n = domain.size

    # Constants defining cosets H, k1H, k2H, etc
    ks = [beta.one(),constants.K1(),constants.K2(),constants.K3()]
    from_gmpy_list(ks)
    ks=from_list_tensor(ks).to('cuda')

    nttclass = nn.Ntt(n, torch.BLS12_381_Fr_G1_Mont)
    inttclass = nn.Intt(n, torch.BLS12_381_Fr_G1_Mont)

    sigma_mappings = [[],[],[],[]]
    sigma_mappings[0] = nttclass.forward(sigma_polys[0].to('cuda'))
    sigma_mappings[1] = nttclass.forward(sigma_polys[1].to('cuda'))
    sigma_mappings[2] = nttclass.forward(sigma_polys[2].to('cuda'))
    sigma_mappings[3] = nttclass.forward(sigma_polys[3].to('cuda'))

    # Transpose wires and sigma values to get "rows" in the form [wl_i,
    # wr_i, wo_i, ... ] where each row contains the wire and sigma
    # values for a single gate
    gatewise_wires = [
        [w0, w1, w2, w3] for w0, w1, w2, w3 in zip(wires[0],wires[1],wires[2],wires[3])
    ]
    # gatewise_sigmas = [
    #     [s0, s1, s2, s3] for s0, s1, s2, s3 in zip(sigma_mappings_0,sigma_mappings_1,
    #                                                sigma_mappings_2,sigma_mappings_3)
    # ]
    gatewise_sigmas=[
        [s0, s1, s2, s3] for s0, s1, s2, s3 in zip(sigma_mappings[0],sigma_mappings[1],
                                                    sigma_mappings[2],sigma_mappings[3])
    ]
    # Compute all roots, same as calculating twiddles, but doubled in size
    log_size = int(math.log2(n))
    roots = [fr.Fr.zero() for _ in range(1 << log_size )]
    roots[0] = beta.one()
    from_gmpy_list(roots)
    roots=from_list_tensor(roots)

    domain_group_gen=domian_trans_tensor(domain.group_gen)

    for idx in range(1, len(roots)):
        roots[idx] =F.mul_mod(roots[idx - 1].clone(),domain_group_gen)

    roots=roots.to('cuda')
    # Initialize an empty list for product_argument
    product_argument = []
    # Associate each wire value in a gate with the k defining its coset
    
    one =torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    size_1=4 #4
    size_2=len(roots) #1024
    beta_tensor=torch.tensor(from_gmpy_list_1(beta),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    gamma_tensor=torch.tensor(from_gmpy_list_1(gamma),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    numerator_product=torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    denominator_product=torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    extend_numerator_product=extend_tensor(numerator_product,size_2)
    extend_denominator_product=extend_tensor(denominator_product,size_2)
    for index in range(size_1):

        # Initialize numerator and denominator products

    
        # Now the ith element represents gate i and will have the form:
        # (root_i, ((w0_i, s0_i, k0), (w1_i, s1_i, k1), ..., (wm_i, sm_i,
        # km)))   for m different wires, which is all the
        # information   needed for a single product coefficient
        # for a single gate Multiply up the numerator and
        # denominator irreducibles for each gate and pair the results
        extend_ks =extend_tensor(ks[index],size_2) #1024*4
        extend_beta_tensor=extend_tensor(beta_tensor,size_2)
        extend_gamma_tensor=extend_tensor(gamma_tensor,size_2)
        extend_one= extend_tensor(one,size_2)


        numerator_temps = numerator_irreducible(roots, wires[index], extend_ks, extend_beta_tensor, extend_gamma_tensor)
        extend_numerator_product = F.mul_mod(numerator_temps,extend_numerator_product)

        #denominator_temps = denominator_irreducible(gate_wires, gate_sigmas, beta_tensor, gamma_tensor)
        denominator_temps = denominator_irreducible(wires[index], sigma_mappings[index], extend_beta_tensor,extend_gamma_tensor)
        extend_denominator_product = F.mul_mod(denominator_temps,extend_denominator_product)

        # Calculate the product coefficient for the gate
        # denominator_product_under = fr.Fr.inverse(denominator_product)
    #div_mod work on CUDA
    denominator_product_under= F.div_mod(extend_one,extend_denominator_product)
   
    gate_coefficient = F.mul_mod(extend_numerator_product,denominator_product_under)
        
        # Append the gate coefficient to the product_argument list
    product_argument.append(gate_coefficient)

    z=torch.zeros(n,4,dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    # First element is one
    state=torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    z[0]=state
    # Accumulate by successively multiplying the scalars 
    i=1       
    for s in gate_coefficient:
        # state = state.mul(s)
        if(i==n):
            # Remove the last(n+1'th) element
            break 
        state=F.mul_mod(state,s)
        z[i]=state
        i=i+1
    #Compute z poly
    z_poly = inttclass.forward(z)
    z_poly = from_coeff_vec(z_poly)
    return z_poly

@calculate_execution_time
# Define a Python function that mirrors the Rust function
def compute_lookup_permutation_poly(domain, f, t, h_1:torch.Tensor, h_2, delta, epsilon):  ####输出为Tensor
    n = domain.size

    assert len(f) == n
    assert len(t) == n
    assert len(h_1) == n
    assert len(h_2) == n
    t_next = torch.zeros(n,4,dtype=torch.BLS12_381_Fr_G1_Mont)
    t_next[:n-1]=t[1:]
    t_next[n-1:n]=t[0]

    h_1_next = torch.zeros(n,4,dtype=torch.BLS12_381_Fr_G1_Mont)
    h_1_next[:n-1]=h_1[1:]
    h_1_next[n-1:n]=h_1[0]

    product_arguments = []
    one=torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    delta=torch.tensor(from_gmpy_list_1(delta),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    epsilon=torch.tensor(from_gmpy_list_1(epsilon),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    size= len(f)
    one=extend_tensor(one,size)
    delta= extend_tensor(delta,size)
    epsilon =extend_tensor(epsilon,size)

    product_arguments=lookup_ratio(one ,delta, epsilon, f.to('cuda'), t.to('cuda'), t_next.to('cuda'), h_1, h_1_next.to('cuda'), h_2)

    state=torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    p = torch.zeros(n,4,dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    p[0]=state
    i=1
    for s in product_arguments:
        if i==n:
            break
        state = F.mul_mod(state,s)
        p[i]=state
        i+=1

    p_poly=INTT_new(domain,p.to('cuda'))
    p_poly = from_coeff_vec(p_poly)
    
    return p_poly

