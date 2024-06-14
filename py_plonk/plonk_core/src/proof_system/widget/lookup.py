from .....bls12_381 import fr
from dataclasses import dataclass
from typing import List, Tuple
import torch.nn.functional as F
from .....arithmetic import poly_add_poly
@dataclass
class Lookup:
    # Lookup selector
    q_lookup: Tuple[List[fr.Fr],List[fr.Fr]]
    # Column 1 of lookup table
    table_1: List[fr.Fr]
    # Column 2 of lookup table
    table_2: List[fr.Fr]
    # Column 3 of lookup table
    table_3: List[fr.Fr]
    # Column 4 of lookup table
    table_4: List[fr.Fr]

# Linear combination of a series of values
# For values [v_0, v_1,... v_k] returns:
# v_0 + challenge * v_1 + ... + challenge^k  * v_k
def lc(values: list, challenge):
    kth_val = values[-1]
    for val in reversed(values[:-1]):
        kth_val = F.mul_mod_scalar(kth_val, challenge)
        kth_val = F.add_mod(kth_val, val)

    return kth_val

def _compute_quotient_i(
        w_l_i,
        w_r_i,
        w_o_i,
        w_4_i,
        f_i,
        table_i,
        table_i_next,
        h1_i,
        h1_i_next,
        h2_i,
        z2_i,
        z2_i_next,
        l1_i,
        delta,
        epsilon,
        zeta,
        lookup_sep,
        proverkey_q_lookup,
        size
    ):
        # q_lookup(X) * (a(X) + zeta * b(X) + (zeta^2 * c(X)) + (zeta^3 * d(X) - f(X))) * α_1
        one = fr.one()
        
        #single scalar OP on CPU
        lookup_sep_sq = F.mul_mod(lookup_sep, lookup_sep)  # Calculate the square of lookup_sep
        lookup_sep_cu = F.mul_mod(lookup_sep_sq, lookup_sep)  # Calculate the cube of lookup_sep
        one_plus_delta = F.add_mod(delta, one)  # Calculate (1 + δ)
        epsilon_one_plus_delta = F.mul_mod(epsilon, one_plus_delta)  # Calculate ε * (1 + δ)

        epsilon = epsilon.to("cuda")
        delta = delta.to("cuda")
        zeta = zeta.to("cuda")
        lookup_sep = lookup_sep.to("cuda")
        lookup_sep_sq = lookup_sep_sq.to("cuda")
        lookup_sep_cu = lookup_sep_cu.to("cuda")
        one_plus_delta = one_plus_delta.to("cuda")
        epsilon_one_plus_delta = epsilon_one_plus_delta.to("cuda")
        one = one.to("cuda")
        extend_mod = F.repeat_to_poly(fr.MODULUS().to("cuda"), size)

        # Calculate q_lookup_i * (compressed_tuple - f_i)
        q_lookup_i = proverkey_q_lookup
        compressed_tuple = lc([w_l_i, w_r_i, w_o_i, w_4_i], zeta)
        mid1 = F.sub_mod(compressed_tuple,f_i)
        mid2 = F.mul_mod(q_lookup_i, mid1)
        a = F.mul_mod_scalar(mid2, lookup_sep)

        # Calculate z2(X) * (1+δ) * (ε+f(X)) * (ε*(1+δ) + t(X) + δt(Xω)) * lookup_sep^2
        b_0 = F.add_mod_scalar(f_i, epsilon)
        b_1_1 = F.add_mod_scalar(table_i, epsilon_one_plus_delta)
        b_1_2 = F.mul_mod_scalar(table_i_next, delta)
        b_1 = F.add_mod(b_1_1, b_1_2)
        mid1 = F.mul_mod_scalar(z2_i, one_plus_delta)
        mid2 = F.mul_mod(mid1, b_0)
        mid3 = F.mul_mod(mid2, b_1)
        b = F.mul_mod_scalar(mid3, lookup_sep_sq)

        # Calculate -z2(Xω) * (ε*(1+δ) + h1(X) + δ*h2(X)) * (ε*(1+δ) + h2(X) + δ*h1(Xω)) * lookup_sep^2
        c_0_1 = F.add_mod_scalar(h1_i, epsilon_one_plus_delta)
        c_0_2 = F.mul_mod_scalar(h2_i, delta)
        c_0 = F.add_mod(c_0_1, c_0_2)
        c_1_1 = F.add_mod_scalar(h2_i, epsilon_one_plus_delta)
        c_1_2 = F.mul_mod_scalar(h1_i_next, delta)
        c_1 = F.add_mod(c_1_1, c_1_2)
        neg_z2_next = F.sub_mod(extend_mod, z2_i_next)
        # neg_z2_next = neg_extend(z2_i_next, size) ???
        mid1 = F.mul_mod(neg_z2_next, c_0)
        mid2 = F.mul_mod(mid1, c_1)
        c = F.mul_mod_scalar(mid2, lookup_sep_sq)

        # Calculate z2(X) - 1 * l1(X) * lookup_sep^3
        d_1 = F.sub_mod_scalar(z2_i, one)
        d_2 = F.mul_mod_scalar(l1_i, lookup_sep_cu)
        d = F.mul_mod(d_1, d_2)

        # Calculate a(X) + b(X) + c(X) + d(X)
        mid1 = F.add_mod(a, b)
        mid2 = F.add_mod(mid1, c)
        res = F.add_mod(mid2, d)
        return res

    
def compute_linearisation_lookup(
    l1_eval,
    a_eval,
    b_eval,
    c_eval,
    d_eval,
    f_eval,
    table_eval,
    table_next_eval,
    h1_next_eval,
    h2_eval,
    z2_next_eval,
    delta,
    epsilon,
    zeta,
    z2_poly,
    h1_poly,
    lookup_sep,
    pk_q_lookup,
):
    lookup_sep_sq = F.mul_mod(lookup_sep, lookup_sep)
    lookup_sep_cu = F.mul_mod(lookup_sep_sq, lookup_sep)
    one_plus_delta = F.add_mod(delta, fr.one().to("cuda"))
    epsilon_one_plus_delta = F.mul_mod(epsilon, one_plus_delta)

    compressed_tuple = lc([a_eval, b_eval, c_eval, d_eval], zeta)
    compressed_tuple_sub_f_eval = F.sub_mod(compressed_tuple, f_eval)
    const1 = F.mul_mod(compressed_tuple_sub_f_eval, lookup_sep)
    a = F.mul_mod_scalar(pk_q_lookup, const1)

    # z2(X) * (1 + δ) * (ε + f_bar) * (ε(1+δ) + t_bar + δ*tω_bar) *
    # lookup_sep^2
    b_0 = F.add_mod(epsilon, f_eval)
    epsilon_one_plus_delta_plus_tabel_eval = F.add_mod(epsilon_one_plus_delta, table_eval)
    delta_times_table_next_eval = F.mul_mod(delta, table_next_eval)
    b_1 = F.add_mod(epsilon_one_plus_delta_plus_tabel_eval, delta_times_table_next_eval)
    b_2 = F.mul_mod(l1_eval, lookup_sep_cu)
    one_plus_delta_b_0 = F.mul_mod(one_plus_delta, b_0)
    one_plus_delta_b_0_b_1 = F.mul_mod(one_plus_delta_b_0, b_1)
    one_plus_delta_b_0_b_1_lookup = F.mul_mod(one_plus_delta_b_0_b_1, lookup_sep_sq)
    const2 = F.add_mod(one_plus_delta_b_0_b_1_lookup, b_2)
    b = F.mul_mod_scalar(z2_poly, const2)

    # h1(X) * (−z2ω_bar) * (ε(1+δ) + h2_bar  + δh1ω_bar) * lookup_sep^2

    neg_z2_next_eval = F.sub_mod(fr.MODULUS().to("cuda"), z2_next_eval)
    c_0 = F.mul_mod(neg_z2_next_eval, lookup_sep_sq)
    epsilon_one_plus_delta_h2_eval = F.add_mod(epsilon_one_plus_delta, h2_eval)
    delta_h1_next_eval =  F.add_mod(delta, h1_next_eval)
    c_1 = F.add_mod(epsilon_one_plus_delta_h2_eval, delta_h1_next_eval)
    c0_c1 = F.mul_mod(c_0, c_1)
    c = F.mul_mod_scalar(h1_poly, c0_c1)

    ab = poly_add_poly(a, b)
    abc = poly_add_poly(ab, c)

    return abc

# Compute lookup portion of quotient polynomial
def compute_lookup_quotient_term(
    n,
    wl_eval_8n,
    wr_eval_8n,
    wo_eval_8n,
    w4_eval_8n,
    f_eval_8n,
    table_eval_8n,
    h1_eval_8n,
    h2_eval_8n,
    z2_eval_8n,
    l1_eval_8n,
    delta,
    epsilon,
    zeta,
    lookup_sep,
    pk_lookup_qlookup_evals):

    size = 8 * n
  
    # Calculate lookup quotient term for each index
    quotient = _compute_quotient_i(
        wl_eval_8n[:size],
        wr_eval_8n[:size],
        wo_eval_8n[:size],
        w4_eval_8n[:size],
        f_eval_8n[:size],
        table_eval_8n[:size],
        table_eval_8n[8:],
        h1_eval_8n[:size],
        h1_eval_8n[8:],
        h2_eval_8n[:size],
        z2_eval_8n[:size],
        z2_eval_8n[8:],
        l1_eval_8n[:size],
        delta,
        epsilon,
        zeta,
        lookup_sep,
        pk_lookup_qlookup_evals,
        size
    )


    return quotient
