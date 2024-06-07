import torch
import torch.nn.functional as F
from ....bls12_381 import fr
from .widget.mod import WitnessValues
from .widget import logic as logic_constraint
from .widget import range as range_constraint
from .widget.fixed_base_scalar_mul import (
    FBSMGate,
    FBSMValues,
)
from .widget.curve_addition import CAGate, CAValues
import torch.nn as nn
from .widget.arithmetic import compute_quotient_i
from ..proof_system.permutation import permutation_compute_quotient
from ..proof_system.widget.lookup import compute_lookup_quotient_term
import numpy as np


# Computes the first lagrange polynomial with the given `scale` over `domain`.
def compute_first_lagrange_poly_scaled(n, scale: torch.Tensor):
    INTT = nn.Intt(n, fr.TYPE())
    x_evals = F.resize_poly(scale, n)
    x_coeffs = INTT(x_evals.to("cuda"))
    return x_coeffs


def compute_gate_constraint_satisfiability(
    coset_NTT,
    range_challenge,
    logic_challenge,
    fixed_base_challenge,
    var_base_challenge,
    arithmetics_evals,
    selectors_evals,
    wl_eval_8n,
    wr_eval_8n,
    wo_eval_8n,
    w4_eval_8n,
    pi_poly,
):

    pi_poly = pi_poly.to("cuda")
    pi_eval_8n = coset_NTT(pi_poly)

    gate_contributions = []

    wit_vals = WitnessValues(
        a_val=wl_eval_8n[:coset_NTT.Size],
        b_val=wr_eval_8n[:coset_NTT.Size],
        c_val=wo_eval_8n[:coset_NTT.Size],
        d_val=w4_eval_8n[:coset_NTT.Size],
    )

    custom_vals = {
        "a_next_eval" : wl_eval_8n[8:],
        "b_next_eval" : wr_eval_8n[8:],
        "d_next_eval" : w4_eval_8n[8:],
        "q_l_eval" : arithmetics_evals.q_l,
        "q_r_eval" : arithmetics_evals.q_r,
        "q_c_eval" : arithmetics_evals.q_c,
            # Possibly unnecessary but included nonetheless...
        "q_hl_eval" : arithmetics_evals.q_hl,
        "q_hr_eval" : arithmetics_evals.q_hr,
        "q_h4_eval" : arithmetics_evals.q_h4,
    }
    
    # start = time.time()
    arithmetic = compute_quotient_i(arithmetics_evals, wit_vals)
    # timings['compute_quotient_i'] += time.time() - start

    # start = time.time()
    range_term = range_constraint.quotient_term(
        selectors_evals.range,
        range_challenge,
        wit_vals,
        custom_vals,
    )

    # start = time.time()
    logic_term = logic_constraint.quotient_term(
        selectors_evals.logic,
        logic_challenge,
        wit_vals,
        custom_vals
    )
    # timings['LogicGate_quotient_term'] += time.time() - start

    # start = time.time()
    fixed_base_scalar_mul_term = FBSMGate.quotient_term(
        selectors_evals.fixed_group_add,
        fixed_base_challenge,
        wit_vals,
        FBSMValues.from_evaluations(custom_vals),
    )
    # timings['FBSMGate_quotient_term'] += time.time() - start

    # start = time.time()
    curve_addition_term = CAGate.quotient_term(
        selectors_evals.variable_group_add,
        var_base_challenge,
        wit_vals,
        CAValues.from_evaluations(custom_vals),
    )
    # timings['CAGate_quotient_term'] += time.time() - start

    mid1 = F.add_mod(arithmetic, pi_eval_8n[:coset_NTT.Size])
    mid2 = F.add_mod(mid1, range_term)
    mid3 = F.add_mod(mid2, logic_term)
    mid4 = F.add_mod(mid3, fixed_base_scalar_mul_term)
    gate_contributions = F.add_mod(mid4, curve_addition_term)

    # for function, total_time in timings.items():
    #     print(f"Total time for {function}: {total_time:.6f} seconds")
    return gate_contributions


def compute_permutation_checks(
    n,
    coset_ntt,
    linear_evaluations_evals,
    permutations_evals,
    wl_eval_8n,
    wr_eval_8n,
    wo_eval_8n,
    w4_eval_8n,
    z_eval_8n,
    alpha,
    beta,
    gamma,
):

    size = 8 * n

    # single scalar OP on CPU
    alpha2 = F.mul_mod(alpha, alpha)

    l1_poly_alpha = compute_first_lagrange_poly_scaled(n, alpha2.to("cuda"))
    l1_alpha_sq_evals = coset_ntt(l1_poly_alpha.to("cuda"))

    # Initialize result list

    # Calculate permutation contribution for each index

    quotient = permutation_compute_quotient(
        size,
        linear_evaluations_evals,
        permutations_evals.left_sigma,
        permutations_evals.right_sigma,
        permutations_evals.out_sigma,
        permutations_evals.fourth_sigma,
        wl_eval_8n[:size],
        wr_eval_8n[:size],
        wo_eval_8n[:size],
        w4_eval_8n[:size],
        z_eval_8n[:size],
        z_eval_8n[8:],
        alpha,
        l1_alpha_sq_evals[:size],
        beta,
        gamma,
    )

    return quotient


def compute_quotient_poly(
    n,
    pk_new,
    z_poly,
    z2_poly,
    w_l_poly,
    w_r_poly,
    w_o_poly,
    w_4_poly,
    public_inputs_poly,
    f_poly,
    table_poly,
    h1_poly,
    h2_poly,
    alpha,
    beta,
    gamma,
    delta,
    epsilon,
    zeta,
    range_challenge,
    logic_challenge,
    fixed_base_challenge,
    var_base_challenge,
    lookup_challenge,
):
    coset_size = 8 * n
    one = fr.one().to("cuda")
    l1_poly = compute_first_lagrange_poly_scaled(n, one)

    coset_NTT = nn.Ntt_coset(fr.TWO_ADICITY(), coset_size, fr.TYPE())
    l1_eval_8n = coset_NTT(l1_poly)
    z_eval_8n = coset_NTT(z_poly)
    z_eval_8n = torch.cat((z_eval_8n, z_eval_8n[:8]), dim=0)

    wl_eval_8n = coset_NTT(w_l_poly)
    wl_eval_8n = torch.cat((wl_eval_8n, wl_eval_8n[:8]), dim=0)

    wr_eval_8n = coset_NTT(w_r_poly)
    wr_eval_8n = torch.cat((wr_eval_8n, wr_eval_8n[:8]), dim=0)

    wo_eval_8n = coset_NTT(w_o_poly)

    w4_eval_8n = coset_NTT(w_4_poly)
    w4_eval_8n = torch.cat((w4_eval_8n, w4_eval_8n[:8]), dim=0)

    z2_eval_8n = coset_NTT(z2_poly)
    z2_eval_8n = torch.cat((z2_eval_8n, z2_eval_8n[:8]), dim=0)

    f_eval_8n = coset_NTT(f_poly)

    table_eval_8n = coset_NTT(table_poly)
    table_eval_8n = torch.cat((table_eval_8n, table_eval_8n[:8]), dim=0)

    h1_eval_8n = coset_NTT(h1_poly)
    h1_eval_8n = torch.cat((h1_eval_8n, h1_eval_8n[:8]), dim=0)

    h2_eval_8n = coset_NTT(h2_poly)

    gate_constraints = compute_gate_constraint_satisfiability(
        coset_NTT,
        range_challenge,
        logic_challenge,
        fixed_base_challenge,
        var_base_challenge,
        pk_new.arithmetics_evals,
        pk_new.selectors_evals,
        wl_eval_8n,
        wr_eval_8n,
        wo_eval_8n,
        w4_eval_8n,
        public_inputs_poly,
    )
    permutation = compute_permutation_checks(
        n,
        coset_NTT,
        pk_new.linear_evaluations_evals,
        pk_new.permutations_evals,
        wl_eval_8n,
        wr_eval_8n,
        wo_eval_8n,
        w4_eval_8n,
        z_eval_8n,
        alpha,
        beta,
        gamma,
    )

    lookup = compute_lookup_quotient_term(
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
        lookup_challenge,
        pk_new.lookups_evals.q_lookup,
    )

    numerator = F.add_mod(gate_constraints, permutation)
    numerator = F.add_mod(numerator, lookup)
    denominator = F.inv_mod(pk_new.v_h_coset_8n_evals)
    res = F.mul_mod(numerator, denominator)
    coset_INTT = nn.Intt_coset(fr.TWO_ADICITY(), fr.TYPE())
    quotient_poly = coset_INTT(res)
    return quotient_poly
