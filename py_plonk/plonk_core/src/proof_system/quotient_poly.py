from ....domain import Radix2EvaluationDomain
import gmpy2
import torch
import copy
import torch.nn.functional as F
from ....bls12_381 import fr
from ....arithmetic import from_coeff_vec, coset_NTT
from .widget.mod import WitnessValues
from .widget.range import RangeGate, RangeValues
from .widget.logic import LogicGate, LogicValues
from .widget.fixed_base_scalar_mul import (
    FBSMGate,
    FBSMValues,
)
from .widget.curve_addition import CAGate, CAValues
from ....arithmetic import (
    from_coeff_vec,
    calculate_execution_time,
    coset_NTT,
    coset_INTT,
)
import torch.nn as nn
from .widget.arithmetic import compute_quotient_i
from ..proof_system.permutation import permutation_compute_quotient
from ..proof_system.widget.lookup import compute_lookup_quotient_term
import numpy as np
import time


# Computes the first lagrange polynomial with the given `scale` over `domain`.
def compute_first_lagrange_poly_scaled(n, scale: torch.Tensor):
    inttclass = nn.Intt(n, fr.TYPE())
    x_evals = torch.zeros(n, fr.LIMBS(), dtype=fr.TYPE())
    x_evals[0] = scale.clone()

    x_coeffs = inttclass.forward(x_evals.to("cuda"))
    result_poly = from_coeff_vec(x_coeffs)
    return result_poly


def compute_gate_constraint_satisfiability(
    n,
    range_challenge,
    logic_challenge,
    fixed_base_challenge,
    var_base_challenge,
    prover_key,
    wl_eval_8n,
    wr_eval_8n,
    wo_eval_8n,
    w4_eval_8n,
    pi_poly,
):

    size = 8 * n
    pi_poly = pi_poly.to("cuda")
    pi_eval_8n = coset_NTT(size, pi_poly)

    gate_contributions = []

    def convert_to_tensors(data):
        for key, value in data.items():
            if isinstance(value, dict):
                convert_to_tensors(value)  # Recursively apply conversion
            elif isinstance(value, np.ndarray):  ##4575657222473777152 ndarray problem
                if np.array_equal(value, np.array(0, dtype=np.uint64)):
                    value = []
                data[key] = torch.tensor(
                    value, dtype=fr.TYPE()
                )  # Convert numpy array to tensor

    prover_key_arithmetic = prover_key["arithmetic"].tolist()
    convert_to_tensors(prover_key_arithmetic)
    for key in [
        "q_l",
        "q_r",
        "q_c",
        "q_m",
        "q_o",
        "q_4",
        "q_hl",
        "q_hr",
        "q_h4",
        "q_arith",
    ]:
        prover_key_arithmetic[key]["evals"] = prover_key_arithmetic[key]["evals"].to(
            "cuda"
        )

    prover_key_range_selector = prover_key["range_selector"].tolist()
    convert_to_tensors(prover_key_range_selector)
    prover_key_range_selector["evals"] = prover_key_range_selector["evals"].to("cuda")

    prover_key_logic_selector = prover_key["logic_selector"].tolist()
    convert_to_tensors(prover_key_logic_selector)
    prover_key_logic_selector["evals"] = prover_key_logic_selector["evals"].to("cuda")

    prover_key_fixed_group_add_selector = prover_key[
        "fixed_group_add_selector"
    ].tolist()
    convert_to_tensors(prover_key_fixed_group_add_selector)
    prover_key_fixed_group_add_selector["evals"] = prover_key_fixed_group_add_selector[
        "evals"
    ].to("cuda")

    prover_key_variable_group_add_selector = prover_key[
        "variable_group_add_selector"
    ].tolist()
    convert_to_tensors(prover_key_variable_group_add_selector)
    prover_key_variable_group_add_selector["evals"] = (
        prover_key_variable_group_add_selector["evals"].to("cuda")
    )

    # timings = {
    # 'compute_quotient_i': 0,
    # 'RangeGate_quotient_term': 0,
    # 'LogicGate_quotient_term': 0,
    # 'FBSMGate_quotient_term': 0,
    # 'CAGate_quotient_term': 0
    # }

    wit_vals = WitnessValues(
        a_val=wl_eval_8n[:size],
        b_val=wr_eval_8n[:size],
        c_val=wo_eval_8n[:size],
        d_val=w4_eval_8n[:size],
    )

    custom_vals = {
        "a_next_eval" : wl_eval_8n[8:],
        "b_next_eval" : wr_eval_8n[8:],
        "d_next_eval" : w4_eval_8n[8:],
        "q_l_eval" : prover_key_arithmetic["q_l"]["evals"].clone(),
        "q_r_eval" : prover_key_arithmetic["q_r"]["evals"].clone(),
        "q_c_eval" : prover_key_arithmetic["q_c"]["evals"].clone(),
            # Possibly unnecessary but included nonetheless...
        "q_hl_eval" : prover_key_arithmetic["q_hl"]["evals"].clone(),
        "q_hr_eval" : prover_key_arithmetic["q_hr"]["evals"].clone(),
        "q_h4_eval" : prover_key_arithmetic["q_h4"]["evals"].clone(),
    }
    
    # start = time.time()
    arithmetic = compute_quotient_i(prover_key_arithmetic, wit_vals)
    # timings['compute_quotient_i'] += time.time() - start

    # start = time.time()
    range_term = RangeGate.quotient_term(
        prover_key_range_selector["evals"],
        range_challenge,
        wit_vals,
        RangeValues.from_evaluations(custom_vals),
    )
    # timings['RangeGate_quotient_term'] += time.time() - start

    # start = time.time()
    logic_term = LogicGate.quotient_term(
        prover_key_logic_selector["evals"],
        logic_challenge,
        wit_vals,
        LogicValues.from_evaluations(custom_vals),
    )
    # timings['LogicGate_quotient_term'] += time.time() - start

    # start = time.time()
    fixed_base_scalar_mul_term = FBSMGate.quotient_term(
        prover_key_fixed_group_add_selector["evals"],
        fixed_base_challenge,
        wit_vals,
        FBSMValues.from_evaluations(custom_vals),
    )
    # timings['FBSMGate_quotient_term'] += time.time() - start

    # start = time.time()
    curve_addition_term = CAGate.quotient_term(
        prover_key_variable_group_add_selector["evals"],
        var_base_challenge,
        wit_vals,
        CAValues.from_evaluations(custom_vals),
    )
    # timings['CAGate_quotient_term'] += time.time() - start

    mid1 = F.add_mod(arithmetic, pi_eval_8n[:size])
    mid2 = F.add_mod(mid1, range_term)
    mid3 = F.add_mod(mid2, logic_term)
    mid4 = F.add_mod(mid3, fixed_base_scalar_mul_term)
    gate_contributions = F.add_mod(mid4, curve_addition_term)

    # for function, total_time in timings.items():
    #     print(f"Total time for {function}: {total_time:.6f} seconds")
    return gate_contributions


def compute_permutation_checks(
    n,
    prover_key,
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
    l1_alpha_sq_evals = coset_NTT(size, l1_poly_alpha.to("cuda"))

    # Initialize result list
    pk_permutation = prover_key["permutation"].tolist()
    pk_linear_evaluations = prover_key["linear_evaluations"].tolist()

    pk_linear_evaluations_evals = torch.tensor(
        pk_linear_evaluations["evals"], dtype=fr.TYPE()
    ).to("cuda")
    pk_left_sigma_evals = torch.tensor(
        pk_permutation["left_sigma"]["evals"], dtype=fr.TYPE()
    ).to("cuda")
    pk_right_sigma_evals = torch.tensor(
        pk_permutation["right_sigma"]["evals"], dtype=fr.TYPE()
    ).to("cuda")
    pk_out_sigma_evals = torch.tensor(
        pk_permutation["out_sigma"]["evals"], dtype=fr.TYPE()
    ).to("cuda")
    pk_fourth_sigma_evals = torch.tensor(
        pk_permutation["fourth_sigma"]["evals"], dtype=fr.TYPE()
    ).to("cuda")

    # Calculate permutation contribution for each index

    quotient = permutation_compute_quotient(
        size,
        pk_linear_evaluations_evals,
        pk_left_sigma_evals,
        pk_right_sigma_evals,
        pk_out_sigma_evals,
        pk_fourth_sigma_evals,
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


@calculate_execution_time
def compute_quotient_poly(
    n,
    prover_key,
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
    one = fr.one()
    l1_poly = compute_first_lagrange_poly_scaled(n, one)

    l1_eval_8n = coset_NTT(coset_size, l1_poly.to("cuda"))
    z_eval_8n = coset_NTT(coset_size, z_poly.to("cuda"))

    z_eval_8n = torch.cat((z_eval_8n, z_eval_8n[:8]), dim=0)

    wl_eval_8n = coset_NTT(coset_size, w_l_poly.to("cuda"))
    wl_eval_8n = torch.cat((wl_eval_8n, wl_eval_8n[:8]), dim=0)

    wr_eval_8n = coset_NTT(coset_size, w_r_poly.to("cuda"))
    wr_eval_8n = torch.cat((wr_eval_8n, wr_eval_8n[:8]), dim=0)

    wo_eval_8n = coset_NTT(coset_size, w_o_poly.to("cuda"))

    w4_eval_8n = coset_NTT(coset_size, w_4_poly.to("cuda"))
    w4_eval_8n = torch.cat((w4_eval_8n, w4_eval_8n[:8]), dim=0)

    z2_eval_8n = coset_NTT(coset_size, z2_poly.to("cuda"))
    z2_eval_8n = torch.cat((z2_eval_8n, z2_eval_8n[:8]), dim=0)

    f_eval_8n = coset_NTT(coset_size, f_poly.to("cuda"))

    table_eval_8n = coset_NTT(coset_size, table_poly.to("cuda"))
    table_eval_8n = torch.cat((table_eval_8n, table_eval_8n[:8]), dim=0)

    h1_eval_8n = coset_NTT(coset_size, h1_poly.to("cuda"))
    h1_eval_8n = torch.cat((h1_eval_8n, h1_eval_8n[:8]), dim=0)

    h2_eval_8n = coset_NTT(coset_size, h2_poly.to("cuda"))

    # range_challenge = range_challenge.to('cuda')
    # logic_challenge = logic_challenge.to('cuda')
    # fixed_base_challenge = fixed_base_challenge.to('cuda')
    # var_base_challenge = var_base_challenge.to('cuda')
    # lookup_challenge = lookup_challenge.to('cuda')

    gate_constraints = compute_gate_constraint_satisfiability(
        n,
        range_challenge,
        logic_challenge,
        fixed_base_challenge,
        var_base_challenge,
        prover_key,
        wl_eval_8n,
        wr_eval_8n,
        wo_eval_8n,
        w4_eval_8n,
        public_inputs_poly,
    )
    permutation = compute_permutation_checks(
        n,
        prover_key,
        wl_eval_8n,
        wr_eval_8n,
        wo_eval_8n,
        w4_eval_8n,
        z_eval_8n,
        alpha,
        beta,
        gamma,
    )
    pk_lookup = prover_key["lookup"].tolist()
    pk_lookup_qlookup_evals = torch.tensor(
        pk_lookup["q_lookup"]["evals"], dtype=fr.TYPE()
    ).to("cuda")

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
        pk_lookup_qlookup_evals,
    )

    prover_key_v_h_coset_8n = prover_key["v_h_coset_8n"].tolist()
    prover_key_v_h_coset_8n_evals = torch.tensor(
        prover_key_v_h_coset_8n["evals"], dtype=fr.TYPE()
    ).to("cuda")

    numerator = F.add_mod(gate_constraints, permutation)
    numerator = F.add_mod(numerator, lookup)
    denominator = F.inv_mod(prover_key_v_h_coset_8n_evals)
    res = F.mul_mod(numerator, denominator)
    quotient_poly = coset_INTT(coset_size, res)
    hx = from_coeff_vec(quotient_poly)

    return hx
