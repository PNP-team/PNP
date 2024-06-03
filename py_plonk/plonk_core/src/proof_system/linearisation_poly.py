from dataclasses import dataclass
from ....domain import Radix2EvaluationDomain
from ....bls12_381 import fr

# from ....plonk_core.src.proof_system.prover_key import Prover_Key
from ....plonk_core.src.proof_system.widget.mod import WitnessValues
from ....plonk_core.src.proof_system.widget.range import RangeGate, RangeValues
from ....plonk_core.src.proof_system.widget.logic import LogicGate, LogicValues
from ....plonk_core.src.proof_system.widget.fixed_base_scalar_mul import (
    FBSMGate,
    FBSMValues,
)
from ....plonk_core.src.proof_system.widget.curve_addition import CAGate, CAValues
from ....arithmetic import (
    poly_mul_const,
    poly_add_poly,
    compute_first_lagrange_evaluation,
    calculate_execution_time,
)
import torch
import torch.nn.functional as F
import numpy as np
from .widget.arithmetic import compute_linearisation_arithmetic
from .widget.lookup import compute_linearisation_lookup
from ..proof_system.permutation import compute_linearisation_permutation


@dataclass
class WireEvaluations:
    # Evaluation of the witness polynomial for the left wire at `z`.
    a_eval: fr.Fr

    # Evaluation of the witness polynomial for the right wire at `z`.
    b_eval: fr.Fr

    # Evaluation of the witness polynomial for the output wire at `z`.
    c_eval: fr.Fr

    # Evaluation of the witness polynomial for the fourth wire at `z`.
    d_eval: fr.Fr


@dataclass
class PermutationEvaluations:
    # Evaluation of the left sigma polynomial at `z`.
    left_sigma_eval: fr.Fr

    # Evaluation of the right sigma polynomial at `z`.
    right_sigma_eval: fr.Fr

    # Evaluation of the out sigma polynomial at `z`.
    out_sigma_eval: fr.Fr

    # Evaluation of the permutation polynomial at `z * omega` where `omega`
    # is a root of unity.
    permutation_eval: fr.Fr


@dataclass
class LookupEvaluations:
    q_lookup_eval: fr.Fr

    # (Shifted) Evaluation of the lookup permutation polynomial at `z * root of unity`
    z2_next_eval: fr.Fr

    # Evaluations of the first half of sorted plonkup poly at `z`
    h1_eval: fr.Fr

    # (Shifted) Evaluations of the even indexed half of sorted plonkup poly
    # at `z root of unity
    h1_next_eval: fr.Fr

    # Evaluations of the odd indexed half of sorted plonkup poly at `z
    # root of unity
    h2_eval: fr.Fr

    # Evaluations of the query polynomial at `z`
    f_eval: fr.Fr

    # Evaluations of the table polynomial at `z`
    table_eval: fr.Fr

    # Evaluations of the table polynomial at `z * root of unity`
    table_next_eval: fr.Fr


@dataclass
class ProofEvaluations:
    # Wire evaluations
    wire_evals: WireEvaluations

    # Permutation and sigma polynomials evaluations
    perm_evals: PermutationEvaluations

    # Lookup evaluations
    lookup_evals: LookupEvaluations

    # Evaluations needed for custom gates. This includes selector polynomials
    # and evaluations of wire polynomials at an offset
    custom_evals: dict


def convert_to_tensors(data):
    for key, value in data.items():
        if isinstance(value, dict):
            convert_to_tensors(value)  # Recursively apply conversion
        elif isinstance(value, np.ndarray):
            if np.array_equal(value, np.array(0, dtype=np.uint64)):
                value = []
            data[key] = torch.tensor(
                value, dtype=torch.BLS12_381_Fr_G1_Mont
            )  # Convert numpy array to tensor


@calculate_execution_time
def compute_linearisation_poly(
    domain: Radix2EvaluationDomain,
    prover_key,
    alpha,
    beta,
    gamma,
    delta,
    epsilon,
    zeta,
    range_separation_challenge,
    logic_separation_challenge,
    fixed_base_separation_challenge,
    var_base_separation_challenge,
    lookup_separation_challenge,
    z_challenge,
    w_l_poly: torch.Tensor,
    w_r_poly: torch.Tensor,
    w_o_poly: torch.Tensor,
    w_4_poly: torch.Tensor,
    t_1_poly: torch.Tensor,
    t_2_poly: torch.Tensor,
    t_3_poly: torch.Tensor,
    t_4_poly: torch.Tensor,
    t_5_poly: torch.Tensor,
    t_6_poly: torch.Tensor,
    t_7_poly: torch.Tensor,
    t_8_poly: torch.Tensor,
    z_poly: torch.Tensor,
    z2_poly: torch.Tensor,
    f_poly: torch.Tensor,
    h1_poly: torch.Tensor,
    h2_poly: torch.Tensor,
    table_poly: torch.Tensor,
):
    n = domain.size
    omega = domain.group_gen
    domain_permutation = Radix2EvaluationDomain.new(n)
    # single scalar OP on CPU
    one = fr.one()
    mod = fr.MODULUS()
    neg_one = F.sub_mod(mod, one)
    shifted_z_challenge = F.mul_mod(z_challenge, omega)
    vanishing_poly_eval = domain.evaluate_vanishing_polynomial(z_challenge)
    z_challenge_to_n = F.add_mod(vanishing_poly_eval, one)
    # Compute the last term in the linearisation polynomial (negative_quotient_term):
    # - Z_h(z_challenge) * [t_1(X) + z_challenge^n * t_2(X) + z_challenge^2n *
    # t_3(X) + z_challenge^3n * t_4(X)]
    l1_eval = compute_first_lagrange_evaluation(
        domain.size, vanishing_poly_eval, z_challenge
    )

    w_l_poly = w_l_poly.to("cuda")
    w_r_poly = w_r_poly.to("cuda")
    w_o_poly = w_o_poly.to("cuda")
    w_4_poly = w_4_poly.to("cuda")
    t_1_poly = t_1_poly.to("cuda")
    t_2_poly = t_2_poly.to("cuda")
    t_3_poly = t_3_poly.to("cuda")
    t_4_poly = t_4_poly.to("cuda")
    t_5_poly = t_5_poly.to("cuda")
    t_6_poly = t_6_poly.to("cuda")
    t_7_poly = t_7_poly.to("cuda")
    t_8_poly = t_8_poly.to("cuda")
    z_poly = z_poly.to("cuda")
    z2_poly = z2_poly.to("cuda")
    f_poly = f_poly.to("cuda")
    h1_poly = h1_poly.to("cuda")
    h2_poly = h2_poly.to("cuda")
    table_poly = table_poly.to("cuda")
    z_challenge = z_challenge.to("cuda")
    shifted_z_challenge = shifted_z_challenge.to("cuda")
    # Wire evaluations
    a_eval = F.evaluate(w_l_poly, z_challenge)
    b_eval = F.evaluate(w_r_poly, z_challenge)
    c_eval = F.evaluate(w_o_poly, z_challenge)
    d_eval = F.evaluate(w_4_poly, z_challenge)

    wire_evals = WireEvaluations(a_eval, b_eval, c_eval, d_eval)

    # Permutation evaluations
    pk_permutation = prover_key["permutation"].tolist()
    pk_left_sigma_coeff = torch.tensor(
        pk_permutation["left_sigma"]["coeffs"], dtype=fr.TYPE()
    ).to("cuda")
    pk_right_sigma_coeff = torch.tensor(
        pk_permutation["right_sigma"]["coeffs"], dtype=fr.TYPE()
    ).to("cuda")
    pk_out_sigma_coeff = torch.tensor(
        pk_permutation["out_sigma"]["coeffs"], dtype=fr.TYPE()
    ).to("cuda")
    pk_fourth_sigma_coeff = torch.tensor(
        pk_permutation["fourth_sigma"]["coeffs"], dtype=fr.TYPE()
    ).to("cuda")
    left_sigma_eval = F.evaluate(pk_left_sigma_coeff, z_challenge)
    right_sigma_eval = F.evaluate(pk_right_sigma_coeff, z_challenge)
    out_sigma_eval = F.evaluate(pk_out_sigma_coeff, z_challenge)
    permutation_eval = F.evaluate(z_poly, shifted_z_challenge)

    perm_evals = PermutationEvaluations(
        left_sigma_eval, right_sigma_eval, out_sigma_eval, permutation_eval
    )

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
        if prover_key_arithmetic[key]["coeffs"].is_cuda:
            pass
        else:
            prover_key_arithmetic[key]["coeffs"] = prover_key_arithmetic[key][
                "coeffs"
            ].to("cuda")

    prover_key_lookup = prover_key["lookup"].tolist()
    convert_to_tensors(prover_key_lookup)
    prover_key_lookup["q_lookup"]["coeffs"] = prover_key_lookup["q_lookup"][
        "coeffs"
    ].to("cuda")
    # Arith selector evaluation
    q_arith_eval = F.evaluate(prover_key_arithmetic["q_arith"]["coeffs"], z_challenge)

    # Lookup selector evaluation
    q_lookup_eval = F.evaluate(prover_key_lookup["q_lookup"]["coeffs"], z_challenge)

    # Custom gate evaluations
    q_c_eval = F.evaluate(prover_key_arithmetic["q_c"]["coeffs"], z_challenge)
    q_l_eval = F.evaluate(prover_key_arithmetic["q_l"]["coeffs"], z_challenge)
    q_r_eval = F.evaluate(prover_key_arithmetic["q_r"]["coeffs"], z_challenge)
    a_next_eval = F.evaluate(w_l_poly, shifted_z_challenge)
    b_next_eval = F.evaluate(w_r_poly, shifted_z_challenge)
    d_next_eval = F.evaluate(w_4_poly, shifted_z_challenge)

    # High degree selector evaluations
    q_hl_eval = F.evaluate(prover_key_arithmetic["q_hl"]["coeffs"], z_challenge)
    q_hr_eval = F.evaluate(prover_key_arithmetic["q_hr"]["coeffs"], z_challenge)
    q_h4_eval = F.evaluate(prover_key_arithmetic["q_h4"]["coeffs"], z_challenge)

    custom_evals = {
        "q_arith_eval": q_arith_eval,
        "q_c_eval": q_c_eval,
        "q_l_eval": q_l_eval,
        "q_r_eval": q_r_eval,
        "q_hl_eval": q_hl_eval,
        "q_hr_eval": q_hr_eval,
        "q_h4_eval": q_h4_eval,
        "a_next_eval": a_next_eval,
        "b_next_eval": b_next_eval,
        "d_next_eval": d_next_eval,
    }

    z2_next_eval = F.evaluate(z2_poly, shifted_z_challenge)
    h1_eval = F.evaluate(h1_poly, z_challenge)
    h1_next_eval = F.evaluate(h1_poly, shifted_z_challenge)
    h2_eval = F.evaluate(h2_poly, z_challenge)
    f_eval = F.evaluate(f_poly, z_challenge)
    table_eval = F.evaluate(table_poly, z_challenge)
    table_next_eval = F.evaluate(table_poly, shifted_z_challenge)

    lookup_evals = LookupEvaluations(
        q_lookup_eval,
        z2_next_eval,
        h1_eval,
        h1_next_eval,
        h2_eval,
        f_eval,
        table_eval,
        table_next_eval,
    )

    gate_constraints = compute_gate_constraint_satisfiability(
        range_separation_challenge,
        logic_separation_challenge,
        fixed_base_separation_challenge,
        var_base_separation_challenge,
        wire_evals,
        q_arith_eval,
        custom_evals,
        prover_key,
        prover_key_arithmetic,
    )

    lookup = compute_linearisation_lookup(
        l1_eval.to("cuda"),
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
        delta.to("cuda"),
        epsilon.to("cuda"),
        zeta.to("cuda"),
        z2_poly,
        h1_poly,
        lookup_separation_challenge.to("cuda"),
        prover_key_lookup["q_lookup"]["coeffs"].to("cuda"),
    )

    permutation = compute_linearisation_permutation(
        z_challenge,
        (alpha.to("cuda"), beta.to("cuda"), gamma.to("cuda")),
        (a_eval, b_eval, c_eval, d_eval),
        (left_sigma_eval, right_sigma_eval, out_sigma_eval),
        permutation_eval,
        z_poly,
        domain_permutation,
        pk_fourth_sigma_coeff,
    )

    z_challenge_to_n = z_challenge_to_n.to("cuda")
    # Calculate t_8_poly * z_challenge_to_n
    term_1 = poly_mul_const(t_8_poly, z_challenge_to_n)

    # Calculate (term_1 + t_7_poly) * z_challenge_to_n

    term_2_1 = poly_add_poly(term_1, t_7_poly)
    term_2 = poly_mul_const(term_2_1, z_challenge_to_n)

    # Calculate (term_2 + t_6_poly) * z_challenge_to_n
    term_3_1 = poly_add_poly(term_2, t_6_poly)
    term_3 = poly_mul_const(term_3_1, z_challenge_to_n)

    # Calculate (term_3 + t_5_poly) * z_challenge_to_n
    term_4_1 = poly_add_poly(term_3, t_5_poly)
    term_4 = poly_mul_const(term_4_1, z_challenge_to_n)

    # Calculate (term_4 + t_4_poly) * z_challenge_to_n
    term_5_1 = poly_add_poly(term_4, t_4_poly)
    term_5 = poly_mul_const(term_5_1, z_challenge_to_n)

    # Calculate (term_5 + t_3_poly) * z_challenge_to_n
    term_6_1 = poly_add_poly(term_5, t_3_poly)
    term_6 = poly_mul_const(term_6_1, z_challenge_to_n)

    # Calculate (term_6 + t_2_poly) * z_challenge_to_n
    term_7_1 = poly_add_poly(term_6, t_2_poly)
    term_7 = poly_mul_const(term_7_1, z_challenge_to_n)

    # Calculate (term_7 + t_1_poly) * vanishing_poly_eval
    term_8_1 = poly_add_poly(term_7, t_1_poly)
    vanishing_poly_eval = vanishing_poly_eval.to("cuda")
    quotient_term = poly_mul_const(term_8_1, vanishing_poly_eval)

    neg_one = neg_one.to("cuda")
    negative_quotient_term = poly_mul_const(quotient_term, neg_one)
    linearisation_polynomial_term_1 = poly_add_poly(gate_constraints, permutation)
    linearisation_polynomial_term_2 = poly_add_poly(lookup, negative_quotient_term)
    linearisation_polynomial = poly_add_poly(
        linearisation_polynomial_term_1, linearisation_polynomial_term_2
    )

    proof_evaluations = ProofEvaluations(
        wire_evals, perm_evals, lookup_evals, custom_evals
    )
    return linearisation_polynomial, proof_evaluations


# Computes the gate constraint satisfiability portion of the linearisation polynomial.
def compute_gate_constraint_satisfiability(
    range_separation_challenge,
    logic_separation_challenge,
    fixed_base_separation_challenge,
    var_base_separation_challenge,
    wire_evals: WireEvaluations,
    q_arith_eval,
    custom_evals,
    prover_key,
    prover_key_arithmetic,
):
    wit_vals = WitnessValues(
        a_val=wire_evals.a_eval,
        b_val=wire_evals.b_eval,
        c_val=wire_evals.c_eval,
        d_val=wire_evals.d_eval,
    )

    arithmetic = compute_linearisation_arithmetic(
        wire_evals.a_eval,
        wire_evals.b_eval,
        wire_evals.c_eval,
        wire_evals.d_eval,
        q_arith_eval,
        prover_key_arithmetic,
    )

    pk_range_selector = prover_key["range_selector"].tolist()
    pk_range_selector["coeffs"] = torch.tensor(
        pk_range_selector["coeffs"], dtype=fr.TYPE()
    ).to("cuda")
    range_separation_challenge = range_separation_challenge.to("cuda")
    range = RangeGate.linearisation_term(
        pk_range_selector["coeffs"],
        range_separation_challenge,
        wit_vals,
        RangeValues.from_evaluations(custom_evals),
    )

    pk_logic_selector = prover_key["logic_selector"].tolist()
    pk_logic_selector["coeffs"] = torch.tensor(
        pk_logic_selector["coeffs"], dtype=fr.TYPE()
    ).to("cuda")
    logic_separation_challenge = logic_separation_challenge.to("cuda")
    logic = LogicGate.linearisation_term(
        pk_logic_selector["coeffs"],
        logic_separation_challenge,
        wit_vals,
        LogicValues.from_evaluations(custom_evals),
    )

    pk_fixed_group_add_selector = prover_key["fixed_group_add_selector"].tolist()
    pk_fixed_group_add_selector["coeffs"] = torch.tensor(
        pk_fixed_group_add_selector["coeffs"], dtype=fr.TYPE()
    ).to("cuda")
    fixed_base_separation_challenge = fixed_base_separation_challenge.to("cuda")
    fixed_base_scalar_mul = FBSMGate.linearisation_term(
        pk_fixed_group_add_selector["coeffs"],
        fixed_base_separation_challenge,
        wit_vals,
        FBSMValues.from_evaluations(custom_evals),
    )

    var_base_separation_challenge = var_base_separation_challenge.to("cuda")
    pk_variable_group_add_selector = prover_key["variable_group_add_selector"].tolist()
    pk_variable_group_add_selector["coeffs"] = torch.tensor(
        pk_variable_group_add_selector["coeffs"], dtype=fr.TYPE()
    ).to("cuda")
    curve_addition = CAGate.linearisation_term(
        pk_variable_group_add_selector["coeffs"],
        var_base_separation_challenge,
        wit_vals,
        CAValues.from_evaluations(custom_evals),
    )

    mid1 = poly_add_poly(arithmetic, range)
    mid2 = poly_add_poly(mid1, logic)
    mid3 = poly_add_poly(mid2, fixed_base_scalar_mul)
    res = poly_add_poly(mid3, curve_addition)
    return res
