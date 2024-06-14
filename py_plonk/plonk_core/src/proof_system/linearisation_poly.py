from dataclasses import dataclass
from ....domain import Radix2EvaluationDomain
from ....bls12_381 import fr

from .widget.mod import WitnessValues
from .widget import logic as logic_constraint
from .widget import range as range_constraint
from .widget.fixed_base_scalar_mul import (
    FBSMGate,
    FBSMValues,
)
from .widget.GAGate import CAGate
from ....arithmetic import (
    poly_add_poly,
)
import torch.nn.functional as F
import numpy as np
from .widget.arithmetic import compute_linearisation_arithmetic
from .widget.lookup import compute_linearisation_lookup
from ..proof_system.permutation import compute_linearisation_permutation

class PermutationEvaluations:
    def __init__(self, left_sigma_eval, right_sigma_eval, out_sigma_eval, permutation_eval):
        self.left_sigma_eval = left_sigma_eval # left sigma polynomial at z
        self.right_sigma_eval = right_sigma_eval # right sigma polynomial at z
        self.out_sigma_eval = out_sigma_eval # out sigma polynomial at z
        self.permutation_eval = permutation_eval # permutation polynomial at z*omega

class LookupEvaluations:

    def __init__(
        self,
        q_lookup_eval,
        z2_next_eval,
        h1_eval,
        h1_next_eval,
        h2_eval,
        f_eval,
        table_eval,
        table_next_eval
    ):
        self.q_lookup_eval = q_lookup_eval
        self.z2_next_eval = z2_next_eval
        self.h1_eval = h1_eval
        self.h1_next_eval = h1_next_eval
        self.h2_eval = h2_eval
        self.f_eval = f_eval
        self.table_eval = table_eval
        self.table_next_eval = table_next_eval
        
@dataclass
class ProofEvaluations:
    def __init__(self, wire_evals, perm_evals, lookup_evals, custom_evals):
        self.wire_evals = wire_evals
        self.perm_evals = perm_evals
        self.lookup_evals = lookup_evals
        self.custom_evals = custom_evals


# The first lagrange polynomial has the expression:
# L_0(X) = mul_from_1_to_(n-1) [(X - omega^i) / (1 - omega^i)]
#
# with `omega` being the generator of the domain (the `n`th root of unity).
#
# We use two equalities:
#   1. `mul_from_2_to_(n-1) [1 / (1 - omega^i)] = 1 / n`
#   2. `mul_from_2_to_(n-1) [(X - omega^i)] = (X^n - 1) / (X - 1)`
# to obtain the expression:
# L_0(X) = (X^n - 1) / n * (X - 1)
def _compute_first_lagrange_evaluation(size, z_h_eval, z_challenge):
    # single scalar OP on CPU
    one = fr.one()
    n_fr = fr.make_tensor(size)
    z_challenge_sub_one = F.sub_mod(z_challenge, one)
    denom = F.mul_mod(n_fr, z_challenge_sub_one)
    denom_in = F.div_mod(one, denom)
    res = F.mul_mod(z_h_eval, denom_in)
    return res  


def compute_linearisation_poly(
    domain: Radix2EvaluationDomain,
    pk,
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
    w_l_poly,
    w_r_poly,
    w_o_poly,
    w_4_poly,
    t_1_poly,
    t_2_poly,
    t_3_poly,
    t_4_poly,
    t_5_poly,
    t_6_poly,
    t_7_poly,
    t_8_poly,
    z_poly,
    z2_poly,
    f_poly,
    h1_poly,
    h2_poly,
    table_poly
):
    n = domain.size
    omega = domain.group_gen
    domain_permutation = Radix2EvaluationDomain(n)
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
    l1_eval = _compute_first_lagrange_evaluation(
        domain.size, vanishing_poly_eval, z_challenge
    )

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

    wire_evals = {
        "a_eval": a_eval,
        "b_eval": b_eval,
        "c_eval": c_eval,
        "d_eval": d_eval,
    }
    
    left_sigma_eval = F.evaluate(pk.permutations_coeffs.left_sigma, z_challenge)
    right_sigma_eval = F.evaluate(pk.permutations_coeffs.right_sigma, z_challenge)
    out_sigma_eval = F.evaluate(pk.permutations_coeffs.out_sigma, z_challenge)
    permutation_eval = F.evaluate(z_poly, shifted_z_challenge)

    perm_evals = PermutationEvaluations(
        left_sigma_eval, right_sigma_eval, out_sigma_eval, permutation_eval
    )

    # Arith selector evaluation
    q_arith_eval = F.evaluate(pk.arithmetics_coeffs.q_arith, z_challenge)

    # Lookup selector evaluation
    q_lookup_eval = F.evaluate(pk.lookups_coeffs.q_lookup, z_challenge)

    # Custom gate evaluations
    q_c_eval = F.evaluate(pk.arithmetics_coeffs.q_c, z_challenge)
    q_l_eval = F.evaluate(pk.arithmetics_coeffs.q_l, z_challenge)
    q_r_eval = F.evaluate(pk.arithmetics_coeffs.q_r, z_challenge)
    a_next_eval = F.evaluate(w_l_poly, shifted_z_challenge)
    b_next_eval = F.evaluate(w_r_poly, shifted_z_challenge)
    d_next_eval = F.evaluate(w_4_poly, shifted_z_challenge)

    # High degree selector evaluations
    q_hl_eval = F.evaluate(pk.arithmetics_coeffs.q_hl, z_challenge)
    q_hr_eval = F.evaluate(pk.arithmetics_coeffs.q_hr, z_challenge)
    q_h4_eval = F.evaluate(pk.arithmetics_coeffs.q_h4, z_challenge)

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
        pk,
        pk.arithmetics_coeffs
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
        pk.lookups_coeffs.q_lookup,
    )

    permutation = compute_linearisation_permutation(
        z_challenge,
        (alpha.to("cuda"), beta.to("cuda"), gamma.to("cuda")),
        (a_eval, b_eval, c_eval, d_eval),
        (left_sigma_eval, right_sigma_eval, out_sigma_eval),
        permutation_eval,
        z_poly,
        domain_permutation,
        pk.permutations_coeffs.fourth_sigma,
    )

    z_challenge_to_n = z_challenge_to_n.to("cuda")
    # Calculate t_8_poly * z_challenge_to_n
    term_1 = F.mul_mod_scalar(t_8_poly, z_challenge_to_n)

    # Calculate (term_1 + t_7_poly) * z_challenge_to_n

    term_2_1 = poly_add_poly(term_1, t_7_poly)
    term_2 = F.mul_mod_scalar(term_2_1, z_challenge_to_n)

    # Calculate (term_2 + t_6_poly) * z_challenge_to_n
    term_3_1 = poly_add_poly(term_2, t_6_poly)
    term_3 = F.mul_mod_scalar(term_3_1, z_challenge_to_n)

    # Calculate (term_3 + t_5_poly) * z_challenge_to_n
    term_4_1 = poly_add_poly(term_3, t_5_poly)
    term_4 = F.mul_mod_scalar(term_4_1, z_challenge_to_n)

    # Calculate (term_4 + t_4_poly) * z_challenge_to_n
    term_5_1 = poly_add_poly(term_4, t_4_poly)
    term_5 = F.mul_mod_scalar(term_5_1, z_challenge_to_n)

    # Calculate (term_5 + t_3_poly) * z_challenge_to_n
    term_6_1 = poly_add_poly(term_5, t_3_poly)
    term_6 = F.mul_mod_scalar(term_6_1, z_challenge_to_n)

    # Calculate (term_6 + t_2_poly) * z_challenge_to_n
    term_7_1 = poly_add_poly(term_6, t_2_poly)
    term_7 = F.mul_mod_scalar(term_7_1, z_challenge_to_n)

    # Calculate (term_7 + t_1_poly) * vanishing_poly_eval
    term_8_1 = poly_add_poly(term_7, t_1_poly)
    vanishing_poly_eval = vanishing_poly_eval.to("cuda")
    quotient_term = F.mul_mod_scalar(term_8_1, vanishing_poly_eval)

    neg_one = neg_one.to("cuda")
    negative_quotient_term = F.mul_mod_scalar(quotient_term, neg_one)
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
    wire_evals,
    q_arith_eval,
    custom_evals,
    pk,
    prover_key_arithmetic,
):
    wit_vals = WitnessValues(
        a_val=wire_evals["a_eval"],
        b_val=wire_evals["b_eval"],
        c_val=wire_evals["c_eval"],
        d_val=wire_evals["d_eval"],
    )

    arithmetic = compute_linearisation_arithmetic(
        wire_evals,
        q_arith_eval,
        prover_key_arithmetic,
    )

    range_separation_challenge = range_separation_challenge.to("cuda")
    range = range_constraint.linearisation_term(
        pk.selectors_coeffs.range,
        range_separation_challenge,
        wit_vals,
        custom_evals,
    )

    logic_separation_challenge = logic_separation_challenge.to("cuda")
    logic = logic_constraint.linearisation_term(
        pk.selectors_coeffs.logic,
        logic_separation_challenge,
        wit_vals,
        custom_evals,
    )

    fixed_base_separation_challenge = fixed_base_separation_challenge.to("cuda")
    fixed_base_scalar_mul = FBSMGate.linearisation_term(
        pk.selectors_coeffs.fixed_group_add,
        fixed_base_separation_challenge,
        wit_vals,
        FBSMValues.from_evaluations(custom_evals),
    )

    var_base_separation_challenge = var_base_separation_challenge.to("cuda")
    curve_addition = CAGate.linearisation_term(
        pk.selectors_coeffs.variable_group_add,
        var_base_separation_challenge,
        wit_vals,
        custom_evals,
    )

    print('arithmetic', arithmetic)
    print('range', range)
    print('logic', logic)
    print('fixed_base_scalar_mul', fixed_base_scalar_mul)
    print('curve_addition', curve_addition)

    mid1 = poly_add_poly(arithmetic, range)
    mid2 = poly_add_poly(mid1, logic)
    mid3 = poly_add_poly(mid2, fixed_base_scalar_mul)
    res = poly_add_poly(mid3, curve_addition)

    print('mid1', mid1)
    print('mid2', mid2)
    print('mid3', mid3)


    return res
