from dataclasses import dataclass
from ....bls12_381 import fr
from typing import List, Tuple
from ....domain import Radix2EvaluationDomain
from ....arithmetic import poly_mul_const,poly_add_poly
from ....plonk_core.src.permutation.constants import K1,K2,K3
import torch.nn.functional as F
@dataclass
class Permutation:
    # Left Permutation
    left_sigma: Tuple[List[fr.Fr],List[fr.Fr]]

    # Right Permutation
    right_sigma: Tuple[List[fr.Fr],List[fr.Fr]]

    # Output Permutation
    out_sigma: Tuple[List[fr.Fr],List[fr.Fr]]

    # Fourth Permutation
    fourth_sigma: Tuple[List[fr.Fr],List[fr.Fr]]

    # Linear Evaluations
    linear_evaluations: List[fr.Fr]

# Computes the following:
# (a(x) + beta * X + gamma) (b(X) + beta * k1 * X + gamma) (c(X) + beta *
# k2 * X + gamma)(d(X) + beta * k3 * X + gamma)z(X) * alpha
def compute_quotient_identity_range_check_i(
    x,
    w_l_i, w_r_i, w_o_i, w_4_i,
    z_i, alpha, beta ,gamma):

    k1 = K1()
    k2 = K2()
    k3 = K3()

    #single scalar OP on CPU
    mid2_1_1 = F.mul_mod(beta, k1)
    mid3_1_1 = F.mul_mod(beta, k2)
    mid4_1_1 = F.mul_mod(beta, k3)

    alpha = alpha.to("cuda")
    beta = beta.to("cuda")
    gamma = gamma.to("cuda")

    mid1_1 = F.mul_mod_scalar(x, beta)
    mid1_2 = F.add_mod(w_l_i, mid1_1)
    mid1 = F.add_mod_scalar(mid1_2, gamma)

    mid2_1 = F.mul_mod_scalar(x, mid2_1_1.to("cuda"))
    mid2_2 = F.add_mod(w_r_i, mid2_1)
    mid2 = F.add_mod_scalar(mid2_2, gamma)

    mid3_1 = F.mul_mod_scalar(x, mid3_1_1.to("cuda"))
    mid3_2 = F.add_mod(w_o_i, mid3_1)
    mid3 = F.add_mod_scalar(mid3_2, gamma)

    mid4_1 = F.mul_mod_scalar(x, mid4_1_1.to("cuda"))
    mid4_2 = F.add_mod(w_4_i, mid4_1)
    mid4 = F.add_mod_scalar(mid4_2, gamma)

    mid5 = F.mul_mod(mid1, mid2)
    mid6 = F.mul_mod(mid5, mid3)
    mid7 = F.mul_mod(mid6, mid4)
    res = F.mul_mod(mid7, z_i)
    res = F.mul_mod_scalar(res, alpha)
    return res

# Computes the following:
# (a(x) + beta* Sigma1(X) + gamma) (b(X) + beta * Sigma2(X) + gamma) (c(X)
# + beta * Sigma3(X) + gamma)(d(X) + beta * Sigma4(X) + gamma) Z(X.omega) *
# alpha
def compute_quotient_copy_range_check_i(
    size,
    pk_left_sigma_evals,
    pk_right_sigma_evals,
    pk_out_sigma_evals,
    pk_fourth_sigma_evals,
    w_l_i,
    w_r_i,
    w_o_i,
    w_4_i,
    z_i_next,
    alpha,
    beta,
    gamma,
):
    alpha = alpha.to("cuda")
    beta = beta.to("cuda")
    gamma = gamma.to("cuda")

    mid1_1 = F.mul_mod_scalar(pk_left_sigma_evals, beta)
    mid1_2 = F.add_mod(w_l_i, mid1_1)
    mid1 = F.add_mod_scalar(mid1_2, gamma)

    mid2_1 = F.mul_mod_scalar(pk_right_sigma_evals, beta)
    mid2_2 = F.add_mod(w_r_i, mid2_1)
    mid2 = F.add_mod_scalar(mid2_2, gamma)

    mid3_1 = F.mul_mod_scalar(pk_out_sigma_evals, beta)
    mid3_2 = F.add_mod(w_o_i, mid3_1)
    mid3 = F.add_mod_scalar(mid3_2, gamma)

    mid4_1 = F.mul_mod_scalar(pk_fourth_sigma_evals, beta)
    mid4_2 = F.add_mod(w_4_i, mid4_1)
    mid4 = F.add_mod_scalar(mid4_2, gamma)

    mid5 = F.mul_mod(mid1, mid2)
    mid5 = F.mul_mod(mid5, mid3)
    mid5 = F.mul_mod(mid5, mid4)
    mid5 = F.mul_mod(mid5, z_i_next)
    product = F.mul_mod_scalar(mid5, alpha)    

    extend_mod = F.repeat_to_poly(fr.MODULUS().to("cuda"), size)
    res = F.sub_mod(extend_mod, product)
    return res

# Computes the following:
# L_1(X)[Z(X) - 1]
def compute_quotient_term_check_one_i(z_i, l1_alpha_sq):
    one = fr.one()
    z_i_sub_one = F.sub_mod_scalar(z_i, one.to("cuda"))
    res = F.mul_mod(z_i_sub_one, l1_alpha_sq)
    return res

# Computes the permutation term of the linearisation polynomial.
def compute_linearisation_permutation(
    z_challenge, 
    challengTuple, 
    wireTuple, 
    sigmaTuple, 
    z_eval, z_poly, domain,
    pk_fourth_sigma_coeff):
    
    a = compute_lineariser_identity_range_check(
        wireTuple[0],wireTuple[1],wireTuple[2],wireTuple[3],
        z_challenge,
        challengTuple[0],challengTuple[1],challengTuple[2],
        z_poly
    )

    mod = fr.MODULUS().to("cuda")
    b = compute_lineariser_copy_range_check(
        mod,
        wireTuple[0], wireTuple[1], wireTuple[2],
        z_eval,
        sigmaTuple[0],sigmaTuple[1],sigmaTuple[2],
        challengTuple[0],challengTuple[1],challengTuple[2],
        pk_fourth_sigma_coeff
    )
    
    alpha2 = F.mul_mod(challengTuple[0], challengTuple[0])
    alpha2 = alpha2.to('cuda')
    c = compute_lineariser_check_is_one(
        domain,
        z_challenge,
        alpha2,
        z_poly
    )
    ab = poly_add_poly(a,b)
    abc = poly_add_poly(ab,c)
    return abc

# Computes the following:
# -(a_eval + beta * sigma_1 + gamma)(b_eval + beta * sigma_2 + gamma)
# (c_eval + beta * sigma_3 + gamma) * beta *z_eval * alpha^2 * Sigma_4(X)
def compute_lineariser_copy_range_check(
    mod,
    a_eval: fr.Fr, b_eval: fr.Fr, c_eval: fr.Fr,
    z_eval: fr.Fr,
    sigma_1_eval: fr.Fr,
    sigma_2_eval: fr.Fr,
    sigma_3_eval: fr.Fr,
    alpha: fr.Fr, beta: fr.Fr, gamma: fr.Fr,
    fourth_sigma_poly,
):
    # a_eval + beta * sigma_1 + gamma
    beta_sigma_1 = F.mul_mod(beta, sigma_1_eval)
    a_0 = F.add_mod(a_eval, beta_sigma_1)
    a_0 = F.add_mod(a_0, gamma)

    # b_eval + beta * sigma_2 + gamma
    beta_sigma_2 = F.mul_mod(beta, sigma_2_eval)
    a_1 = F.add_mod(b_eval, beta_sigma_2)
    a_1 = F.add_mod(a_1, gamma)

    # c_eval + beta * sigma_3 + gamma
    beta_sigma_3 = F.mul_mod(beta, sigma_3_eval)
    a_2 = F.add_mod(c_eval, beta_sigma_3)
    a_2 = F.add_mod(a_2, gamma)

    beta_z_eval = F.mul_mod(beta, z_eval)
    a = F.mul_mod(a_0, a_1)
    a = F.mul_mod(a, a_2)
    a = F.mul_mod(a, beta_z_eval)
    a = F.mul_mod(a, alpha)
    neg_a = F.sub_mod(mod, a)

    res = poly_mul_const(fourth_sigma_poly,neg_a)
    return res

# Computes the following:
# (a_eval + beta * z_challenge + gamma)(b_eval + beta * K1 * z_challenge +
# gamma)(c_eval + beta * K2 * z_challenge + gamma) * alpha z(X)
def compute_lineariser_identity_range_check(
    a_eval: fr.Fr, b_eval: fr.Fr, c_eval: fr.Fr, d_eval: fr.Fr,
    z_challenge: fr.Fr,
    alpha: fr.Fr, beta: fr.Fr, gamma: fr.Fr,
    z_poly
):
    beta_z = F.mul_mod(beta, z_challenge)
    # a_eval + beta * z_challenge + gamma
    a_0 = F.add_mod(a_eval, beta_z)
    a_0 = F.add_mod(a_0, gamma)

    # b_eval + beta * K1 * z_challenge + gamma
    k1 = K1().to('cuda')
    beta_z_k1 = F.mul_mod(k1, beta_z)
    a_1 = F.add_mod(b_eval, beta_z_k1)
    a_1 = F.add_mod(a_1, gamma)

    # c_eval + beta * K2 * z_challenge + gamma
    k2 = K2().to('cuda')
    beta_z_k2 = F.mul_mod(k2, beta_z)
    a_2 = F.add_mod(c_eval, beta_z_k2)
    a_2 = F.add_mod(a_2, gamma)

    # d_eval + beta * K3 * z_challenge + gamma
    k3 = K3().to('cuda')
    beta_z_k3 = F.mul_mod(k3, beta_z)
    a_3 = F.add_mod(d_eval, beta_z_k3)
    a_3 = F.add_mod(a_3, gamma)

    a = F.mul_mod(a_0, a_1)
    a = F.mul_mod(a, a_2)
    a = F.mul_mod(a, a_3)
    a = F.mul_mod(a, alpha)
    res = poly_mul_const(z_poly, a)
    return res

# Computes the following:
# L_1(X)[Z(X) - 1]
def compute_lineariser_check_is_one(
    domain: Radix2EvaluationDomain, 
    z_challenge: fr.Fr, 
    alpha_sq: fr.Fr, 
    z_coeffs: List[fr.Fr]):

    lagrange_coefficients = domain.evaluate_all_lagrange_coefficients(z_challenge)
    l_1_z = lagrange_coefficients[0]
    const = F.mul_mod(l_1_z, alpha_sq)
    res = poly_mul_const(z_coeffs, const)
    return res


def permutation_compute_quotient(
        size,
        pk_linear_evaluations_evals,
        pk_left_sigma_evals,
        pk_right_sigma_evals,
        pk_out_sigma_evals,
        pk_fourth_sigma_evals,
        w_l_i, w_r_i, w_o_i, w_4_i,
        z_i, z_i_next,
        alpha, l1_alpha_sq,
        beta, gamma):

        a = compute_quotient_identity_range_check_i(
          pk_linear_evaluations_evals, w_l_i, w_r_i, w_o_i, w_4_i, z_i, alpha, beta, gamma
        )
        b = compute_quotient_copy_range_check_i(
            size,
            pk_left_sigma_evals,
            pk_right_sigma_evals,
            pk_out_sigma_evals,
            pk_fourth_sigma_evals,  w_l_i, w_r_i, w_o_i, w_4_i, z_i_next, alpha, beta, gamma,
        )
        c = compute_quotient_term_check_one_i(z_i, l1_alpha_sq)

        res =F.add_mod(a,b)
        res= F.add_mod(res,c)
        return res
    
   