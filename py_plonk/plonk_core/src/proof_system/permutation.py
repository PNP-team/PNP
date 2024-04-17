from dataclasses import dataclass
from ....bls12_381 import fr
from typing import List, Tuple
from ....domain import Radix2EvaluationDomain
from ....arithmetic import poly_mul_const,poly_add_poly,from_list_gmpy,from_list_tensor,from_tensor_list,from_gmpy_list,from_gmpy_list_1,neg
from ....plonk_core.src.permutation.constants import K1,K2,K3
import torch.nn.functional as F
import torch
import copy
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

    def compute_quotient_i(self, index,
        w_l_i: fr.Fr, w_r_i: fr.Fr, w_o_i: fr.Fr, w_4_i: fr.Fr,
        z_i: fr.Fr, z_i_next: fr.Fr,
        alpha: fr.Fr, l1_alpha_sq: fr.Fr,
        beta: fr.Fr, gamma: fr.Fr):

        a = self.compute_quotient_identity_range_check_i(
            index, w_l_i, w_r_i, w_o_i, w_4_i, z_i, alpha, beta, gamma,
        )
        b = self.compute_quotient_copy_range_check_i(
            index, w_l_i, w_r_i, w_o_i, w_4_i, z_i_next, alpha, beta, gamma,
        )
        c = self.compute_quotient_term_check_one_i(z_i, l1_alpha_sq)

        res =F.add_mod(a,b)
        res= F.add_mod(res,c)
        return res
    
    # Computes the following:
    # (a(x) + beta * X + gamma) (b(X) + beta * k1 * X + gamma) (c(X) + beta *
    # k2 * X + gamma)(d(X) + beta * k3 * X + gamma)z(X) * alpha
    def compute_quotient_identity_range_check_i(
        self,index,
        w_l_i: fr.Fr,w_r_i: fr.Fr,w_o_i: fr.Fr,w_4_i: fr.Fr,
        z_i: fr.Fr,alpha: fr.Fr,beta: fr.Fr,gamma: fr.Fr,):

        x = self.linear_evaluations[index]
        # x=torch.tensor(from_gmpy_list_1(x),dtype=torch.BLS12_381_Fr_G1_Mont)
        k1 = K1()
        k2 = K2()
        k3 = K3()

        k1= torch.tensor(from_gmpy_list_1(k1),dtype=torch.BLS12_381_Fr_G1_Mont)
        k2= torch.tensor(from_gmpy_list_1(k2),dtype=torch.BLS12_381_Fr_G1_Mont)
        k3= torch.tensor(from_gmpy_list_1(k3),dtype=torch.BLS12_381_Fr_G1_Mont)
        mid1_1 = F.mul_mod(beta, x)
        mid1_2 = F.add_mod(w_l_i, mid1_1)
        mid1 = F.add_mod(mid1_2, gamma)

        mid2_1_1 = F.mul_mod(beta, k1)
        mid2_1 = F.mul_mod(mid2_1_1, x)
        mid2_2 = F.add_mod(w_r_i, mid2_1)
        mid2 = F.add_mod(mid2_2, gamma)

        mid3_1_1 = F.mul_mod(beta, k2)
        mid3_1 = F.mul_mod(mid3_1_1, x)
        mid3_2 = F.add_mod(w_o_i, mid3_1)
        mid3 = F.add_mod(mid3_2, gamma)

        mid4_1_1 = F.mul_mod(beta, k3)
        mid4_1 = F.mul_mod(mid4_1_1, x)
        mid4_2 = F.add_mod(w_4_i, mid4_1)
        mid4 = F.add_mod(mid4_2, gamma)

        mid5 = F.mul_mod(mid1, mid2)
        mid6 = F.mul_mod(mid5, mid3)
        mid7 = F.mul_mod(mid6, mid4)
        res = F.mul_mod(mid7, z_i)
        res = F.mul_mod(res, alpha)

        return res

    # Computes the following:
    # (a(x) + beta* Sigma1(X) + gamma) (b(X) + beta * Sigma2(X) + gamma) (c(X)
    # + beta * Sigma3(X) + gamma)(d(X) + beta * Sigma4(X) + gamma) Z(X.omega) *
    # alpha
    def compute_quotient_copy_range_check_i(
        self,index,
        w_l_i: fr.Fr,
        w_r_i: fr.Fr,
        w_o_i: fr.Fr,
        w_4_i: fr.Fr,
        z_i_next: fr.Fr,
        alpha: fr.Fr,
        beta: fr.Fr,
        gamma: fr.Fr
    ):
        
        # left_sigma_eval = torch.tensor(from_gmpy_list_1(self.left_sigma[1][index]),dtype=torch.BLS12_381_Fr_G1_Mont)
        # right_sigma_eval = torch.tensor(from_gmpy_list_1(self.right_sigma[1][index]),dtype=torch.BLS12_381_Fr_G1_Mont)
        # out_sigma_eval = torch.tensor(from_gmpy_list_1(self.out_sigma[1][index]),dtype=torch.BLS12_381_Fr_G1_Mont)
        # fourth_sigma_eval = torch.tensor(from_gmpy_list_1(self.fourth_sigma[1][index]),dtype=torch.BLS12_381_Fr_G1_Mont)

        mid1_1 = F.mul_mod(beta, self.left_sigma[1][index])
        mid1_2 = F.add_mod(w_l_i, mid1_1)
        mid1 = F.add_mod(mid1_2, gamma)

        mid2_1 = F.mul_mod(beta, self.right_sigma[1][index])
        mid2_2 = F.add_mod(w_r_i, mid2_1)
        mid2 = F.add_mod(mid2_2, gamma)

        mid3_1 = F.mul_mod(beta, self.out_sigma[1][index])
        mid3_2 = F.add_mod(w_o_i, mid3_1)
        mid3 = F.add_mod(mid3_2, gamma)

        mid4_1 = F.mul_mod(beta, self.fourth_sigma[1][index])
        mid4_2 = F.add_mod(w_4_i, mid4_1)
        mid4 = F.add_mod(mid4_2, gamma)

        mid5 = F.mul_mod(mid1, mid2)
        mid5 = F.mul_mod(mid5, mid3)
        mid5 = F.mul_mod(mid5, mid4)
        mid5 = F.mul_mod(mid5, z_i_next)
        product = F.mul_mod(mid5, alpha)         
        res = neg(product)
        return res

    # Computes the following:
    # L_1(X)[Z(X) - 1]
    def compute_quotient_term_check_one_i(self, z_i: fr.Fr, l1_alpha_sq: fr.Fr):
        one = torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont)
        z_i_sub_one = F.sub_mod(z_i, one)
        res = F.mul_mod(z_i_sub_one, l1_alpha_sq)

        return res
    
    # Computes the permutation term of the linearisation polynomial.
    def compute_linearisation(
        self, 
        n, 
        z_challenge: fr.Fr, 
        challengTuple: Tuple[fr.Fr,fr.Fr,fr.Fr], 
        wireTuple: Tuple[fr.Fr,fr.Fr,fr.Fr,fr.Fr], 
        sigmaTuple: Tuple[fr.Fr,fr.Fr,fr.Fr], 
        z_eval, z_poly,domain):
        a = self.compute_lineariser_identity_range_check(
            wireTuple[0],wireTuple[1],wireTuple[2],wireTuple[3],
            z_challenge,
            challengTuple[0],challengTuple[1],challengTuple[2],
            z_poly
        )
        # from_gmpy_list(self.fourth_sigma[0])
        # self_fourth_sigma0=from_list_tensor(self.fourth_sigma[0])
        b = self.compute_lineariser_copy_range_check(
            wireTuple[0], wireTuple[1], wireTuple[2],
            z_eval,
            sigmaTuple[0],sigmaTuple[1],sigmaTuple[2],
            challengTuple[0],challengTuple[1],challengTuple[2],
            self.fourth_sigma[0]
        )
        alpha2 = challengTuple[0].square()
        alpha2=torch.tensor(from_gmpy_list_1(alpha2),dtype=torch.BLS12_381_Fr_G1_Mont)
        c = self.compute_lineariser_check_is_one(
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
        self,
        a_eval: fr.Fr, b_eval: fr.Fr, c_eval: fr.Fr,
        z_eval: fr.Fr,
        sigma_1_eval: fr.Fr,
        sigma_2_eval: fr.Fr,
        sigma_3_eval: fr.Fr,
        alpha: fr.Fr, beta: fr.Fr, gamma: fr.Fr,
        fourth_sigma_poly: List[fr.Fr],
    ):
        beta=torch.tensor(from_gmpy_list_1(beta),dtype=torch.BLS12_381_Fr_G1_Mont)
        gamma=torch.tensor(from_gmpy_list_1(gamma),dtype=torch.BLS12_381_Fr_G1_Mont)
        alpha=torch.tensor(from_gmpy_list_1(alpha),dtype=torch.BLS12_381_Fr_G1_Mont)
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
        neg_a = neg(a)


        res = poly_mul_const(fourth_sigma_poly,neg_a)
        return res
    
    # Computes the following:
    # (a_eval + beta * z_challenge + gamma)(b_eval + beta * K1 * z_challenge +
    # gamma)(c_eval + beta * K2 * z_challenge + gamma) * alpha z(X)
    def compute_lineariser_identity_range_check(
        self,
        a_eval: fr.Fr, b_eval: fr.Fr, c_eval: fr.Fr, d_eval: fr.Fr,
        z_challenge: fr.Fr,
        alpha: fr.Fr, beta: fr.Fr, gamma: fr.Fr,
        z_poly: List[fr.Fr]
    ):
        beta=torch.tensor(from_gmpy_list_1(beta),dtype=torch.BLS12_381_Fr_G1_Mont)
        gamma=torch.tensor(from_gmpy_list_1(gamma),dtype=torch.BLS12_381_Fr_G1_Mont)
        alpha=torch.tensor(from_gmpy_list_1(alpha),dtype=torch.BLS12_381_Fr_G1_Mont)
        beta_z = F.mul_mod(beta, z_challenge)
        # a_eval + beta * z_challenge + gamma
        a_0 = F.add_mod(a_eval, beta_z)
        a_0 = F.add_mod(a_0, gamma)

        # b_eval + beta * K1 * z_challenge + gamma
        k1=torch.tensor(from_gmpy_list_1(K1()),dtype=torch.BLS12_381_Fr_G1_Mont)
        beta_z_k1 = F.mul_mod(k1,beta_z)
        a_1 = F.add_mod(b_eval, beta_z_k1)
        a_1 = F.add_mod(a_1, gamma)

        # c_eval + beta * K2 * z_challenge + gamma
        k2=torch.tensor(from_gmpy_list_1(K2()),dtype=torch.BLS12_381_Fr_G1_Mont)
        beta_z_k2 = F.mul_mod(k2,beta_z)
        a_2 = F.add_mod(c_eval, beta_z_k2)
        a_2 = F.add_mod(a_2, gamma)

        # d_eval + beta * K3 * z_challenge + gamma
        k3=torch.tensor(from_gmpy_list_1(K3()),dtype=torch.BLS12_381_Fr_G1_Mont)
        beta_z_k3 = F.mul_mod(k3,beta_z)
        a_3 = F.add_mod(d_eval, beta_z_k3)
        a_3 = F.add_mod(a_3, gamma)

        a = F.mul_mod(a_0, a_1)
        a = F.mul_mod(a, a_2)
        a = F.mul_mod(a, a_3)
        a = F.mul_mod(a, alpha)
        res = poly_mul_const(z_poly, a)
        return res

    
    def compute_lineariser_check_is_one(
        self, 
        domain: Radix2EvaluationDomain, 
        z_challenge: fr.Fr, 
        alpha_sq: fr.Fr, 
        z_coeffs: List[fr.Fr]):

        lagrange_coefficients = domain.evaluate_all_lagrange_coefficients(z_challenge)
        l_1_z = lagrange_coefficients[0]
        const =F.mul_mod(l_1_z,alpha_sq)
        res = poly_mul_const(z_coeffs,const)
        return res


# Computes the following:
# (a(x) + beta * X + gamma) (b(X) + beta * k1 * X + gamma) (c(X) + beta *
# k2 * X + gamma)(d(X) + beta * k3 * X + gamma)z(X) * alpha
def compute_quotient_identity_range_check_i(
    index,
    x,
    w_l_i: fr.Fr,w_r_i: fr.Fr,w_o_i: fr.Fr,w_4_i: fr.Fr,
    z_i: fr.Fr,alpha: fr.Fr,beta: fr.Fr,gamma: fr.Fr,):


    # x=torch.tensor(from_gmpy_list_1(x),dtype=torch.BLS12_381_Fr_G1_Mont)
    k1 = K1()
    k2 = K2()
    k3 = K3()

    k1= torch.tensor(from_gmpy_list_1(k1),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    k2= torch.tensor(from_gmpy_list_1(k2),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    k3= torch.tensor(from_gmpy_list_1(k3),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    mid1_1 = F.mul_mod(beta, x[index])
    mid1_2 = F.add_mod(w_l_i, mid1_1)
    mid1 = F.add_mod(mid1_2, gamma)

    mid2_1_1 = F.mul_mod(beta, k1)
    mid2_1 = F.mul_mod(mid2_1_1, x[index])
    mid2_2 = F.add_mod(w_r_i, mid2_1)
    mid2 = F.add_mod(mid2_2, gamma)

    mid3_1_1 = F.mul_mod(beta, k2)
    mid3_1 = F.mul_mod(mid3_1_1, x[index])
    mid3_2 = F.add_mod(w_o_i, mid3_1)
    mid3 = F.add_mod(mid3_2, gamma)

    mid4_1_1 = F.mul_mod(beta, k3)
    mid4_1 = F.mul_mod(mid4_1_1,x[index])
    mid4_2 = F.add_mod(w_4_i, mid4_1)
    mid4 = F.add_mod(mid4_2, gamma)

    mid5 = F.mul_mod(mid1, mid2)
    mid6 = F.mul_mod(mid5, mid3)
    mid7 = F.mul_mod(mid6, mid4)
    res = F.mul_mod(mid7, z_i)
    res = F.mul_mod(res, alpha)

    return res

# Computes the following:
# (a(x) + beta* Sigma1(X) + gamma) (b(X) + beta * Sigma2(X) + gamma) (c(X)
# + beta * Sigma3(X) + gamma)(d(X) + beta * Sigma4(X) + gamma) Z(X.omega) *
# alpha
def compute_quotient_copy_range_check_i(
    pk_left_sigma_evals,
    pk_right_sigma_evals,
    pk_out_sigma_evals,
    pk_fourth_sigma_evals,
    index,
    w_l_i: fr.Fr,
    w_r_i: fr.Fr,
    w_o_i: fr.Fr,
    w_4_i: fr.Fr,
    z_i_next: fr.Fr,
    alpha: fr.Fr,
    beta: fr.Fr,
    gamma: fr.Fr
):
    
    # left_sigma_eval = torch.tensor(from_gmpy_list_1(self.left_sigma[1][index]),dtype=torch.BLS12_381_Fr_G1_Mont)
    # right_sigma_eval = torch.tensor(from_gmpy_list_1(self.right_sigma[1][index]),dtype=torch.BLS12_381_Fr_G1_Mont)
    # out_sigma_eval = torch.tensor(from_gmpy_list_1(self.out_sigma[1][index]),dtype=torch.BLS12_381_Fr_G1_Mont)
    # fourth_sigma_eval = torch.tensor(from_gmpy_list_1(self.fourth_sigma[1][index]),dtype=torch.BLS12_381_Fr_G1_Mont)

    mid1_1 = F.mul_mod(beta, pk_left_sigma_evals[index])
    mid1_2 = F.add_mod(w_l_i, mid1_1)
    mid1 = F.add_mod(mid1_2, gamma)

    mid2_1 = F.mul_mod(beta, pk_right_sigma_evals[index])
    mid2_2 = F.add_mod(w_r_i, mid2_1)
    mid2 = F.add_mod(mid2_2, gamma)

    mid3_1 = F.mul_mod(beta, pk_out_sigma_evals[index])
    mid3_2 = F.add_mod(w_o_i, mid3_1)
    mid3 = F.add_mod(mid3_2, gamma)

    mid4_1 = F.mul_mod(beta, pk_fourth_sigma_evals[index])
    mid4_2 = F.add_mod(w_4_i, mid4_1)
    mid4 = F.add_mod(mid4_2, gamma)

    mid5 = F.mul_mod(mid1, mid2)
    mid5 = F.mul_mod(mid5, mid3)
    mid5 = F.mul_mod(mid5, mid4)
    mid5 = F.mul_mod(mid5, z_i_next)
    product = F.mul_mod(mid5, alpha)         
    res = neg(product)
    return res

# Computes the following:
# L_1(X)[Z(X) - 1]
def compute_quotient_term_check_one_i(z_i: fr.Fr, l1_alpha_sq: fr.Fr):
    one = torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    z_i_sub_one = F.sub_mod(z_i, one)
    res = F.mul_mod(z_i_sub_one, l1_alpha_sq)

    return res

# Computes the permutation term of the linearisation polynomial.
def compute_linearisation_permutation(
    n, 
    z_challenge: fr.Fr, 
    challengTuple: Tuple[fr.Fr,fr.Fr,fr.Fr], 
    wireTuple: Tuple[fr.Fr,fr.Fr,fr.Fr,fr.Fr], 
    sigmaTuple: Tuple[fr.Fr,fr.Fr,fr.Fr], 
    z_eval, z_poly,domain,
    pk_fourth_sigma_coeff):
    z_challenge=z_challenge.to('cuda')
    a = compute_lineariser_identity_range_check(
        wireTuple[0],wireTuple[1],wireTuple[2],wireTuple[3],
        z_challenge,
        challengTuple[0],challengTuple[1],challengTuple[2],
        z_poly
    )
    b = compute_lineariser_copy_range_check(
        wireTuple[0], wireTuple[1], wireTuple[2],
        z_eval,
        sigmaTuple[0],sigmaTuple[1],sigmaTuple[2],
        challengTuple[0],challengTuple[1],challengTuple[2],
        pk_fourth_sigma_coeff
    )
    alpha2 = challengTuple[0].square()
    alpha2=torch.tensor(from_gmpy_list_1(alpha2),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
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
    a_eval: fr.Fr, b_eval: fr.Fr, c_eval: fr.Fr,
    z_eval: fr.Fr,
    sigma_1_eval: fr.Fr,
    sigma_2_eval: fr.Fr,
    sigma_3_eval: fr.Fr,
    alpha: fr.Fr, beta: fr.Fr, gamma: fr.Fr,
    fourth_sigma_poly: List[fr.Fr],
):
    beta=torch.tensor(from_gmpy_list_1(beta),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    gamma=torch.tensor(from_gmpy_list_1(gamma),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    alpha=torch.tensor(from_gmpy_list_1(alpha),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
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
    neg_a = neg(a)


    res = poly_mul_const(fourth_sigma_poly,neg_a)
    return res

# Computes the following:
# (a_eval + beta * z_challenge + gamma)(b_eval + beta * K1 * z_challenge +
# gamma)(c_eval + beta * K2 * z_challenge + gamma) * alpha z(X)
def compute_lineariser_identity_range_check(
    a_eval: fr.Fr, b_eval: fr.Fr, c_eval: fr.Fr, d_eval: fr.Fr,
    z_challenge: fr.Fr,
    alpha: fr.Fr, beta: fr.Fr, gamma: fr.Fr,
    z_poly: List[fr.Fr]
):
    beta=torch.tensor(from_gmpy_list_1(beta),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    gamma=torch.tensor(from_gmpy_list_1(gamma),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    alpha=torch.tensor(from_gmpy_list_1(alpha),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    beta_z = F.mul_mod(beta, z_challenge)
    # a_eval + beta * z_challenge + gamma
    a_0 = F.add_mod(a_eval, beta_z)
    a_0 = F.add_mod(a_0, gamma)

    # b_eval + beta * K1 * z_challenge + gamma
    k1=torch.tensor(from_gmpy_list_1(K1()),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    beta_z_k1 = F.mul_mod(k1,beta_z)
    a_1 = F.add_mod(b_eval, beta_z_k1)
    a_1 = F.add_mod(a_1, gamma)

    # c_eval + beta * K2 * z_challenge + gamma
    k2=torch.tensor(from_gmpy_list_1(K2()),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    beta_z_k2 = F.mul_mod(k2,beta_z)
    a_2 = F.add_mod(c_eval, beta_z_k2)
    a_2 = F.add_mod(a_2, gamma)

    # d_eval + beta * K3 * z_challenge + gamma
    k3=torch.tensor(from_gmpy_list_1(K3()),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    beta_z_k3 = F.mul_mod(k3,beta_z)
    a_3 = F.add_mod(d_eval, beta_z_k3)
    a_3 = F.add_mod(a_3, gamma)

    a = F.mul_mod(a_0, a_1)
    a = F.mul_mod(a, a_2)
    a = F.mul_mod(a, a_3)
    a = F.mul_mod(a, alpha)
    res = poly_mul_const(z_poly, a)
    return res


def compute_lineariser_check_is_one(
    domain: Radix2EvaluationDomain, 
    z_challenge: fr.Fr, 
    alpha_sq: fr.Fr, 
    z_coeffs: List[fr.Fr]):

    lagrange_coefficients = domain.evaluate_all_lagrange_coefficients(z_challenge)
    l_1_z = lagrange_coefficients[0]
    const =F.mul_mod(l_1_z,alpha_sq)
    res = poly_mul_const(z_coeffs,const)
    return res


def permutation_compute_quotient_i( 
            pk_linear_evaluations_evals,
            pk_left_sigma_evals,
            pk_right_sigma_evals,
            pk_out_sigma_evals,
            pk_fourth_sigma_evals, index,
        w_l_i: fr.Fr, w_r_i: fr.Fr, w_o_i: fr.Fr, w_4_i: fr.Fr,
        z_i: fr.Fr, z_i_next: fr.Fr,
        alpha: fr.Fr, l1_alpha_sq: fr.Fr,
        beta: fr.Fr, gamma: fr.Fr):

        a = compute_quotient_identity_range_check_i(
          index,pk_linear_evaluations_evals, w_l_i, w_r_i, w_o_i, w_4_i, z_i, alpha, beta, gamma,
        )
        b = compute_quotient_copy_range_check_i(
            pk_left_sigma_evals,
            pk_right_sigma_evals,
            pk_out_sigma_evals,
            pk_fourth_sigma_evals, index, w_l_i, w_r_i, w_o_i, w_4_i, z_i_next, alpha, beta, gamma,
        )
        c = compute_quotient_term_check_one_i(z_i, l1_alpha_sq)

        res =F.add_mod(a,b)
        res= F.add_mod(res,c)
        return res
    
   