from dataclasses import dataclass
from ..structure import UniversalParams,OpenProof
from ..jacobian import ProjectivePointG1
from ..bls12_381 import fr,fq
from typing import List
from ..arithmetic import skip_leading_zeros_and_convert_to_bigints,convert_to_bigints,\
                         rand_poly,poly_add_poly_mul_const,MSM,calculate_execution_time
from ..plonk_core.src.proof_system.linearisation_poly import ProofEvaluations
import random
import torch 
import torch.nn.functional as F
from ..jacobian import to_affine,add_assign_mixed,add_assign


#########func for randomness#########
def empty_randomness():
    return torch.tensor([], dtype = fr.TYPE())


def calculate_hiding_polynomial_degree(hiding_bound):
    return hiding_bound + 1

def push(self, a):
    self.blind_poly.append(a)

    
def randomness_rand(hiding_bound):
    hiding_poly_degree = calculate_hiding_polynomial_degree(hiding_bound)
    return rand_poly(hiding_poly_degree)

def rand_add_assign(self, f, other):
    self = poly_add_poly_mul_const(self, f, other)


def commit(powers_of_g, powers_of_gamma_g, polynomial, hiding_bound):
    plain_coeffs = skip_leading_zeros_and_convert_to_bigints(polynomial)
    commitment = MSM(powers_of_g, plain_coeffs)
    randomness = empty_randomness()
    if hiding_bound:
        randomness = randomness_rand(hiding_bound)
    random_ints = convert_to_bigints(randomness)

    random_commitment:ProjectivePointG1 = MSM(powers_of_gamma_g, random_ints)
    random_commitment_affine = to_affine(random_commitment)
    commitment = add_assign_mixed(commitment,random_commitment_affine)
    commitment_affine = to_affine(commitment)
    return commitment_affine,randomness
    
# On input a list of labeled polynomials and a query point, `open` outputs a proof of evaluation
# of the polynomials at the query point.
@calculate_execution_time
def open(
    ck: UniversalParams,
    labeled_polynomials,
    point,
    opening_challenge,
    rands,
    _rng=None
):
    
    combined_polynomial = torch.tensor([], dtype = fr.TYPE())
    combined_rand = empty_randomness()

    opening_challenge_counter = 0

    curr_challenge = opening_challenges(opening_challenge, opening_challenge_counter)
    opening_challenge_counter += 1
   
    i=0
    for (polynomial, _), rand in zip(labeled_polynomials, rands):
        combined_polynomial = poly_add_poly_mul_const(combined_polynomial, curr_challenge.to("cuda"), polynomial)  #polynomial.poly is tensor
        rand_add_assign(combined_rand, curr_challenge, rand)
        curr_challenge = opening_challenges(opening_challenge, opening_challenge_counter)
        opening_challenge_counter += 1
        i=i+1

    proof = open_proof(ck[0], ck[1], combined_polynomial.to("cuda"), point, combined_rand)
    return proof


# class LabeledCommitment:
#     def __init__(self,label,commitment):
#         # self.label = label
#         self.commitment =commitment

#     @classmethod
#     def new(cls,label,commitment):
#         return cls(label = label,commitment = commitment)

# class LabeledPoly:
#     def __init__(self, label, hiding_bound, poly):
#         self.hiding_bound = hiding_bound
#         self.poly = poly



def commit_poly_new(ck:UniversalParams, polys):
    random.seed(42)
    randomness = []
    labeled_comm = []
  
    for (polynomial, hiding_bound) in polys:
        comm,rand = commit(ck[0], ck[1], polynomial, hiding_bound) 
        labeled_comm.append(comm)
        randomness.append(rand)
        
    return labeled_comm,randomness




def opening_challenges(opening_challenge, exp):
    res = F.exp_mod(opening_challenge, exp)
    return res

# Compute witness polynomial.
#
# The witness polynomial w(x) the quotient of the division (p(x) - p(z)) / (x - z)
# Observe that this quotient does not change with z because
# p(z) is the remainder term. We can therefore omit p(z) when computing the quotient.
def compute_witness_polynomial(p: List[fr.Fr], point, randomness):
    mod = fr.MODULUS().to("cuda")
    neg_p = F.sub_mod(mod, point)
    if p.size(0) != 0:
        witness_polynomial = F.poly_div_poly(p, neg_p)
    random_witness_polynomial = None
    if len(randomness) != 0:
        random_p = randomness
        random_witness_polynomial = F.poly_div_poly(random_p, neg_p)
    return witness_polynomial, random_witness_polynomial

def open_with_witness_polynomial(
    powers_of_g,
    powers_of_gamma_g, 
    point, 
    randomness,
    witness_polynomial, 
    hiding_witness_polynomial):


    witness_coeffs =skip_leading_zeros_and_convert_to_bigints(witness_polynomial)
    w = MSM(powers_of_g, witness_coeffs)
    random_v = None
    if hiding_witness_polynomial is not None:
        blinding_p = randomness
        blinding_evaluation = F.evaluate(blinding_p, point.to("cuda"))
        random_witness_coeffs = convert_to_bigints(hiding_witness_polynomial)
        random_commit = MSM(powers_of_gamma_g,random_witness_coeffs)
        w = add_assign(w,random_commit)
        random_v = blinding_evaluation
    
    return OpenProof(to_affine(w), random_v)

# On input a polynomial `p` and a point `point`, outputs a proof for the same.
def open_proof(powers_of_g, powers_of_gamma_g, p: list, point, rand):
    witness_poly, hiding_witness_poly = compute_witness_polynomial(p, point, rand)
    proof = open_with_witness_polynomial(
            powers_of_g,
            powers_of_gamma_g,
            point,
            rand,
            witness_poly,
            hiding_witness_poly,
        )
    return proof
@dataclass
class Proof:
    # Commitment to the witness polynomial for the left wires.
    a_comm: any

    # Commitment to the witness polynomial for the right wires.
    b_comm: any

    # Commitment to the witness polynomial for the output wires.
    c_comm: any

    # Commitment to the witness polynomial for the fourth wires.
    d_comm: any

    # Commitment to the permutation polynomial.
    z_comm: any

    # Commitment to the lookup query polynomial.
    f_comm: any

    # Commitment to first half of sorted polynomial
    h_1_comm: any

    # Commitment to second half of sorted polynomial
    h_2_comm: any

    # Commitment to the lookup permutation polynomial.
    z_2_comm: any

    # Commitment to the quotient polynomial.
    t_1_comm: any

    # Commitment to the quotient polynomial.
    t_2_comm: any

    # Commitment to the quotient polynomial.
    t_3_comm: any

    # Commitment to the quotient polynomial.
    t_4_comm: any

    # Commitment to the quotient polynomial.
    t_5_comm: any

    # Commitment to the quotient polynomial.
    t_6_comm: any

    # Commitment to the quotient polynomial.
    t_7_comm: any

    # Commitment to the quotient polynomial.
    t_8_comm: any

    # Batch opening proof of the aggregated witnesses
    aw_opening: OpenProof

    # Batch opening proof of the shifted aggregated witnesses
    saw_opening: OpenProof

    # Subset of all of the evaluations added to the proof.
    evaluations: ProofEvaluations

