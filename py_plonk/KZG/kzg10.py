from dataclasses import dataclass
from ..structure import UniversalParams,OpenProof
from ..jacobian import ProjectivePointG1
from ..field import field
from ..bls12_381 import fr,fq
from typing import List
from ..arithmetic import skip_leading_zeros_and_convert_to_bigints,convert_to_bigints,\
                         rand_poly,poly_add_poly_mul_const,from_coeff_vec,MSM_new,pow_single
from ..plonk_core.src.proof_system.linearisation_poly import ProofEvaluations
import random
import torch 
import torch.nn.functional as F
from ..jacobian import to_affine,add_assign_mixed,add_assign


class Randomness:
    def __init__(self, blind_poly: List[fr.Fr]):
        self.blind_poly = blind_poly

    @classmethod
    def empty(cls):
        return cls(torch.tensor([],dtype=torch.BLS12_381_Fr_G1_Mont))

    @classmethod
    def calculate_hiding_polynomial_degree(cls, hiding_bound):
        return hiding_bound + 1

    def push(self, a):
        self.blind_poly.append(a)

    @classmethod
    def rand(cls, hiding_bound):
        hiding_poly_degree = cls.calculate_hiding_polynomial_degree(hiding_bound)
        return cls(blind_poly = rand_poly(hiding_poly_degree))
    
    def add_assign(self, f:field, other: 'Randomness'):
        self.blind_poly = poly_add_poly_mul_const(self.blind_poly, f, other.blind_poly)

class Commitment:
    def __init__(self,value):
        self.value = value
    
    @classmethod
    def commit_new(cls, powers_of_g, powers_of_gamma_g, polynomial, hiding_bound):
        num_leading_zeros, plain_coeffs = skip_leading_zeros_and_convert_to_bigints(polynomial)
        commitment = MSM_new(powers_of_g[num_leading_zeros:], plain_coeffs)
        randomness = Randomness.empty()
        if hiding_bound:
            randomness = Randomness.rand(hiding_bound)
        random_ints = convert_to_bigints(randomness.blind_poly)

        random_commitment:ProjectivePointG1 = MSM_new(powers_of_gamma_g[:],random_ints)
        random_commitment_affine = to_affine(random_commitment)
        commitment = add_assign_mixed(commitment,random_commitment_affine)
        commitment_affine = to_affine(commitment)
        return Commitment(value=commitment_affine),randomness
    
# On input a list of labeled polynomials and a query point, `open` outputs a proof of evaluation
# of the polynomials at the query point.
def open(
    ck: UniversalParams,
    labeled_polynomials: 'LabeledPoly',
    _commitments: 'LabeledCommitment',
    point,
    opening_challenge: field,
    rands,
    _rng=None
):
    
    combined_polynomial = torch.tensor([], dtype = fr.Fr.Dtype)
    combined_rand = Randomness.empty()

    opening_challenge_counter = 0

    curr_challenge = opening_challenges(opening_challenge, opening_challenge_counter)
    opening_challenge_counter += 1
   
    i=0
    for polynomial, rand in zip(labeled_polynomials, rands):
        combined_polynomial = poly_add_poly_mul_const(combined_polynomial, curr_challenge.to("cuda"), polynomial.poly)  #polynomial.poly is tensor
        combined_rand.add_assign(curr_challenge, rand)
        curr_challenge = opening_challenges(opening_challenge, opening_challenge_counter)
        opening_challenge_counter += 1
        i=i+1

    powers_of_g = torch.tensor(ck["powers_of_g"], dtype = fq.Fq.Dtype)
    powers_of_gamma_g = torch.tensor(ck["powers_of_gamma_g"], dtype = fq.Fq.Dtype)
    proof = open_proof(powers_of_g, powers_of_gamma_g, combined_polynomial.to("cuda"), point, combined_rand)
    return proof

dataclass
class LabeledCommitment:
    def __init__(self,label,commitment):
        self.label = label
        self.commitment =commitment

    @classmethod
    def new(cls,label,commitment):
        return cls(label = label,commitment = commitment)

class LabeledPoly:
    def __init__(self, label, hiding_bound, poly):
        self.label = label
        self.hiding_bound = hiding_bound
        self.poly = poly

    @classmethod
    def new(cls, label, hiding_bound, poly):
        return cls(label=label, hiding_bound=hiding_bound, poly=poly)


def commit_poly(ck:UniversalParams,polys,params):
    random.seed(42)
    randomness = []
    labeled_comm = []
    for labeled_poly in polys:
        polynomial = labeled_poly.poly
        hiding_bound = labeled_poly.hiding_bound
        label = labeled_poly.label

        powers = [ck.powers_of_g,ck.powers_of_gamma_g]

        comm,rand = Commitment.commit(powers,polynomial,hiding_bound,params) 
        labeled_comm.append(LabeledCommitment.new(label,comm))
        randomness.append(rand)
    return labeled_comm,randomness

def commit_poly_new(ck:UniversalParams, polys):
    random.seed(42)
    randomness = []
    labeled_comm = []
  
    for labeled_poly in polys:
        polynomial = labeled_poly.poly
        hiding_bound = labeled_poly.hiding_bound
        label = labeled_poly.label
        powers_of_g = torch.tensor(ck["powers_of_g"], dtype = fq.Fq.Dtype)
        powers_of_gamma_g = torch.tensor(ck["powers_of_gamma_g"], dtype = fq.Fq.Dtype)
        # for var in variables:
        #     if isinstance(var, torch.Tensor):
        #         print(var.device)
        #     else:
        #         print("Variable {} is not a PyTorch tensor.".format(var))
        comm,rand = Commitment.commit_new(powers_of_g,powers_of_gamma_g,polynomial,hiding_bound) 
        labeled_comm.append(LabeledCommitment.new(label,comm))
        randomness.append(rand)
        
    return labeled_comm,randomness




def opening_challenges(opening_challenge, exp):
    res = pow_single(opening_challenge, exp)
    return res

# Compute witness polynomial.
#
# The witness polynomial w(x) the quotient of the division (p(x) - p(z)) / (x - z)
# Observe that this quotient does not change with z because
# p(z) is the remainder term. We can therefore omit p(z) when computing the quotient.
import time
def compute_witness_polynomial(p: List[fr.Fr], point, randomness: Randomness):
    mod = fr.Fr.MODULUS.to("cuda")
    neg_p = F.sub_mod(mod, point)
    if p.size(0) != 0:
        witness_polynomial = F.poly_div_poly(p, neg_p)
    random_witness_polynomial = None
    if len(randomness.blind_poly) != 0:
        random_p = randomness.blind_poly
        random_witness_polynomial = F.poly_div_poly(random_p, neg_p)
    return witness_polynomial, random_witness_polynomial

def open_with_witness_polynomial(
    powers_of_g,
    powers_of_gamma_g, 
    point, 
    randomness: Randomness,
    witness_polynomial, 
    hiding_witness_polynomial):


    num_leading_zeros, witness_coeffs =skip_leading_zeros_and_convert_to_bigints(witness_polynomial)
    w = MSM_new(powers_of_g[num_leading_zeros:], witness_coeffs)
    random_v = None
    if hiding_witness_polynomial is not None:
        blinding_p = randomness.blind_poly
        blinding_evaluation = F.evaluate(blinding_p, point.to("cuda"))
        random_witness_coeffs = convert_to_bigints(hiding_witness_polynomial)
        random_commit = MSM_new(powers_of_gamma_g,random_witness_coeffs)
        w = add_assign(w,random_commit)
        random_v = blinding_evaluation
    
    return OpenProof(to_affine(w), random_v)

# On input a polynomial `p` and a point `point`, outputs a proof for the same.
def open_proof(powers_of_g,powers_of_gamma_g, p: List[field], point: field, rand: Randomness):
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
    a_comm: Commitment

    # Commitment to the witness polynomial for the right wires.
    b_comm: Commitment

    # Commitment to the witness polynomial for the output wires.
    c_comm: Commitment

    # Commitment to the witness polynomial for the fourth wires.
    d_comm: Commitment

    # Commitment to the permutation polynomial.
    z_comm: Commitment

    # Commitment to the lookup query polynomial.
    f_comm: Commitment

    # Commitment to first half of sorted polynomial
    h_1_comm: Commitment

    # Commitment to second half of sorted polynomial
    h_2_comm: Commitment

    # Commitment to the lookup permutation polynomial.
    z_2_comm: Commitment

    # Commitment to the quotient polynomial.
    t_1_comm: Commitment

    # Commitment to the quotient polynomial.
    t_2_comm: Commitment

    # Commitment to the quotient polynomial.
    t_3_comm: Commitment

    # Commitment to the quotient polynomial.
    t_4_comm: Commitment

    # Commitment to the quotient polynomial.
    t_5_comm: Commitment

    # Commitment to the quotient polynomial.
    t_6_comm: Commitment

    # Commitment to the quotient polynomial.
    t_7_comm: Commitment

    # Commitment to the quotient polynomial.
    t_8_comm: Commitment

    # Batch opening proof of the aggregated witnesses
    aw_opening: OpenProof

    # Batch opening proof of the shifted aggregated witnesses
    saw_opening: OpenProof

    # Subset of all of the evaluations added to the proof.
    evaluations: ProofEvaluations
