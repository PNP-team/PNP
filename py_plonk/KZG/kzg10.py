import random
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from ..arithmetic import (
    MSM,
)
from ..bls12_381 import fq, fr
from ..jacobian import add_assign, to_affine
from ..structure import OpenProof


#########func for randomness#########
def empty_randomness():
    return torch.tensor([], dtype=fr.TYPE())


def calculate_hiding_polynomial_degree(hiding_bound):
    return hiding_bound + 1

def push(self, a):
    self.blind_poly.append(a)


def randomness_rand(hiding_bound):
    random.seed(42)
    hiding_poly_degree = calculate_hiding_polynomial_degree(hiding_bound)
    random_coeffs = [fr.make_tensor(random.random) for _ in range(hiding_poly_degree + 1)]
    return random_coeffs


def commit(powers_of_g, powers_of_gamma_g, polynomial, hiding_bound):
    plain_coeffs = F.to_base(polynomial)
    commitment = MSM(powers_of_g, plain_coeffs)
    randomness = empty_randomness()
    if hiding_bound:
        randomness = randomness_rand(hiding_bound)
    random_ints = F.to_base(randomness)

    random_commitment = MSM(powers_of_gamma_g, random_ints)
    commitment = add_assign(commitment, random_commitment)
    commitment_affine = to_affine(commitment)
    return commitment_affine, randomness


# On input a list of labeled polynomials and a query point, `open` outputs a proof of evaluation
# of the polynomials at the query point.
def open_proof(
    ck, labeled_polynomials, point, opening_challenge, rands, _rng=None
):

    combined_polynomial = torch.tensor([], dtype=fr.TYPE())

    for counter, ((polynomial, _), rand) in enumerate(zip(labeled_polynomials, rands)):
        # combined_polynomial += curr_challenge * polynomial
        curr_challenge = opening_challenges(opening_challenge, counter)
        if polynomial.size(0) != 0:
            temp = F.mul_mod_scalar(polynomial, curr_challenge)
            if combined_polynomial.size(0) != 0:
                combined_polynomial = F.resize_poly(combined_polynomial, polynomial.size(0))
                combined_polynomial = F.add_mod(combined_polynomial, temp)
            else:
                combined_polynomial = temp

    proof = open_proof_internal(
        ck.powers_of_g,
        ck.powers_of_gamma_g,
        combined_polynomial,
        point,
        empty_randomness(),
    )
    return proof

def commit_poly_new(ck, polys):
    random.seed(42)
    randomness = []
    labeled_comm = []

    for polynomial, hiding_bound in polys:
        comm, rand = commit(
            ck.powers_of_g, ck.powers_of_gamma_g, polynomial, hiding_bound
        )
        labeled_comm.append(comm)
        randomness.append(rand)

    return labeled_comm, randomness


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
    hiding_witness_polynomial,
):

    witness_coeffs = F.to_base(witness_polynomial)
    w = MSM(powers_of_g, witness_coeffs)
    random_v = None
    if hiding_witness_polynomial is not None:
        blinding_p = randomness
        blinding_evaluation = F.evaluate(blinding_p, point.to("cuda"))
        random_witness_coeffs = F.to_base(hiding_witness_polynomial)
        random_commit = MSM(powers_of_gamma_g, random_witness_coeffs)
        w = add_assign(w, random_commit)
        random_v = blinding_evaluation

    return OpenProof(to_affine(w), random_v)


# On input a polynomial `p` and a point `point`, outputs a proof for the same.
def open_proof_internal(powers_of_g, powers_of_gamma_g, p: list, point, rand):
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


class Proof:
    def __init__(
        self,
        a_comm,
        b_comm,
        c_comm,
        d_comm,
        z_comm,
        f_comm,
        h_1_comm,
        h_2_comm,
        z_2_comm,
        t_1_comm,
        t_2_comm,
        t_3_comm,
        t_4_comm,
        t_5_comm,
        t_6_comm,
        t_7_comm,
        t_8_comm,
        aw_opening,
        saw_opening,
        evaluations,
    ):
        self.a_comm = a_comm  # Commitment to the witness polynomial for the left wires.
        self.b_comm = (
            b_comm  # Commitment to the witness polynomial for the right wires.
        )
        self.c_comm = (
            c_comm  # Commitment to the witness polynomial for the output wires.
        )
        self.d_comm = (
            d_comm  # Commitment to the witness polynomial for the fourth wires.
        )
        self.z_comm = z_comm  # Commitment to the permutation polynomial.
        self.f_comm = f_comm  # Commitment to the lookup query polynomial.
        self.h_1_comm = h_1_comm  # Commitment to first half of sorted polynomial
        self.h_2_comm = h_2_comm  # Commitment to second half of sorted polynomial
        self.z_2_comm = z_2_comm  # Commitment to the lookup permutation polynomial.
        self.t_1_comm = t_1_comm  # Commitment to the quotient polynomial.
        self.t_2_comm = t_2_comm  # Commitment to the quotient polynomial.
        self.t_3_comm = t_3_comm  # Commitment to the quotient polynomial.
        self.t_4_comm = t_4_comm  # Commitment to the quotient polynomial.
        self.t_5_comm = t_5_comm  # Commitment to the quotient polynomial.
        self.t_6_comm = t_6_comm  # Commitment to the quotient polynomial.
        self.t_7_comm = t_7_comm  # Commitment to the quotient polynomial.
        self.t_8_comm = t_8_comm  # Commitment to the quotient polynomial.
        self.aw_opening = aw_opening  # Batch opening proof of the aggregated witnesses
        self.saw_opening = (
            saw_opening  # Batch opening proof of the shifted aggregated witnesses
        )
        self.evaluations = (
            evaluations  # Subset of all of the evaluations added to the proof.
        )

    def write_proof(self, file_name):
        with open(file_name, "w") as f:
            for attribute, value in self.__dict__.items():
                if attribute == "evaluations":
                    pass
                elif attribute == "aw_opening" or attribute == "saw_opening":
                    print(
                        "{}: ({},{})".format(
                            attribute, value.w.x.tolist(), value.w.y.tolist()
                        ),
                        file=f,
                    )
                else:
                    print(
                        "{}: ({},{})".format(
                            attribute, value.x.tolist(), value.y.tolist()
                        ),
                        file=f,
                    )
