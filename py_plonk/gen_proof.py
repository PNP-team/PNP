import itertools
from .domain import Radix2EvaluationDomain
from .transcript import transcript
from .composer import StandardComposer
from .bls12_381 import fr,fq
from .plonk_core.lookup.multiset import combine_split
from .plonk_core.src.permutation import mod
from .plonk_core.src.proof_system.pi import into_dense_poly
from .plonk_core.src.proof_system import quotient_poly
from .plonk_core.src.proof_system import linearisation_poly
import numpy as np
from .KZG import kzg10
import torch
import torch.nn.functional as F
from torch.pnp import zkp
import time
import torch.nn as nn


# data_set2 = [
#     "../../data/MERKLE-HEIGHT-3/pp-3.npz",
#     "../../data/MERKLE-HEIGHT-3/pk-3.npz",
#     "../../data/MERKLE-HEIGHT-3/cs-3.npz",
#     "../../data/MERKLE-HEIGHT-3/w_l_scalar-3.npy",
#     "../../data/MERKLE-HEIGHT-3/w_r_scalar-3.npy",
#     "../../data/MERKLE-HEIGHT-3/w_o_scalar-3.npy",
#     "../../data/MERKLE-HEIGHT-3/w_4_scalar-3.npy",
# ]
data_set2=["../../data/MERKLE-HEIGHT-9/pp-9.npz","../../data/MERKLE-HEIGHT-9/pk-9.npz","../../data/MERKLE-HEIGHT-9/cs-9.npz","../../data/MERKLE-HEIGHT-9/w_l_scalar-9.npy","../../data/MERKLE-HEIGHT-9/w_r_scalar-9.npy","../../data/MERKLE-HEIGHT-9/w_o_scalar-9.npy","../../data/MERKLE-HEIGHT-9/w_4_scalar-9.npy"]


def split_tx_poly(n, t_x):
    # t_x = F.pad_poly(t_x, n << 3)
    return [
        t_x[0:n],
        t_x[n : 2 * n],
        t_x[2 * n : 3 * n],
        t_x[3 * n : 4 * n],
        t_x[4 * n : 5 * n],
        t_x[5 * n : 6 * n],
        t_x[6 * n : 7 * n],
        t_x[7 * n :],
    ]


class gen_proof(torch.nn.Module):
    def __init__(self):
        super(gen_proof, self).__init__()

        self.INTT = nn.Intt(fr.TWO_ADICITY(), fr.TYPE())
        # self.NTT = nn.Ntt(fr.TWO_ADICITY(), fr.TYPE())
        # self.NTT_COSET = nn.Ntt_coset(fr.TWO_ADICITY(), fr.TYPE())
        # self.INTT_COSET = nn.Intt_coset(fr.TWO_ADICITY(), fr.TYPE())

    def __call__(
        self,
        pp,
        pk,
        cs: StandardComposer,
        transcript: transcript.Transcript,
    ):

        domain = Radix2EvaluationDomain.new(cs.circuit_bound())
        n = domain.size
        transcript.append_pi(
            b"pi",
            torch.tensor(cs.public_inputs, dtype=torch.BLS12_381_Fr_G1_Mont),
            int(cs.intended_pi_pos),
        )

        # 1. Compute witness Polynomials
        w_l_scalar = torch.tensor(
            np.load(data_set2[3], allow_pickle=True), dtype=torch.BLS12_381_Fr_G1_Mont
        )
        w_r_scalar = torch.tensor(
            np.load(data_set2[4], allow_pickle=True), dtype=torch.BLS12_381_Fr_G1_Mont
        )
        w_o_scalar = torch.tensor(
            np.load(data_set2[5], allow_pickle=True), dtype=torch.BLS12_381_Fr_G1_Mont
        )
        w_4_scalar = torch.tensor(
            np.load(data_set2[6], allow_pickle=True), dtype=torch.BLS12_381_Fr_G1_Mont
        )
        w_l_scalar = w_l_scalar.to("cuda")
        w_r_scalar = w_r_scalar.to("cuda")
        w_o_scalar = w_o_scalar.to("cuda")
        w_4_scalar = w_4_scalar.to("cuda")

        w_l_poly = self.INTT(w_l_scalar)
        w_r_poly = self.INTT(w_r_scalar)
        w_o_poly = self.INTT(w_o_scalar)
        w_4_poly = self.INTT(w_4_scalar)

        w_polys = [
            kzg10.LabeledPoly.new(label="w_l_poly", hiding_bound=None, poly=w_l_poly),
            kzg10.LabeledPoly.new(label="w_r_poly", hiding_bound=None, poly=w_r_poly),
            kzg10.LabeledPoly.new(label="w_o_poly", hiding_bound=None, poly=w_o_poly),
            kzg10.LabeledPoly.new(label="w_4_poly", hiding_bound=None, poly=w_4_poly),
        ]
        #####pre-load commit points#####
        powers_of_g = torch.tensor(pp["powers_of_g"], dtype = fq.TYPE())[:n]
        powers_of_gamma_g = torch.tensor(pp["powers_of_gamma_g"], dtype = fq.TYPE())[:n]
        ck = [powers_of_g.to("cuda"), powers_of_gamma_g.to("cuda")]

        w_commits, w_rands = kzg10.commit_poly_new(ck, w_polys)

        transcript.append(b"w_l", w_commits[0].commitment)
        transcript.append(b"w_r", w_commits[1].commitment)
        transcript.append(b"w_o", w_commits[2].commitment)
        transcript.append(b"w_4", w_commits[3].commitment)

        # 2. Derive lookup polynomials

        # Generate table compression factor
        zeta = transcript.challenge_scalar(b"zeta")
        transcript.append(b"zeta", zeta)

        pk_lookup = pk["lookup"].tolist()
        pk_lookup_table1 = torch.tensor(
            pk_lookup["table1"]["coeffs"], dtype=torch.BLS12_381_Fr_G1_Mont
        ).to("cuda")
        pk_lookup_table2 = torch.tensor(
            pk_lookup["table2"]["coeffs"], dtype=torch.BLS12_381_Fr_G1_Mont
        ).to("cuda")
        pk_lookup_table3 = torch.tensor(
            pk_lookup["table3"]["coeffs"], dtype=torch.BLS12_381_Fr_G1_Mont
        ).to("cuda")
        pk_lookup_table4 = torch.tensor(
            pk_lookup["table4"]["coeffs"], dtype=torch.BLS12_381_Fr_G1_Mont
        ).to("cuda")

        # Compress lookup table into vector of single elements
        compressed_t_multiset = zkp.compress(
            [pk_lookup_table1, pk_lookup_table2, pk_lookup_table3, pk_lookup_table4],
            zeta.to("cuda"),
        )
        # Compute table poly

        table_poly = self.INTT(compressed_t_multiset)

        # Compute query table f
        # When q_lookup[i] is zero the wire value is replaced with a dummy
        # value currently set as the first row of the public table
        # If q_lookup[i] is one the wire values are preserved
        # This ensures the ith element of the compressed query table
        # is an element of the compressed lookup table even when
        # q_lookup[i] is 0 so the lookup check will pass
        # q_lookup_pad = [0 for _ in range(n-len(cs.q_lookup))]
        # padded_q_lookup = list(cs.q_lookup) + q_lookup_pad

        compressed_f_multiset = zkp.compute_query_table(
            torch.tensor(cs.q_lookup, dtype=fr.TYPE()).to("cuda"),
            w_l_scalar,
            w_r_scalar,
            w_o_scalar,
            w_4_scalar,
            compressed_t_multiset,
            zeta.to("cuda"),
        )

        # Compute query poly
        f_poly = self.INTT(compressed_f_multiset)
        f_polys = [
            kzg10.LabeledPoly.new(label="f_poly", hiding_bound=None, poly=f_poly)
        ]

        # Commit to query polynomial
        f_poly_commit, _ = kzg10.commit_poly_new(ck, f_polys)
        transcript.append(b"f", f_poly_commit[0].commitment)

        # Compute s, as the sorted and concatenated version of f and t
        # work on cpu
        h_1, h_2 = combine_split(compressed_t_multiset, compressed_f_multiset)

        # Compute h polys
        h_1 = h_1.to("cuda")
        h_2 = h_2.to("cuda")
        h_1_poly = self.INTT(h_1)
        h_2_poly = self.INTT(h_2)

        # Commit to h polys
        h_1_polys = [
            kzg10.LabeledPoly.new(label="h_1_poly", hiding_bound=None, poly=h_1_poly)
        ]
        h_2_polys = [
            kzg10.LabeledPoly.new(label="h_2_poly", hiding_bound=None, poly=h_2_poly)
        ]
        h_1_poly_commit, _ = kzg10.commit_poly_new(ck, h_1_polys)
        h_2_poly_commit, _ = kzg10.commit_poly_new(ck, h_2_polys)

        # Add h polynomials to transcript
        transcript.append(b"h1", h_1_poly_commit[0].commitment)
        transcript.append(b"h2", h_2_poly_commit[0].commitment)

        # 3. Compute permutation polynomial

        # Compute permutation challenge `beta`.
        beta = transcript.challenge_scalar(b"beta")
        transcript.append(b"beta", beta)

        # Compute permutation challenge `gamma`.
        gamma = transcript.challenge_scalar(b"gamma")
        transcript.append(b"gamma", gamma)

        # Compute permutation challenge `delta`.
        delta = transcript.challenge_scalar(b"delta")
        transcript.append(b"delta", delta)

        # Compute permutation challenge `epsilon`.
        epsilon = transcript.challenge_scalar(b"epsilon")
        transcript.append(b"epsilon", epsilon)

        # Challenges must be different
        assert F.trace_equal(beta, gamma) == False, "challenges must be different"
        assert F.trace_equal(beta, delta) == False, "challenges must be different"
        assert F.trace_equal(beta, epsilon) == False, "challenges must be different"
        assert F.trace_equal(gamma, delta) == False, "challenges must be different"
        assert F.trace_equal(gamma, epsilon) == False, "challenges must be different"
        assert F.trace_equal(delta, epsilon) == False, "challenges must be different"

        pk_permutation = pk["permutation"].tolist()
        pk_permutation["left_sigma"]["coeffs"] = torch.tensor(
            pk_permutation["left_sigma"]["coeffs"], dtype=fr.TYPE()
        )
        pk_permutation["right_sigma"]["coeffs"] = torch.tensor(
            pk_permutation["right_sigma"]["coeffs"], dtype=fr.TYPE()
        )
        pk_permutation["out_sigma"]["coeffs"] = torch.tensor(
            pk_permutation["out_sigma"]["coeffs"], dtype=fr.TYPE()
        )
        pk_permutation["fourth_sigma"]["coeffs"] = torch.tensor(
            pk_permutation["fourth_sigma"]["coeffs"], dtype=fr.TYPE()
        )
        z_poly = mod.compute_permutation_poly(
            domain,
            (w_l_scalar, w_r_scalar, w_o_scalar, w_4_scalar),
            beta,
            gamma,
            [
                pk_permutation["left_sigma"]["coeffs"],
                pk_permutation["right_sigma"]["coeffs"],
                pk_permutation["out_sigma"]["coeffs"],
                pk_permutation["fourth_sigma"]["coeffs"],
            ],
        )
        # Commit to permutation polynomial.

        z_polys = [
            kzg10.LabeledPoly.new(label="z_poly", hiding_bound=None, poly=z_poly)
        ]
        z_poly_commit, _ = kzg10.commit_poly_new(ck, z_polys)

        # Add permutation polynomial commitment to transcript.
        transcript.append(b"z", z_poly_commit[0].commitment)

        # return 438.730

        # Compute mega permutation polynomial.
        # Compute lookup permutation poly
        z_2_poly = mod.compute_lookup_permutation_poly(
            n, compressed_f_multiset, compressed_t_multiset, h_1, h_2, delta.to("cuda"), epsilon.to("cuda")
        )
        # Commit to lookup permutation polynomial.
        z_2_polys = [
            kzg10.LabeledPoly.new(label="z_2_poly", hiding_bound=None, poly=z_2_poly)
        ]
        z_2_poly_commit, _ = kzg10.commit_poly_new(ck, z_2_polys)

        # return 477.107

        # 3. Compute public inputs polynomial
        cs_public_inputs = torch.tensor(cs.public_inputs, dtype=fr.TYPE())
        pi_poly = into_dense_poly(cs_public_inputs, int(cs.intended_pi_pos), n, self.INTT)

        # 4. Compute quotient polynomial
        # Compute quotient challenge `alpha`, and gate-specific separation challenges.
        alpha = transcript.challenge_scalar(b"alpha")
        transcript.append(b"alpha", alpha)

        range_sep_challenge = transcript.challenge_scalar(b"range separation challenge")
        transcript.append(b"range seperation challenge", range_sep_challenge)

        logic_sep_challenge = transcript.challenge_scalar(b"logic separation challenge")
        transcript.append(b"logic seperation challenge", logic_sep_challenge)

        fixed_base_sep_challenge = transcript.challenge_scalar(
            b"fixed base separation challenge"
        )
        transcript.append(b"fixed base separation challenge", fixed_base_sep_challenge)

        var_base_sep_challenge = transcript.challenge_scalar(
            b"variable base separation challenge"
        )
        transcript.append(b"variable base separation challenge", var_base_sep_challenge)

        lookup_sep_challenge = transcript.challenge_scalar(
            b"lookup separation challenge"
        )
        transcript.append(b"lookup separation challenge", lookup_sep_challenge)

        t_poly = quotient_poly.compute_quotient_poly(
            n,
            pk,
            z_poly,
            z_2_poly,
            w_l_poly,
            w_r_poly,
            w_o_poly,
            w_4_poly,
            pi_poly,
            f_poly,
            table_poly,
            h_1_poly,
            h_2_poly,
            alpha,
            beta,
            gamma,
            delta,
            epsilon,
            zeta,
            range_sep_challenge,
            logic_sep_challenge,
            fixed_base_sep_challenge,
            var_base_sep_challenge,
            lookup_sep_challenge,
        )

        # return 778.620

        start = time.time()
        t_i_poly = split_tx_poly(n, t_poly)
        elapse = time.time() - start
        print("split time:", elapse)

        t_i_polys = [
            kzg10.LabeledPoly.new(
                label="t_i_polys[0]", hiding_bound=None, poly=t_i_poly[0]
            ),
            kzg10.LabeledPoly.new(
                label="t_i_polys[1]", hiding_bound=None, poly=t_i_poly[1]
            ),
            kzg10.LabeledPoly.new(
                label="t_i_polys[2]", hiding_bound=None, poly=t_i_poly[2]
            ),
            kzg10.LabeledPoly.new(
                label="t_i_polys[3]", hiding_bound=None, poly=t_i_poly[3]
            ),
            kzg10.LabeledPoly.new(
                label="t_i_polys[4]", hiding_bound=None, poly=t_i_poly[4]
            ),
            kzg10.LabeledPoly.new(
                label="t_i_polys[5]", hiding_bound=None, poly=t_i_poly[5]
            ),
            kzg10.LabeledPoly.new(
                label="t_i_polys[6]", hiding_bound=None, poly=t_i_poly[6]
            ),
            kzg10.LabeledPoly.new(
                label="t_i_polys[7]", hiding_bound=None, poly=t_i_poly[7]
            ),
        ]

        t_commits, _ = kzg10.commit_poly_new(ck, t_i_polys)

        # Add quotient polynomial commitments to transcript
        for i in range(0, 8):
            transcript.append(f"t_{i+1}".encode(), t_commits[i].commitment)

        # 4. Compute linearisation polynomial
        # Compute evaluation challenge `z`.
        z_challenge = transcript.challenge_scalar(b"z")
        transcript.append(b"z", z_challenge)
        lin_poly, evaluations = linearisation_poly.compute_linearisation_poly(
            domain,
            pk,
            alpha,
            beta,
            gamma,
            delta,
            epsilon,
            zeta,
            range_sep_challenge,
            logic_sep_challenge,
            fixed_base_sep_challenge,
            var_base_sep_challenge,
            lookup_sep_challenge,
            z_challenge,
            w_l_poly,
            w_r_poly,
            w_o_poly,
            w_4_poly,
            t_i_poly[0],
            t_i_poly[1],
            t_i_poly[2],
            t_i_poly[3],
            t_i_poly[4],
            t_i_poly[5],
            t_i_poly[6],
            t_i_poly[7],
            z_poly,
            z_2_poly,
            f_poly,
            h_1_poly,
            h_2_poly,
            table_poly,
        )

        evaluations.wire_evals.a_eval = evaluations.wire_evals.a_eval.to("cpu")
        evaluations.wire_evals.b_eval = evaluations.wire_evals.b_eval.to("cpu")
        evaluations.wire_evals.c_eval = evaluations.wire_evals.c_eval.to("cpu")
        evaluations.wire_evals.d_eval = evaluations.wire_evals.d_eval.to("cpu")
        evaluations.perm_evals.left_sigma_eval = (
            evaluations.perm_evals.left_sigma_eval.to("cpu")
        )
        evaluations.perm_evals.right_sigma_eval = (
            evaluations.perm_evals.right_sigma_eval.to("cpu")
        )
        evaluations.perm_evals.out_sigma_eval = (
            evaluations.perm_evals.out_sigma_eval.to("cpu")
        )
        evaluations.perm_evals.permutation_eval = (
            evaluations.perm_evals.permutation_eval.to("cpu")
        )
        evaluations.lookup_evals.f_eval = evaluations.lookup_evals.f_eval.to("cpu")
        evaluations.lookup_evals.q_lookup_eval = (
            evaluations.lookup_evals.q_lookup_eval.to("cpu")
        )
        evaluations.lookup_evals.z2_next_eval = (
            evaluations.lookup_evals.z2_next_eval.to("cpu")
        )
        evaluations.lookup_evals.h1_eval = evaluations.lookup_evals.h1_eval.to("cpu")
        evaluations.lookup_evals.h1_next_eval = (
            evaluations.lookup_evals.h1_next_eval.to("cpu")
        )
        evaluations.lookup_evals.h2_eval = evaluations.lookup_evals.h2_eval.to("cpu")

        # Add evaluations to transcript.
        # First wire evals
        transcript.append(b"a_eval", evaluations.wire_evals.a_eval)
        transcript.append(b"b_eval", evaluations.wire_evals.b_eval)
        transcript.append(b"c_eval", evaluations.wire_evals.c_eval)
        transcript.append(b"d_eval", evaluations.wire_evals.d_eval)

        # Second permutation evals
        transcript.append(b"left_sig_eval", evaluations.perm_evals.left_sigma_eval)
        transcript.append(b"right_sig_eval", evaluations.perm_evals.right_sigma_eval)
        transcript.append(b"out_sig_eval", evaluations.perm_evals.out_sigma_eval)
        transcript.append(b"perm_eval", evaluations.perm_evals.permutation_eval)

        # Third lookup evals
        transcript.append(b"f_eval", evaluations.lookup_evals.f_eval)
        transcript.append(b"q_lookup_eval", evaluations.lookup_evals.q_lookup_eval)
        transcript.append(b"lookup_perm_eval", evaluations.lookup_evals.z2_next_eval)
        transcript.append(b"h_1_eval", evaluations.lookup_evals.h1_eval)
        transcript.append(b"h_1_next_eval", evaluations.lookup_evals.h1_next_eval)
        transcript.append(b"h_2_eval", evaluations.lookup_evals.h2_eval)

        # Fourth, all evals needed for custom gates
        for label, eval in evaluations.custom_evals.items():
            static_label = label.encode("utf-8")
            transcript.append(static_label, eval)

        # 5. Compute Openings using KZG10
        #
        # We merge the quotient polynomial using the `z_challenge` so the SRS
        # is linear in the circuit size `n`

        # Compute aggregate witness to polynomials evaluated at the evaluation
        # challenge `z`
        aw_challenge = transcript.challenge_scalar(b"aggregate_witness")

        # XXX: The quotient polynomials is used here and then in the
        # opening poly. It is being left in for now but it may not
        # be necessary. Warrants further investigation.
        # Ditto with the out_sigma poly.

        aw_polys = [
            kzg10.LabeledPoly.new(label="lin_poly", hiding_bound=None, poly=lin_poly),
            kzg10.LabeledPoly.new(
                label="prover_key.permutation.left_sigma.0.clone()",
                hiding_bound=None,
                poly=pk_permutation["left_sigma"]["coeffs"],
            ),
            kzg10.LabeledPoly.new(
                label="prover_key.permutation.right_sigma.0.clone()",
                hiding_bound=None,
                poly=pk_permutation["right_sigma"]["coeffs"],
            ),
            kzg10.LabeledPoly.new(
                label="prover_key.permutation.out_sigma.0.clone()",
                hiding_bound=None,
                poly=pk_permutation["out_sigma"]["coeffs"],
            ),
            kzg10.LabeledPoly.new(label="f_poly", hiding_bound=None, poly=f_poly),
            kzg10.LabeledPoly.new(label="h_2_poly", hiding_bound=None, poly=h_2_poly),
            kzg10.LabeledPoly.new(
                label="table_poly", hiding_bound=None, poly=table_poly
            ),
        ]

        aw_commits, aw_rands = kzg10.commit_poly_new(ck, aw_polys)
        aw_opening = kzg10.open(
            ck,
            itertools.chain(aw_polys, w_polys),
            itertools.chain(aw_commits, w_commits),
            z_challenge.to("cuda"),
            aw_challenge.to("cuda"),
            itertools.chain(aw_rands, w_rands),
            None,
        )

        saw_challenge = transcript.challenge_scalar(b"aggregate_witness")
        saw_polys = [
            kzg10.LabeledPoly.new(label="z_poly", hiding_bound=None, poly=z_poly),
            kzg10.LabeledPoly.new(label="w_l_poly", hiding_bound=None, poly=w_l_poly),
            kzg10.LabeledPoly.new(label="w_r_poly", hiding_bound=None, poly=w_r_poly),
            kzg10.LabeledPoly.new(label="w_4_poly", hiding_bound=None, poly=w_4_poly),
            kzg10.LabeledPoly.new(label="h_1_poly", hiding_bound=None, poly=h_1_poly),
            kzg10.LabeledPoly.new(label="z_2_poly", hiding_bound=None, poly=z_2_poly),
            kzg10.LabeledPoly.new(
                label="table_poly", hiding_bound=None, poly=table_poly
            ),
        ]

        saw_commits, saw_rands = kzg10.commit_poly_new(ck, saw_polys)
        element = domain.element(1)
        open_point = F.mul_mod(z_challenge.to("cuda"), element)
        saw_opening = kzg10.open(
            ck,
            saw_polys,
            saw_commits,
            open_point.to("cuda"),
            saw_challenge.to("cuda"),
            saw_rands,
            None,
        )

        Proof = kzg10.Proof(
            a_comm=w_commits[0].commitment,
            b_comm=w_commits[1].commitment,
            c_comm=w_commits[2].commitment,
            d_comm=w_commits[3].commitment,
            z_comm=saw_commits[0].commitment,
            f_comm=f_poly_commit[0].commitment,
            h_1_comm=h_1_poly_commit[0].commitment,
            h_2_comm=h_2_poly_commit[0].commitment,
            z_2_comm=z_2_poly_commit[0].commitment,
            t_1_comm=t_commits[0].commitment,
            t_2_comm=t_commits[1].commitment,
            t_3_comm=t_commits[2].commitment,
            t_4_comm=t_commits[3].commitment,
            t_5_comm=t_commits[4].commitment,
            t_6_comm=t_commits[5].commitment,
            t_7_comm=t_commits[6].commitment,
            t_8_comm=t_commits[7].commitment,
            aw_opening=aw_opening,
            saw_opening=saw_opening,
            evaluations=evaluations,
        )

        def write_to_file(data, filename):
            try:
                with open(filename, "a") as file:
                    file.write(str(data) + "\n")
                print(f"Data is successfully written to the file {filename}")
            except Exception as e:
                print(f"An error occurred while writing to the file: {e}")

        with open("proof_data_new.txt", "w") as f:
            pass

        attributes = vars(Proof)
        for attribute, value in attributes.items():
            # print("====", value)
            try:
                my_data = {f"{attribute}: {value.x},{value.y}"}
                print(f"{attribute}: {value.x},{value.y}")
                write_to_file(my_data, "proof_data_new.txt")
            except:
                try:
                    my_data = {f"{attribute}: {value.w.x},{value.w.y}"}
                    print(f"{attribute}: {value.w.x},{value.w.y}")
                    write_to_file(my_data, "proof_data_new.txt")
                except:
                    try:
                        my_data = {f"{attribute}: {value}"}
                        print(f"{attribute}: {value.w.x.value},{value.w.y.value}")
                        write_to_file(my_data, "proof_data_new.txt")
                    except:
                        pass

        return w_commits[0].commitment

        return Proof
