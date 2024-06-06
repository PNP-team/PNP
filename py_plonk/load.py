import torch
import numpy as np
from .structure import UniversalParams
from .bls12_381 import fr, fq
from .composer import StandardComposer

class PublicKey:
    def __init__(self, lookup_tables, permutations):
        self.lookup_tables = [torch.tensor(table, dtype=fr.TYPE(), device="cuda") for table in lookup_tables]
        self.permutation_left_sigma = torch.tensor(permutations[0], dtype=fr.TYPE(), device="cuda")
        self.permutation_right_sigma = torch.tensor(permutations[1], dtype=fr.TYPE(), device="cuda")
        self.permutation_out_sigma = torch.tensor(permutations[2], dtype=fr.TYPE(), device="cuda")
        self.permutation_fourth_sigma = torch.tensor(permutations[3], dtype=fr.TYPE(), device="cuda")



def parse_pp(pp_data, N):
    powers_of_g = pp_data["powers_of_g"][:N]
    powers_of_gamma_g = pp_data["powers_of_gamma_g"][:N]
    return UniversalParams(
        torch.tensor(powers_of_g, dtype=fq.TYPE(), device="cuda"),
        torch.tensor(powers_of_gamma_g, dtype=fq.TYPE(), device="cuda"),
    )

def parse_pk(pk_data):

    pk_lookup = pk_data["lookup"].tolist()
    lookup_tables = [pk_lookup["table1"]["coeffs"], pk_lookup["table2"]["coeffs"], pk_lookup["table3"]["coeffs"], pk_lookup["table4"]["coeffs"]]
    pk_permutation = pk_data["permutation"].tolist()
    permutations = [pk_permutation["left_sigma"]["coeffs"], pk_permutation["right_sigma"]["coeffs"], pk_permutation["out_sigma"]["coeffs"], pk_permutation["fourth_sigma"]["coeffs"]]

    return PublicKey(lookup_tables, permutations)


def parse_cs(cs_data):
    cs = StandardComposer(
        n=cs_data["n"],
        public_inputs=cs_data["public_inputs"],
        q_lookup=cs_data["q_lookup"],
        intended_pi_pos=cs_data["intended_pi_pos"],
        lookup_table=cs_data["lookup_table"],
    )
    return cs


def load(dir_name):
    cs_data = np.load(dir_name + "cs-9.npz", allow_pickle=True)
    pp_data = np.load(dir_name + "pp-9.npz", allow_pickle=True)
    pk_data = np.load(dir_name + "pk-9.npz", allow_pickle=True)

    cs = parse_cs(cs_data)

    num_coeffs = cs.circuit_bound()
    N = num_coeffs if num_coeffs & (num_coeffs - 1) == 0 else 2 ** num_coeffs.bit_length()
    pp = parse_pp(pp_data, N)

    pk = parse_pk(pk_data)

    return pp, pk, cs