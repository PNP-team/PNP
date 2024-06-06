import torch
import numpy as np
from .structure import UniversalParams
from .bls12_381 import fr, fq
from .composer import StandardComposer


def to_fr_tensor(data):
    if data.size == 1:
        return torch.tensor([], dtype=fr.TYPE(), device="cuda")
    else:
        return torch.tensor(data, dtype=fr.TYPE(), device="cuda")


class LookupTable:
    def __init__(self, q_lookup, tables):
        self.q_lookup = to_fr_tensor(q_lookup)
        self.lookup_tables = [to_fr_tensor(table) for table in tables]


class Permutation:
    def __init__(self, left, right, out, fourth):
        self.left_sigma = to_fr_tensor(left)
        self.right_sigma = to_fr_tensor(right)
        self.out_sigma = to_fr_tensor(out)
        self.fourth_sigma = to_fr_tensor(fourth)


class Arithmetic:
    def __init__(self, q_arith, q_c, q_l, q_r, q_hl, q_hr, q_h4, q_o, q_4, q_m):
        self.q_arith = to_fr_tensor(q_arith)
        self.q_c = to_fr_tensor(q_c)
        self.q_l = to_fr_tensor(q_l)
        self.q_r = to_fr_tensor(q_r)
        self.q_hl = to_fr_tensor(q_hl)
        self.q_hr = to_fr_tensor(q_hr)
        self.q_h4 = to_fr_tensor(q_h4)
        self.q_o = to_fr_tensor(q_o)
        self.q_4 = to_fr_tensor(q_4)
        self.q_m = to_fr_tensor(q_m)


class Selector:
    def __init__(
        self,
        range_selector,
        logic_selector,
        fixed_group_add_selector,
        variable_group_add_selector,
    ):
        self.range = to_fr_tensor(range_selector)
        self.logic = to_fr_tensor(logic_selector)
        self.fixed_group_add = to_fr_tensor(fixed_group_add_selector)
        self.variable_group_add = to_fr_tensor(variable_group_add_selector)


class PublicKey:

    def __init__(
        self,
        lookups_coeffs,
        permutations_coeffs,
        arithmetics_coeffs,
        selectors_coeffs,
        lookups_evals,
        permutations_evals,
        arithmetics_evals,
        selectors_evals,
        linear_evaluations_evals,
        v_h_coset_8n_evals,
    ):
        self.lookups_coeffs = lookups_coeffs
        self.lookups_evals = lookups_evals
        self.permutations_coeffs = permutations_coeffs
        self.permutations_evals = permutations_evals
        self.arithmetics_coeffs = arithmetics_coeffs
        self.arithmetics_evals = arithmetics_evals
        self.selectors_coeffs = selectors_coeffs
        self.selectors_evals = selectors_evals
        self.linear_evaluations_evals = to_fr_tensor(linear_evaluations_evals)
        self.v_h_coset_8n_evals = to_fr_tensor(v_h_coset_8n_evals)


def parse_pp(pp_data, N):
    powers_of_g = pp_data["powers_of_g"][:N]
    powers_of_gamma_g = pp_data["powers_of_gamma_g"][:N]
    return UniversalParams(
        torch.tensor(powers_of_g, dtype=fq.TYPE(), device="cuda"),
        torch.tensor(powers_of_gamma_g, dtype=fq.TYPE(), device="cuda"),
    )


def parse_pk(pk_data):

    pk_lookup = pk_data["lookup"].tolist()
    lookups_coeffs = LookupTable(
        pk_lookup["q_lookup"]["coeffs"],
        [
            pk_lookup["table1"]["coeffs"],
            pk_lookup["table2"]["coeffs"],
            pk_lookup["table3"]["coeffs"],
            pk_lookup["table4"]["coeffs"],
        ],
    )

    lookups_evals = LookupTable(
        pk_lookup["q_lookup"]["evals"],
        [],
    )

    pk_permutation = pk_data["permutation"].tolist()
    permutations_coeffs = Permutation(
        pk_permutation["left_sigma"]["coeffs"],
        pk_permutation["right_sigma"]["coeffs"],
        pk_permutation["out_sigma"]["coeffs"],
        pk_permutation["fourth_sigma"]["coeffs"],
    )

    permutations_evals = Permutation(
        pk_permutation["left_sigma"]["evals"],
        pk_permutation["right_sigma"]["evals"],
        pk_permutation["out_sigma"]["evals"],
        pk_permutation["fourth_sigma"]["evals"],
    )

    pk_arithmetic = pk_data["arithmetic"].tolist()
    arithmetics_coeffs = Arithmetic(
        q_arith=pk_arithmetic["q_arith"]["coeffs"],
        q_c=pk_arithmetic["q_c"]["coeffs"],
        q_l=pk_arithmetic["q_l"]["coeffs"],
        q_r=pk_arithmetic["q_r"]["coeffs"],
        q_hl=pk_arithmetic["q_hl"]["coeffs"],
        q_hr=pk_arithmetic["q_hr"]["coeffs"],
        q_h4=pk_arithmetic["q_h4"]["coeffs"],
        q_o=pk_arithmetic["q_o"]["coeffs"],
        q_4=pk_arithmetic["q_4"]["coeffs"],
        q_m=pk_arithmetic["q_m"]["coeffs"],
    )

    arithmetics_evals = Arithmetic(
        q_arith=pk_arithmetic["q_arith"]["evals"],
        q_c=pk_arithmetic["q_c"]["evals"],
        q_l=pk_arithmetic["q_l"]["evals"],
        q_r=pk_arithmetic["q_r"]["evals"],
        q_hl=pk_arithmetic["q_hl"]["evals"],
        q_hr=pk_arithmetic["q_hr"]["evals"],
        q_h4=pk_arithmetic["q_h4"]["evals"],
        q_o=pk_arithmetic["q_o"]["evals"],
        q_4=pk_arithmetic["q_4"]["evals"],
        q_m=pk_arithmetic["q_m"]["evals"],
    )

    selectors_coeffs = Selector(
        pk_data["range_selector"].tolist()["coeffs"],
        pk_data["logic_selector"].tolist()["coeffs"],
        pk_data["fixed_group_add_selector"].tolist()["coeffs"],
        pk_data["variable_group_add_selector"].tolist()["coeffs"],
    )

    selectors_evals = Selector(
        pk_data["range_selector"].tolist()["evals"],
        pk_data["logic_selector"].tolist()["evals"],
        pk_data["fixed_group_add_selector"].tolist()["evals"],
        pk_data["variable_group_add_selector"].tolist()["evals"],
    )

    linear_evaluations_evals = pk_data["linear_evaluations"].tolist()["evals"]
    v_h_coset_8n_evals = pk_data["v_h_coset_8n"].tolist()["evals"]

    return PublicKey(
        lookups_coeffs,
        permutations_coeffs,
        arithmetics_coeffs,
        selectors_coeffs,
        lookups_evals,
        permutations_evals,
        arithmetics_evals,
        selectors_evals,
        linear_evaluations_evals,
        v_h_coset_8n_evals,
    )


def parse_cs(cs_data):
    cs = StandardComposer(
        n=cs_data["n"],
        public_inputs=to_fr_tensor(cs_data["public_inputs"]),
        q_lookup=to_fr_tensor(cs_data["q_lookup"]),
        intended_pi_pos=cs_data["intended_pi_pos"],
        lookup_table=cs_data["lookup_table"],
    )
    return cs


def load_meta(dir_name):
    cs_data = np.load(dir_name + "cs-9.npz", allow_pickle=True)
    pp_data = np.load(dir_name + "pp-9.npz", allow_pickle=True)
    pk_data = np.load(dir_name + "pk-9.npz", allow_pickle=True)

    cs = parse_cs(cs_data)

    num_coeffs = cs.circuit_bound()
    N = (
        num_coeffs
        if num_coeffs & (num_coeffs - 1) == 0
        else 2 ** num_coeffs.bit_length()
    )
    pp = parse_pp(pp_data, N)

    pk = parse_pk(pk_data)

    return pp, pk, cs
