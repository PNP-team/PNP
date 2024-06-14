import torch
import numpy as np
from .bls12_381 import fr, fq


def to_fr_tensor(data):
    if data.size == 1:
        return torch.tensor([], dtype=fr.TYPE(), device="cuda")
    else:
        return torch.tensor(data, dtype=fr.TYPE(), device="cuda")

class UniversalParams:
    def __init__(self, powers_of_g, powers_of_gamma_g):
        self.powers_of_g = powers_of_g
        self.powers_of_gamma_g = powers_of_gamma_g

class StandardComposer:

    def __init__(self, n, q_lookup, intended_pi_pos, public_inputs, lookup_table_size):
        self.n = n
        self.q_lookup = q_lookup
        self.intended_pi_pos = intended_pi_pos
        self.public_inputs = public_inputs
        self.lookup_table_size = lookup_table_size

        self.total_size = int(max(self.n, self.lookup_table_size))
        self.circuit_bound = (1 << (self.total_size - 1).bit_length())


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
        lookup_table_size=len(cs_data["lookup_table"]),
    )
    return cs


def save_meta(file_dir, pp, pk, cs):
    cs_dict = {
        "n": cs.n,
        "public_inputs": np.array(cs.public_inputs.tolist(), dtype=np.uint64),
        "q_lookup": np.array(cs.q_lookup.tolist(), dtype=np.uint64),
        "intended_pi_pos": cs.intended_pi_pos,
        "lookup_table_size": cs.lookup_table_size,
    }

    pp_dict = {
        "powers_of_g": np.array(pp.powers_of_g.tolist(), dtype=np.uint64),
        "powers_of_gamma_g": np.array(pp.powers_of_gamma_g.tolist(), dtype=np.uint64),
    }

    pk_dict = {
        "lookup_q_lookup_coeffs": np.array(
            pk.lookups_coeffs.q_lookup.tolist(), dtype=np.uint64
        ),
        "lookup_q_lookup_evals": np.array(
            pk.lookups_evals.q_lookup.tolist(), dtype=np.uint64
        ),
        "lookup_table1_coeffs": np.array(
            pk.lookups_coeffs.lookup_tables[0].tolist(), dtype=np.uint64
        ),
        "lookup_table2_coeffs": np.array(
            pk.lookups_coeffs.lookup_tables[1].tolist(), dtype=np.uint64
        ),
        "lookup_table3_coeffs": np.array(
            pk.lookups_coeffs.lookup_tables[2].tolist(), dtype=np.uint64
        ),
        "lookup_table4_coeffs": np.array(
            pk.lookups_coeffs.lookup_tables[3].tolist(), dtype=np.uint64
        ),
        "permutation_left_sigma_coeffs": np.array(
            pk.permutations_coeffs.left_sigma.tolist(), dtype=np.uint64
        ),
        "permutation_left_sigma_evals": np.array(
            pk.permutations_evals.left_sigma.tolist(), dtype=np.uint64
        ),
        "permutation_right_sigma_coeffs": np.array(
            pk.permutations_coeffs.right_sigma.tolist(), dtype=np.uint64
        ),
        "permutation_right_sigma_evals": np.array(
            pk.permutations_evals.right_sigma.tolist(), dtype=np.uint64
        ),
        "permutation_out_sigma_coeffs": np.array(
            pk.permutations_coeffs.out_sigma.tolist(), dtype=np.uint64
        ),
        "permutation_out_sigma_evals": np.array(
            pk.permutations_evals.out_sigma.tolist(), dtype=np.uint64
        ),
        "permutation_fourth_sigma_coeffs": np.array(
            pk.permutations_coeffs.fourth_sigma.tolist(), dtype=np.uint64
        ),
        "permutation_fourth_sigma_evals": np.array(
            pk.permutations_evals.fourth_sigma.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_arith_coeffs": np.array(
            pk.arithmetics_coeffs.q_arith.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_arith_evals": np.array(
            pk.arithmetics_evals.q_arith.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_c_coeffs": np.array(
            pk.arithmetics_coeffs.q_c.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_c_evals": np.array(
            pk.arithmetics_evals.q_c.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_l_coeffs": np.array(
            pk.arithmetics_coeffs.q_l.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_l_evals": np.array(
            pk.arithmetics_evals.q_l.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_r_coeffs": np.array(
            pk.arithmetics_coeffs.q_r.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_r_evals": np.array(
            pk.arithmetics_evals.q_r.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_hl_coeffs": np.array(
            pk.arithmetics_coeffs.q_hl.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_hl_evals": np.array(
            pk.arithmetics_evals.q_hl.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_hr_coeffs": np.array(
            pk.arithmetics_coeffs.q_hr.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_hr_evals": np.array(
            pk.arithmetics_evals.q_hr.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_h4_coeffs": np.array(
            pk.arithmetics_coeffs.q_h4.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_h4_evals": np.array(
            pk.arithmetics_evals.q_h4.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_o_coeffs": np.array(
            pk.arithmetics_coeffs.q_o.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_o_evals": np.array(
            pk.arithmetics_evals.q_o.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_4_coeffs": np.array(
            pk.arithmetics_coeffs.q_4.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_4_evals": np.array(
            pk.arithmetics_evals.q_4.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_m_coeffs": np.array(
            pk.arithmetics_coeffs.q_m.tolist(), dtype=np.uint64
        ),
        "arithmetic_q_m_evals": np.array(
            pk.arithmetics_evals.q_m.tolist(), dtype=np.uint64
        ),
        "selector_range_coeffs": np.array(
            pk.selectors_coeffs.range.tolist(), dtype=np.uint64
        ),
        "selector_range_evals": np.array(
            pk.selectors_evals.range.tolist(), dtype=np.uint64
        ),
        "selector_logic_coeffs": np.array(
            pk.selectors_coeffs.logic.tolist(), dtype=np.uint64
        ),
        "selector_logic_evals": np.array(
            pk.selectors_evals.logic.tolist(), dtype=np.uint64
        ),
        "selector_fixed_group_add_coeffs": np.array(
            pk.selectors_coeffs.fixed_group_add.tolist(), dtype=np.uint64
        ),
        "selector_fixed_group_add_evals": np.array(
            pk.selectors_evals.fixed_group_add.tolist(), dtype=np.uint64
        ),
        "selector_variable_group_add_coeffs": np.array(
            pk.selectors_coeffs.variable_group_add.tolist(), dtype=np.uint64
        ),
        "selector_variable_group_add_evals": np.array(
            pk.selectors_evals.variable_group_add.tolist(), dtype=np.uint64
        ),
        "linear_evaluations_evals": np.array(
            pk.linear_evaluations_evals.tolist(), dtype=np.uint64
        ),
        "v_h_coset_8n_evals": np.array(pk.v_h_coset_8n_evals.tolist(), dtype=np.uint64),
    }

    np.savez(file_dir+"/cs.npz", **cs_dict)
    np.savez(file_dir+"/pp.npz", **pp_dict)
    np.savez(file_dir+"/pk.npz", **pk_dict)


def load_from_npz(file_dir):
    cs_data = np.load(file_dir + "/cs.npz", allow_pickle=True)
    pp_data = np.load(file_dir + "/pp.npz", allow_pickle=True)
    pk_data = np.load(file_dir + "/pk.npz", allow_pickle=True)

    cs = StandardComposer(
        n=cs_data["n"],
        public_inputs=to_fr_tensor(cs_data["public_inputs"]),
        q_lookup=to_fr_tensor(cs_data["q_lookup"]),
        intended_pi_pos=cs_data["intended_pi_pos"],
        lookup_table_size=cs_data["lookup_table_size"],
    )

    pp = UniversalParams(
        torch.tensor(pp_data["powers_of_g"], dtype=fq.TYPE(), device="cuda"),
        torch.tensor(pp_data["powers_of_gamma_g"], dtype=fq.TYPE(), device="cuda"),
    )

    pk = PublicKey(
        LookupTable(
            pk_data["lookup_q_lookup_coeffs"],
            [
                pk_data["lookup_table1_coeffs"],
                pk_data["lookup_table2_coeffs"],
                pk_data["lookup_table3_coeffs"],
                pk_data["lookup_table4_coeffs"],
            ],
        ),
        Permutation(
            pk_data["permutation_left_sigma_coeffs"],
            pk_data["permutation_right_sigma_coeffs"],
            pk_data["permutation_out_sigma_coeffs"],
            pk_data["permutation_fourth_sigma_coeffs"],
        ),
        Arithmetic(
            q_arith=pk_data["arithmetic_q_arith_coeffs"],
            q_c=pk_data["arithmetic_q_c_coeffs"],
            q_l=pk_data["arithmetic_q_l_coeffs"],
            q_r=pk_data["arithmetic_q_r_coeffs"],
            q_hl=pk_data["arithmetic_q_hl_coeffs"],
            q_hr=pk_data["arithmetic_q_hr_coeffs"],
            q_h4=pk_data["arithmetic_q_h4_coeffs"],
            q_o=pk_data["arithmetic_q_o_coeffs"],
            q_4=pk_data["arithmetic_q_4_coeffs"],
            q_m=pk_data["arithmetic_q_m_coeffs"],
        ),
        Selector(
            pk_data["selector_range_coeffs"],
            pk_data["selector_logic_coeffs"],
            pk_data["selector_fixed_group_add_coeffs"],
            pk_data["selector_variable_group_add_coeffs"],
        ),
        LookupTable(
            pk_data["lookup_q_lookup_evals"],
            [],
        ),
        Permutation(
            pk_data["permutation_left_sigma_evals"],
            pk_data["permutation_right_sigma_evals"],
            pk_data["permutation_out_sigma_evals"],
            pk_data["permutation_fourth_sigma_evals"],
        ),
        Arithmetic(
            q_arith=pk_data["arithmetic_q_arith_evals"],
            q_c=pk_data["arithmetic_q_c_evals"],
            q_l=pk_data["arithmetic_q_l_evals"],
            q_r=pk_data["arithmetic_q_r_evals"],
            q_hl=pk_data["arithmetic_q_hl_evals"],
            q_hr=pk_data["arithmetic_q_hr_evals"],
            q_h4=pk_data["arithmetic_q_h4_evals"],
            q_o=pk_data["arithmetic_q_o_evals"],
            q_4=pk_data["arithmetic_q_4_evals"],
            q_m=pk_data["arithmetic_q_m_evals"],
        ),
        Selector(
            pk_data["selector_range_evals"],
            pk_data["selector_logic_evals"],
            pk_data["selector_fixed_group_add_evals"],
            pk_data["selector_variable_group_add_evals"],
        ),
        pk_data["linear_evaluations_evals"],
        pk_data["v_h_coset_8n_evals"],
    )

    return pp, pk, cs


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
