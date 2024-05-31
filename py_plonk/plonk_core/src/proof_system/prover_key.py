from dataclasses import dataclass
# from ....field import field
from typing import List, Tuple
from ....plonk_core.src.proof_system.widget.arithmetic import Arith
from ....plonk_core.src.proof_system.widget.lookup import Lookup
from ....plonk_core.src.proof_system.permutation import Permutation
@dataclass 
class Prover_Key:

    def __init__(self, arithmetic, range_selector, logic_selector, lookup, Lookup, fixed_group_add_selector, variable_group_add_selector, permutation, Permutation, v_h_coset_8n):
        self.arithmetic = arithmetic
        self.range_selector = range_selector
        self.logic_selector = logic_selector
        self.lookup = lookup
        self.fixed_group_add_selector = fixed_group_add_selector
        self.variable_group_add_selector = variable_group_add_selector
        self.permutation = permutation
        self.v_h_coset_8n = v_h_coset_8n

    # arithmetic: Arith

    # range_selector: Tuple[List[field],List[field]]

    # logic_selector: Tuple[List[field],List[field]]

    # lookup: Lookup

    # fixed_group_add_selector: Tuple[List[field],List[field]]

    # variable_group_add_selector: Tuple[List[field],List[field]]

    # permutation: Permutation

    # v_h_coset_8n: List[field]