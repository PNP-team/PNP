from ...plonk_core.src import utils
from collections import defaultdict
from ...field import to_tensor_mont
from ...bls12_381 import fr
from ...arithmetic import tensor_to_int
from dataclasses import dataclass
import torch

@dataclass
class MultiSet:
    elements:torch.Tensor

    # def push(self, element):
    #     self.elements.append(element)

    # def pad(self, n):
    #     #judge whether n is power of 2
    #     assert n & (n - 1) == 0  
    #     #if self is empty, push one 0
    #     if not self.elements:
    #         self.push(0)  
    #     #if size < n, push 0
    #     while len(self.elements) < n:
    #         self.elements.append(self.elements[0])  # use the first element to pad
        
    def compress(self, alpha):
        compress_poly = utils.Multiset_lc(self, alpha)
        compress_poly = MultiSet(compress_poly)
        return compress_poly
    
    def combine_split(self, f_elements:'MultiSet'):
        # create buckets and init
        counters = defaultdict(int) #TODO: how to hash tensor
        for element in self.elements:
            integer = tensor_to_int(element)
            counters[integer] += 1

        # Insert the elements of f into the corresponding bucket and 
        # check whether there is a corresponding element in t
        for element in f_elements.elements:
            integer = tensor_to_int(element)
            if integer in counters and counters[integer] > 0:
                counters[integer] += 1
            else:
                raise ValueError("ElementNotIndexed")

        # Split s into two alternating halves evens and odd
        evens = torch.tensor([], dtype = fr.Fr.Dtype)
        odds = torch.tensor([], dtype = fr.Fr.Dtype)
        parity = 0
        for key, value in counters.items():
            item = to_tensor_mont(key, fr.Fr)
            half_count = value//2
            item_repeat = item.repeat(half_count, 1)
            evens = torch.cat((evens,item_repeat))
            odds = torch.cat((odds,item_repeat))
            if value % 2 ==1:
                if parity == 1:
                    odds = torch.cat((odds, item))
                    parity = 0
                else:
                    evens = torch.cat((evens, item))
                    parity = 1
        return evens, odds

    # def combine_split(self, f_elements:'MultiSet'):
    #     temp_s=from_tensor_list(self.elements)
    #     temp_f=from_tensor_list(f_elements.elements)
    #     from_list_gmpy(temp_s)
    #     from_list_gmpy(temp_f)
    #     # create buckets and init
    #     counters = defaultdict(gmpy2.mpz)
    #     for element in temp_s:
    #         counters[element.value] += 1

    #     # Insert the elements of f into the corresponding bucket and 
    #     # check whether there is a corresponding element in t
    #     for element in temp_f:
    #         if element.value in counters and counters[element.value] > 0:
    #             counters[element.value] += 1
    #         else:
    #             raise ValueError("ElementNotIndexed")

    #     # Split s into two alternating halves evens and odd

    #     evens = []
    #     odds = []
    #     parity =0
    #     for key, value in counters.items():
    #         key = fr.Fr(value=key)
    #         half_count = value//2
    #         evens.extend([key for _ in range(half_count)])
    #         # evens=torch.zeros(half_count,4,dtype=torch.BLS12_381_Fr_G1_Mont)
    #         # odds= torch.zeros(half_count,4,dtype=torch.BLS12_381_Fr_G1_Mont)
    #         odds.extend([key for _ in range(half_count)])
    #         if value % 2 ==1:
    #             if parity == 1:
    #                 odds.append(key)
    #                 parity = 0
    #             else:
    #                 evens.append(key)
    #                 parity = 1

    #     from_gmpy_list(evens)
    #     from_gmpy_list(odds)
    #     evens=from_list_tensor(evens)
    #     odds=from_list_tensor(odds)
    #     return evens, odds