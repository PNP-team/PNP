from ...plonk_core.src import utils
from collections import defaultdict
from ...bls12_381 import fr
from ...arithmetic import tensor_to_int
import torch


def combine_split(t_elements: torch.Tensor, f_elements: torch.Tensor):
    #TODO this function has not been implemented 
    return t_elements.clone(), f_elements.clone()
    
    t_elements_tensor = t_elements.to('cpu')
    f_elements_tensor = f_elements.to('cpu')

    # create buckets and init
    counters = defaultdict(int) #TODO: how to hash tensor
    for element in t_elements_tensor:
        integer = tensor_to_int(element)
        counters[integer] += 1

    # Insert the elements of f into the corresponding bucket and 
    # check whether there is a corresponding element in t
    for element in f_elements_tensor:
        integer = tensor_to_int(element)
        if integer in counters and counters[integer] > 0:
            counters[integer] += 1
        else:
            raise ValueError("ElementNotIndexed")

    # Split s into two alternating halves evens and odd
    evens = torch.tensor([], dtype = fr.TYPE())
    odds = torch.tensor([], dtype = fr.TYPE())
    parity = 0
    for key, value in counters.items():
        item = fr.make_tensor(key, to_mont=False)
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