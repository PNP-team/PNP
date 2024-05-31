from dataclasses import dataclass
from typing import List, Tuple
@dataclass
class CustomEvaluations:
    def __init__(self, vals):
        self.vals = vals
    # vals: List[Tuple[str, field]]
    # Get the evaluation of the specified label.
    # This funtions panics if the requested label is not found
    def get(self, label):
        for entry in self.vals:
            if entry[0] == label:
                return entry[1]
        raise ValueError(f"{label} label not found in evaluations set")