class StandardComposer:

    def __init__(self, n, q_lookup, intended_pi_pos, public_inputs, lookup_table):
        self.n = n
        self.q_lookup = q_lookup
        self.intended_pi_pos = intended_pi_pos
        self.public_inputs = public_inputs
        self.lookup_table = lookup_table
        

    def total_size(self):
        return max(self.n,len(self.lookup_table))
    
    def circuit_bound(self):
        def next_power_of_2(x):
            return 1<<(x-1).bit_length()
        return next_power_of_2(int(self.total_size()))
