from dataclasses import dataclass
from .bls12_381 import fr
import torch
import torch.nn.functional as F


def neg(self):
    a=torch.tensor([18446744069414584321, 6034159408538082302, 3691218898639771653, 8353516859464449352],dtype=torch.BLS12_381_Fr_G1_Mont)
    if torch.equal(self,torch.tensor([0,0,0,0],dtype=torch.BLS12_381_Fr_G1_Mont)):
        return self
    else:
        res= F.sub_mod(a,self)
        return res

@dataclass
class Radix2EvaluationDomain:
    size: int
    log_size_of_group: int
    size_as_field_element: torch.Tensor
    size_inv: torch.Tensor
    group_gen: torch.Tensor
    group_gen_inv: torch.Tensor
    generator_inv: torch.Tensor

    @classmethod
    def new(cls, num_coeffs: int):
        # Compute the size of our evaluation domain
        size = num_coeffs if num_coeffs & (num_coeffs - 1) == 0 else 2 ** num_coeffs.bit_length()
        log_size_of_group = size.bit_length()-1
        
        # Check if log_size_of_group exceeds TWO_ADICITY
        if log_size_of_group > fr.Fr.TWO_ADICITY:
            return None

        # Compute the generator for the multiplicative subgroup.
        # It should be the 2^(log_size_of_group) root of unity.
        group_gen = fr.Fr.get_root_of_unity(size)
        
        # Check that it is indeed the 2^(log_size_of_group) root of unity.
        group_gen_pow = F.exp_mod(group_gen,size)
        assert torch.equal(group_gen_pow, fr.Fr.one())

        size_as_field_element=fr.Fr.from_repr(size)
        size_inv = fr.Fr.inverse(size_as_field_element)
        group_gen_inv = fr.Fr.inverse(group_gen)
        generator_inv = fr.Fr.inverse(fr.Fr.multiplicative_generator())

        return cls(size, log_size_of_group, size_as_field_element, size_inv, group_gen, group_gen_inv, generator_inv)
    
    # @classmethod
    # def new(cls,num_coeffs:int):
    #     # Compute the size of our evaluation domain
    #     size = num_coeffs if num_coeffs & (num_coeffs - 1) == 0 else 2 ** num_coeffs.bit_length()
    #     # print(size)
    #     log_size_of_group = size.bit_length()-1
        
    #     # Check if log_size_of_group exceeds TWO_ADICITY
    #     if log_size_of_group > fr.Fr.TWO_ADICITY:
    #         return None

    #     # Compute the generator for the multiplicative subgroup.
    #     # It should be the 2^(log_size_of_group) root of unity.
    #     group_gen = fr.Fr.get_root_of_unity(size)
        
    #     # Check that it is indeed the 2^(log_size_of_group) root of unity.
    #     group_gen_pow = group_gen.pow(size)
    #     assert group_gen_pow == fr.Fr.one()

    #     size_as_field_element=fr.Fr.from_repr(size)
    #     size_inv = fr.Fr.inverse(size_as_field_element)
    #     group_gen_inv = fr.Fr.inverse(group_gen)
    #     generator_inv = fr.Fr.inverse(fr.Fr.multiplicative_generator())

    #     return cls(size, log_size_of_group, size_as_field_element, size_inv, group_gen, group_gen_inv, generator_inv)
    
    # Evaluate all Lagrange polynomials at tau to get the lagrange coefficients.
    # Define the following as
    # - H: The coset we are in, with generator g and offset h
    # - m: The size of the coset H
    # - Z_H: The vanishing polynomial for H. Z_H(x) = prod_{i in m} (x - hg^i) = x^m - h^m
    # - v_i: A sequence of values, where v_0 = 1/(m * h^(m-1)), and v_{i + 1} = g * v_i
    #
    # We then compute L_{i,H}(tau) as `L_{i,H}(tau) = Z_H(tau) * v_i / (tau - h g^i)`
    #
    # However, if tau in H, both the numerator and denominator equal 0
    # when i corresponds to the value tau equals, and the coefficient is 0 everywhere else.
    # We handle this case separately, and we can easily detect by checking if the vanishing poly is 0.
    def evaluate_all_lagrange_coefficients(self, tau):
        size = self.size
        group_gen = self.group_gen
        tau = tau.to("cpu")
        t_size = F.exp_mod(tau, size)
        zero = fr.Fr.zero()
        one = fr.Fr.one()
        mod = fr.Fr.MODULUS
        domain_offset = one.clone()
        z_h_at_tau = F.sub_mod(t_size, domain_offset)

        if torch.equal(z_h_at_tau, zero):
            u = zero.repeat(size, 1)
            omega_i = domain_offset
            for i in range(size):
                if torch.equal(omega_i, tau):
                    u[i] = one.clone()
                    break
                omega_i = F.mul_mod(omega_i, group_gen)
            return u
        else:
            
            # In this case we have to compute `Z_H(tau) * v_i / (tau - h g^i)`
            # for i in 0..size
            # We actually compute this by computing (Z_H(tau) * v_i)^{-1} * (tau - h g^i)
            # and then batch inverting to get the correct lagrange coefficients.
            # We let `l_i = (Z_H(tau) * v_i)^-1` and `r_i = tau - h g^i`
            # Notice that since Z_H(tau) is i-independent,
            # and v_i = g * v_{i-1}, it follows that
            # l_i = g^-1 * l_{i-1}
            # TODO: consider caching the computation of l_i to save N multiplications
            from .arithmetic import batch_inversion

            tau = tau.to("cuda")
            mod = mod.to("cuda")
            group_gen = group_gen.to("cuda")
            f_size = fr.Fr.from_repr(size).to("cuda")
            domain_offset = domain_offset.to("cuda")
            z_h_at_tau = z_h_at_tau.to("cuda")

            pow_dof = F.exp_mod(domain_offset, size - 1) 
            v_0_inv = F.mul_mod(f_size, pow_dof)
            v_0 = F.inv_mod(v_0_inv)
            coeff_v = F.gen_sequence(size, group_gen)
            coeff_v = F.mul_mod_scalar(coeff_v, v_0)
            nominator = F.mul_mod_scalar(coeff_v, z_h_at_tau)

            negative_cur_elem = F.sub_mod(mod, domain_offset)
            r_0 = F.add_mod(tau, negative_cur_elem)
            coeff_r = F.gen_sequence(size, group_gen)
            coeff_r = F.mul_mod_scalar(coeff_r, r_0)
            coeff_tau = tau.repeat(size, 1)
            denominator = F.sub_mod(coeff_tau, coeff_r)
            denominator_inv = F.inv_mod(denominator)

            lagrange_coefficients = F.mul_mod(nominator, denominator_inv)
            # l_i = F.mul_mod(z_h_at_tau_inv, v_0_inv)
            
            # mod = mod.to("cuda")
            # domain_offset = domain_offset.to("cuda")
            # negative_cur_elem = negative_cur_elem.to("cuda")
            # lagrange_coefficients_inverse = zero.repeat(size, 1)
            # group_gen = self.group_gen.to("cuda")
            # group_gen_inv = self.group_gen_inv.to("cuda")
            # #TODO
            # for i in range(size):
            #     r_i = F.add_mod(tau, negative_cur_elem)
            #     lagrange_coefficients_inverse[i] = F.mul_mod(l_i, r_i)
            #     # Increment l_i and negative_cur_elem
            #     l_i = F.mul_mod(l_i, group_gen_inv)
            #     negative_cur_elem = F.mul_mod(negative_cur_elem, group_gen)
            # lagrange_coefficients_inverse = lagrange_coefficients_inverse.to('cuda')
            # res = batch_inversion(lagrange_coefficients_inverse)
            return lagrange_coefficients

    
    # This evaluates the vanishing polynomial for this domain at tau.
    # For multiplicative subgroups, this polynomial is `z(X) = X^self.size - 1`.
    def evaluate_vanishing_polynomial(self, tau):
        one = fr.Fr.one()
        pow_tau = F.exp_mod(tau, self.size)
        return F.sub_mod(pow_tau, one)
    
    # Returns the `i`-th element of the domain, where elements are ordered by
    # their power of the generator which they correspond to.
    # e.g. the `i`-th element is g^i
    def element(self, i):
        # TODO: Consider precomputed exponentiation tables if we need this to be faster.
        res = self.group_gen.clone()
        for j in range(i):
            res = F.mul_mod(res, self.group_gen)
        return res
    
        