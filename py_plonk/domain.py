from .bls12_381 import fr
import torch
import torch.nn.functional as F


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

        def get_root_of_unity(n):
            assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
            log_size_of_group = n.bit_length() - 1
            assert log_size_of_group <= fr.TWO_ADICITY(), "logn must <= TWO_ADICITY"

            base = fr.TWO_ADIC_ROOT_OF_UNITY()
            exponent = 1 << (fr.TWO_ADICITY() - log_size_of_group)
            return F.exp_mod(base, exponent)


        # Compute the size of our evaluation domain
        size = num_coeffs if num_coeffs & (num_coeffs - 1) == 0 else 2 ** num_coeffs.bit_length()
        log_size_of_group = size.bit_length()-1
        
        # Check if log_size_of_group exceeds TWO_ADICITY
        if log_size_of_group > fr.TWO_ADICITY():
            return None

        # Compute the generator for the multiplicative subgroup.
        # It should be the 2^(log_size_of_group) root of unity.
        group_gen = get_root_of_unity(size)
        
        # Check that it is indeed the 2^(log_size_of_group) root of unity.
        group_gen_pow = F.exp_mod(group_gen,size)
        assert F.trace_equal(group_gen_pow, fr.one())

        size_as_field_element=fr.make_tensor(size)
        size_inv = F.inv_mod(size_as_field_element)
        group_gen_inv = F.inv_mod(group_gen)
        generator_inv = F.inv_mod(fr.GENERATOR())

        return cls(size, log_size_of_group, size_as_field_element, size_inv, group_gen, group_gen_inv, generator_inv)
    
    
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
        zero = fr.zero()
        one = fr.one()
        domain_offset = one.clone()
        z_h_at_tau = F.sub_mod(t_size, domain_offset)

        if F.trace_equal(z_h_at_tau, zero):
            u = zero.repeat(size, 1)
            omega_i = domain_offset
            for i in range(size):
                if F.trace_equal(omega_i, tau):
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


            f_size = fr.make_tensor(size)
            pow_dof = F.exp_mod(domain_offset, size - 1) 
            v_0_inv = F.mul_mod(f_size, pow_dof)
            v_0 = F.div_mod(one, v_0_inv)

            tau = tau.to("cuda")
            group_gen = group_gen.to("cuda")
            z_h_at_tau = z_h_at_tau.to("cuda")

            coeff_v = F.gen_sequence(size, group_gen)
            coeff_v = F.mul_mod_scalar(coeff_v, v_0.to("cuda"))
            nominator = F.mul_mod_scalar(coeff_v, z_h_at_tau)

            coeff_r = F.gen_sequence(size, group_gen)
            coeff_tau = tau.repeat(size, 1)
            denominator = F.sub_mod(coeff_tau, coeff_r)
            denominator_inv = F.inv_mod(denominator)

            lagrange_coefficients = F.mul_mod(nominator, denominator_inv)
            return lagrange_coefficients

    
    # This evaluates the vanishing polynomial for this domain at tau.
    # For multiplicative subgroups, this polynomial is `z(X) = X^self.size - 1`.
    def evaluate_vanishing_polynomial(self, tau):
        one = fr.one()
        pow_tau = F.exp_mod(tau, self.size)
        return F.sub_mod(pow_tau, one)
    
    # Returns the `i`-th element of the domain, where elements are ordered by
    # their power of the generator which they correspond to.
    # e.g. the `i`-th element is g^i
    def element(self, i):
        # TODO: Consider precomputed exponentiation tables if we need this to be faster.
        group_gen = self.group_gen.clone()
        coeff = F.gen_sequence(self.size, group_gen.to("cuda"))
        return coeff[i]
    
        