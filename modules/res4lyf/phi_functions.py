# Copyright 2025 The RES4LYF Team (Clybius) and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
from mpmath import exp as mp_exp
from mpmath import factorial as mp_factorial
from mpmath import mp, mpf

# Set precision for mpmath
mp.dps = 80


def calculate_gamma(c2: float, c3: float) -> float:
    """Calculates the gamma parameter for RES 3s samplers."""
    return (3 * (c3**3) - 2 * c3) / (c2 * (2 - 3 * c2))


def _torch_factorial(n: int) -> float:
    return float(math.factorial(n))


def phi_standard_torch(j: int, neg_h: torch.Tensor) -> torch.Tensor:
    r"""
    Standard implementation of phi functions using torch.
    ϕj(-h) = (e^(-h) - \sum_{k=0}^{j-1} (-h)^k / k!) / (-h)^j
    For h=0, ϕj(0) = 1/j!
    """
    assert j > 0

    # Handle h=0 case
    if torch.all(neg_h == 0):
        return torch.full_like(neg_h, 1.0 / _torch_factorial(j))

    # We use double precision for the series to avoid early overflow/precision loss
    orig_dtype = neg_h.dtype
    neg_h = neg_h.to(torch.float64)

    # For very small h, use series expansion to avoid 0/0
    if torch.any(torch.abs(neg_h) < 1e-4):
        # 1/j! + z/(j+1)! + z^2/(2!(j+2)!) ...
        result = torch.full_like(neg_h, 1.0 / _torch_factorial(j))
        term = torch.full_like(neg_h, 1.0 / _torch_factorial(j))
        for k in range(1, 5):
            term = term * neg_h / (j + k)
            result += term
        return result.to(orig_dtype)

    remainder = torch.zeros_like(neg_h)
    for k in range(j):
        remainder += (neg_h**k) / _torch_factorial(k)

    phi_val = (torch.exp(neg_h) - remainder) / (neg_h**j)
    return phi_val.to(orig_dtype)


def phi_mpmath_series(j: int, neg_h: float) -> float:
    """Arbitrary-precision phi_j(-h) via series definition."""
    j = int(j)
    z = mpf(float(neg_h))

    # Handle h=0 case: phi_j(0) = 1/j!
    if z == 0:
        return float(1.0 / mp_factorial(j))

    s_val = mp.mpf("0")
    for k in range(j):
        s_val += (z**k) / mp_factorial(k)
    phi_val = (mp_exp(z) - s_val) / (z**j)
    return float(phi_val)


class Phi:
    """
    Class to manage phi function calculations and caching.
    Supports both standard torch-based and high-precision mpmath-based solutions.
    """

    def __init__(self, h: torch.Tensor, c: list[float | mpf], analytic_solution: bool = True):
        self.h = h
        self.c = c
        self.cache: dict[tuple[int, int], float | torch.Tensor] = {}
        self.analytic_solution = analytic_solution

        if analytic_solution:
            self.phi_f = phi_mpmath_series
            self.h_mpf = mpf(float(h))
            self.c_mpf = [mpf(float(c_val)) for c_val in c]
        else:
            self.phi_f = phi_standard_torch

    def __call__(self, j: int, i: int = -1) -> float | torch.Tensor:
        if (j, i) in self.cache:
            return self.cache[(j, i)]

        if i < 0:
            c_val = 1.0
        else:
            c_val = self.c[i - 1]
            if c_val == 0:
                self.cache[(j, i)] = 0.0
                return 0.0

        if self.analytic_solution:
            h_val = self.h_mpf
            c_mapped = self.c_mpf[i - 1] if i >= 0 else 1.0

            if j == 0:
                result = float(mp_exp(-h_val * c_mapped))
            else:
                # Use the mpmath internal function for higher precision
                z = -h_val * c_mapped
                if z == 0:
                    result = float(1.0 / mp_factorial(j))
                else:
                    s_val = mp.mpf("0")
                    for k in range(j):
                        s_val += (z**k) / mp_factorial(k)
                    result = float((mp_exp(z) - s_val) / (z**j))
        else:
            h_val = self.h
            c_mapped = float(c_val)

            if j == 0:
                result = torch.exp(-h_val * c_mapped)
            else:
                result = self.phi_f(j, -h_val * c_mapped)

        self.cache[(j, i)] = result
        return result
