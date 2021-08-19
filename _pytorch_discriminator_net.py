# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Discriminator
"""

import logging
from qiskit.exceptions import MissingOptionalLibraryError

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
except ImportError:
    if logger.isEnabledFor(logging.INFO):
        EXC = MissingOptionalLibraryError(
            libname="Pytorch",
            name="DiscriminatorNet",
            pip_install="pip install 'qiskit-machine-learning[torch]'",
        )
        logger.info(str(EXC))

# torch 1.6.0 fixed a mypy error about not applying contravariance rules
# to inputs by defining forward as a value, rather than a function.  See also
# https://github.com/python/mypy/issues/8795
# The fix introduced an error on Module class about '_forward_unimplemented'
# not being implemented.
# The pylint disable=abstract-method fixes it.


class DiscriminatorNet(torch.nn.Module):  # pylint: disable=abstract-method
    """
    Discriminator
    """

    def __init__(self, n_hidden0: int=50,
                    n_hidden1: int = 20,
                    include_bias: bool = False) -> None:
        """
        Initialize the discriminator network.

        Args:
            n_features: Dimension of input data samples.
            n_out: n out
        """
        # supports only 1 dimension:
        n_features = 1
        n_out = 1

        # if n_hidden0 == None:
        #     n_hidden0 = 50
        # if n_hidden1 == None:
        #     n_hidden1 = 20
        # if include_bias == None:
        #     include_bias = True

        super().__init__()

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, n_hidden0,bias=include_bias),
            nn.LeakyReLU(0.2),
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(n_hidden0, n_hidden1, bias=include_bias),
            nn.LeakyReLU(0.2),
        )
        self.out = nn.Sequential(nn.Linear(n_hidden1, n_out, bias=include_bias), nn.Sigmoid())

    def forward(self, x):  # pylint: disable=arguments-differ
        """

        Args:
            x (torch.Tensor): Discriminator input, i.e. data sample.

        Returns:
            torch.Tensor: Discriminator output, i.e. data label.
        """
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.out(x)

        return x
