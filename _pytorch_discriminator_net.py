# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
# 
# (C) Changed by Eden Schirman in August 2021. (schirman.eden@gmail.com)
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

    def __init__(self, n_hidden0: int=512,
                    n_hidden1: int = 256,
                    include_bias: bool = False,
                    dropouts: bool = False,
                    conv_net: bool = False,
                    third_layer: bool = False) -> None:
        """
        Initialize the discriminator network.

        Args:
            n_features: Dimension of input data samples.
            n_out: n out
        """
        super().__init__()
        # supports only 1 dimension:
        n_features = 1
        n_out = 1
        self.todo_dropouts = dropouts and not(conv_net)
        self.conv_net = conv_net
        self.third_layer = third_layer

        if self.third_layer:
            self.hidden_neg1 = nn.Sequential(
            nn.Linear(n_features, n_hidden1,bias=include_bias),
            nn.LeakyReLU(0.2))

            self.hidden0 = nn.Sequential(
            nn.Linear(n_hidden1, n_hidden0,bias=include_bias),
            nn.LeakyReLU(0.2))

            
        else:
            self.hidden0 = nn.Sequential(
                nn.Linear(n_features, n_hidden0,bias=include_bias),
                nn.LeakyReLU(0.2))
            
        if not self.conv_net:
            self.hidden1 = nn.Sequential(
                nn.Linear(n_hidden0, n_hidden1, bias=include_bias),
                nn.LeakyReLU(0.2))             
        else:
            #TODO currently this option does not work !
            raise Exception('Convolution layer is not supported yet.')
            self.hidden1 = nn.Sequential(
                nn.Conv1d(n_hidden0, n_hidden1,kernel_size=5, bias=include_bias),
                nn.LeakyReLU(0.2)) 
        
        self.out = nn.Sequential(nn.Linear(n_hidden1, n_out, bias=include_bias), nn.Sigmoid())

        if self.todo_dropouts:
            self.dropout = nn.Dropout()

    def forward(self, x):  # pylint: disable=arguments-differ
        """

        Args:
            x (torch.Tensor): Discriminator input, i.e. data sample.

        Returns:
            torch.Tensor: Discriminator output, i.e. data label.
        """
        if self.third_layer:
            x = self.hidden_neg1(x)
        if self.todo_dropouts:
            x = self.dropout(x)
        x = self.hidden0(x)
        if self.todo_dropouts:
            x = self.dropout(x)
        if self.conv_net:
            x = x.unsqueeze(dim=2)
        x = self.hidden1(x)
        if self.todo_dropouts:
            x = self.dropout(x)
        x = self.out(x)

        return x
