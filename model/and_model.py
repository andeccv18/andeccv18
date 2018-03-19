"""
This file holds the model employed for the ShanghaiTech dataset
experiments.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from operator import mul
from itertools import chain


class MaskedConv2d(nn.Conv2d):
    """
    A masked convolutional layer, described by Eq. (4)
    """

    def __init__(self, mask_type, idx, *args, **kwargs):
        """
        Class constructor.

        Parameters
        ----------
        mask_type: str
            connectivity tipe, among [`A`, `B`].
        idx: int
            the index of the convolution inside the
            MaskedStackedConvolutional layer.
        """

        super(MaskedConv2d, self).__init__(*args, **kwargs)

        # Build mask
        assert mask_type in ['A', 'B']
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kt, kd = self.weight.size()
        assert kt == 3
        self.mask.fill_(0)
        self.mask[:, :, :kt // 2, :] = 1
        if idx + (mask_type == 'B') > 0:
            self.mask[:, :, kt // 2, :idx + (mask_type == 'B')] = 1

    def forward(self, x):
        """
        Forward function.
        """

        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class ListModule(nn.Module):
    """
    Pytorch workaround for having modules with
    lists of non-sequential modules within.
    See [1] for details.
    """

    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class MaskedStackedConvolutions(nn.Module):
    """
    Masked Stacked Convolutional layer, as described by Eq. 5.
    """

    def __init__(self, mask_type, code_length, in_channels, out_channels):
        """
        Class constructor.

        Parameters
        ----------
        mask_type: str
            type of masked convolutions within the layer, among [`A`, `B`].
        code_length: int
            dimensionality of feature maps along the code dimension.
        in_channels: int
            channels of the input feature maps.
        out_channels: int
            channels of the output feature maps.
        """

        super(MaskedStackedConvolutions, self).__init__()
        self._mask_type = mask_type
        self._code_length = code_length
        self._in_channels = in_channels
        self._out_channels = out_channels

        # Build inner ListModule
        layers = []
        for i in range(0, code_length):
            layers.append(
                MaskedConv2d(mask_type=mask_type,
                             idx=i,
                             in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=(3, code_length),
                             padding=(1, 0))
            )
        self._conv_layers = ListModule(*layers)

    def forward(self, x):
        """
        Forward function.
        """

        # Apply convolutions
        out = []
        for i in range(0, self._code_length):
            out.append(self._conv_layers[i](x))

        # Concat
        out = torch.cat(out, dim=-1)

        return out

    @property
    def n_parameters(self):
        """
        Returns
        -------
        int
            the number of trainable parameters.
        """

        # Count parameters
        n = 0
        for l in self._conv_layers:
            n += torch.sum(l.mask)  # do not count masked params
            n += reduce(mul, l.bias.size())
        return n

    def __repr__(self):
        """
        String representation.
        """
        return self.__class__.__name__ + '(' \
               + 'mask_type=' + str(self._mask_type) \
               + ', code_length=' + str(self._code_length)\
               + ', in_channels=' + str(self._in_channels) \
               + ', out_features=' + str(self._out_channels) + ')'


class ANDEstimator2D(nn.Module):
    """ This class models the estimation network. """

    def __init__(self, code_length, autoregression_bins):
        """
        Class constructor.

        Parameters
        ----------
        code_length: int
            dimensionality of feature maps along the code dimension.
        autoregression_bins: int
            the number of autoregression bins.
        """

        super(ANDEstimator2D, self).__init__()

        self._code_length = code_length
        self._autoregression_bins = autoregression_bins

        # Build autoregression layers (Tab 1, `AR layers`)
        self._layers = nn.Sequential(
            MaskedStackedConvolutions(mask_type='A', code_length=code_length, in_channels=1, out_channels=4),
            nn.LeakyReLU(),
            MaskedStackedConvolutions(mask_type='B', code_length=code_length, in_channels=4, out_channels=4),
            nn.LeakyReLU(),
            MaskedStackedConvolutions(mask_type='B', code_length=code_length, in_channels=4,
                                      out_channels=autoregression_bins),
        )

    def forward(self, x):
        """
        Forward function
        """

        h = torch.unsqueeze(x, dim=1)  # add singleton channel dim
        h = self._layers(h)
        o = h

        return o

    def __call__(self, *args, **kwargs):
        return super(ANDEstimator2D, self).__call__(*args, **kwargs)

    @property
    def n_parameters(self):
        """
        Returns
        -------
        int
            the number of trainable parameters.
        """

        # Count
        n = 0
        for l in self._layers:
            if l.__class__.__name__ == 'MaskedStackedConvolutions':
                n += l.n_parameters
        return int(n)

    def __repr__(self):
        """
        String representation.
        """

        good_old = super(ANDEstimator2D, self).__repr__()
        addition = 'Total number of parameters: {:,}'.format(self.n_parameters)

        return good_old + '\n' + addition


class TimeDistributedDense(nn.Module):
    """
    Applies the same fully connected layer to all time-steps.
    """

    def __init__(self, in_features, out_features, bias):
        """
        Class constructor.

        Parameters
        ----------
        in_features: int
            number of input features.
        out_features: int
            number of output features.
        bias: bool
            whether or not to use bias.
        """

        super(TimeDistributedDense, self).__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._bias = bias

        # The layer to be applied at each time-step
        self._linear = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.BatchNorm1d(num_features=out_features)
        )

    def forward(self, x):
        """
        Forward function.
        """

        b, t, d = x.size()

        output = []
        for i in range(0, t):
            # Apply dense layer
            output.append(self._linear(x[:, i, :]))
        output = torch.stack(output, 1)

        return output


class MaskedConv3d(nn.Conv3d):
    """
    Masked 3D convolution, used in encoder blocks.
    """

    def __init__(self, *args, **kwargs):
        """
        Class constructor.
        """

        super(MaskedConv3d, self).__init__(*args, **kwargs)

        # Build the mask
        self.register_buffer('mask', self.weight.data.clone())
        _, _, k_t, k_h, k_w = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, k_t // 2 + 1:] = 0

    def forward(self, x):
        """
        Forward function.
        """

        self.weight.data *= self.mask
        return super(MaskedConv3d, self).forward(x)

    def __call__(self, *args, **kwargs):
        return super(MaskedConv3d, self).__call__(*args, **kwargs)


class EncoderBlock(nn.Module):
    """
    Encoder block, as described in Fig. 3, video model
    """

    def __init__(self, channel_in, channel_out, stride):
        """
        Class constructor.

        Parameters
        ----------
        channel_in: int
            number of input channels.
        channel_out: int
            number of output channels.
        stride: tuple
            strides applied by the first convolution of each path.
        """

        super(EncoderBlock, self).__init__()

        self._channel_in = channel_in
        self._channel_out = channel_out

        # Build layers
        self.conv1a = MaskedConv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
                                   padding=1, stride=stride, bias=True)
        self.conv1b = nn.Conv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=1,
                                padding=0, stride=stride, bias=True)
        self.conv2 = MaskedConv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                  padding=1, stride=1, bias=True)
        self.conv3 = MaskedConv3d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                  padding=1, stride=1, bias=True)

        self.bn1a = nn.BatchNorm3d(num_features=channel_out, momentum=0.9)
        self.bn1b = nn.BatchNorm3d(num_features=channel_out, momentum=0.9)
        self.bn2 = nn.BatchNorm3d(num_features=channel_out, momentum=0.9)
        self.bn3 = nn.BatchNorm3d(num_features=channel_out, momentum=0.9)

    def forward(self, x):
        """
        Forward function.
        """

        # Conv branch
        ha = x
        ha = self.conv1a(ha)
        ha = self.bn1a(ha)
        ha = F.leaky_relu(ha)
        ha = self.conv2(ha)
        ha = self.bn2(ha)
        ha = F.leaky_relu(ha)
        ha = self.conv3(ha)
        ha = self.bn3(ha)

        # Identity branch
        hb = x
        hb = self.conv1b(hb)
        hb = self.bn1b(hb)

        # Residual connection
        out = ha + hb

        return out


class DecoderBlock(nn.Module):
    """
    Decoder block, as described in Fig. 3, video model
    """

    def __init__(self, channel_in, channel_out, stride, output_padding):
        """
        Class constructor.

        Parameters
        ----------
        channel_in: int
            number of input channels.
        channel_out: int
            number of output channels.
        stride: tuple
            strides applied by the first transposed convolution of each path.
        output_padding: tuple
            output padding applied by the first transposed convolution of each path.
        """

        super(DecoderBlock, self).__init__()

        self._channel_in = channel_in
        self._channel_out = channel_out

        # Build layers
        self.conv1a = nn.ConvTranspose3d(channel_in, channel_out, kernel_size=5, padding=2, stride=stride,
                                         output_padding=output_padding, bias=True)
        self.conv1b = nn.ConvTranspose3d(channel_in, channel_out, kernel_size=1, padding=0, stride=stride,
                                         output_padding=output_padding, bias=True)
        self.conv2 = nn.Conv3d(channel_out, channel_out, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv3 = nn.Conv3d(channel_out, channel_out, kernel_size=3, padding=1, stride=1, bias=True)

        self.bn1a = nn.BatchNorm3d(channel_out, momentum=0.9)
        self.bn1b = nn.BatchNorm3d(channel_out, momentum=0.9)
        self.bn2 = nn.BatchNorm3d(channel_out, momentum=0.9)
        self.bn3 = nn.BatchNorm3d(channel_out, momentum=0.9)

    def forward(self, x):
        """
        Forward function.
        """

        # Conv branch
        ha = x
        ha = self.conv1a(ha)
        ha = self.bn1a(ha)
        ha = F.leaky_relu(ha)
        ha = self.conv2(ha)
        ha = self.bn2(ha)
        ha = F.leaky_relu(ha)
        ha = self.conv3(ha)
        ha = self.bn3(ha)

        # Identity branch
        hb = x
        hb = self.conv1b(hb)
        hb = self.bn1b(hb)

        # Residual connection
        out = ha + hb

        return out


class ANDEncoder(nn.Module):
    """
    Encoder network.
    """

    def __init__(self, input_shape, code_length):
        """
        Class constructor.

        Parameters
        ----------
        input_shape: tuple
            shape of input clips.
        code_length: int
            dimensionality of the compressed representation.
        """

        super(ANDEncoder, self).__init__()

        self._input_shape = input_shape
        self._code_length = code_length

        self._in_c, self._in_t, self._in_h, self._in_w = input_shape

        # Build layers as in Tab 1, ShanghaiTech
        c, t, h, w = input_shape
        channels_in = [c, 8, 16, 32, 64]
        channels_out = [8, 16, 32, 64, 64]
        strides = [(2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]

        layers_list = []
        for c_in, c_out, stride in zip(channels_in, channels_out, strides):
            layers_list.append(EncoderBlock(c_in, c_out, stride))
            c = c_out
            t, h, w = (x1 // x2 for x1, x2 in zip((t, h, w), stride))

        self._deepest_shape = (c, t, h, w)
        self._deepest_time_steps = t

        self.conv = nn.Sequential(*layers_list)
        self.tdd = nn.Sequential(TimeDistributedDense(in_features=(c * h * w),
                                                      out_features=512,
                                                      bias=True),
                                 nn.Tanh(),
                                 TimeDistributedDense(in_features=512,
                                                      out_features=code_length,
                                                      bias=True),
                                 nn.Sigmoid())

    def forward(self, x):
        """
        Forward function.
        """

        h = x
        h = self.conv(h)

        # reshape for tdd
        c, t, height, width = self._deepest_shape
        h = torch.transpose(h, 1, 2).contiguous()
        h = h.view(-1, t, (c * height * width))
        o = self.tdd(h)

        return o

    @property
    def deepest_shape(self):
        """
        Returns
        -------
        tuple
            dimensionalty of the deepest convolutional map.
        """

        return self._deepest_shape

    @property
    def deepest_time_steps(self):
        """
        Returns
        -------
        int
            number of time-steps in the deepest convolutional map.
        """

        return self._deepest_time_steps

    def __call__(self, *args, **kwargs):
        return super(ANDEncoder, self).__call__(*args, **kwargs)


class ANDDecoder(nn.Module):
    """
    Decoder network.
    """

    def __init__(self, code_length, deepest_shape, output_shape):
        """
        Class constructor.

        Parameters
        ----------
        code_length: int
            dimensionality of the compressed representation.
        deepest_shape: tuple
            shape the deepest encoder feature map.
        output_shape: tuple
            the desired shape of the output, equal to the input clip.
        """

        super(ANDDecoder, self).__init__()

        self._code_length = code_length
        self._deepest_shape = deepest_shape
        self._output_shape = output_shape

        # Build layers as in Tab 1, ShanghaiTech
        c, t, h, w = deepest_shape
        self.tdd = nn.Sequential(TimeDistributedDense(in_features=code_length,
                                                      out_features=512,
                                                      bias=True),
                                 nn.Tanh(),
                                 TimeDistributedDense(in_features=512,
                                                      out_features=(c * h * w),
                                                      bias=True),
                                 nn.Tanh())

        channels_in = [64, 64, 32, 16, 8]
        channels_out = [64, 32, 16, 8, 8]
        strides = [(1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2)]
        output_paddings = [(0, 1, 1), (0, 1, 1), (0, 1, 1), (1, 1, 1), (1, 1, 1)]

        layers_list = []
        out_c = 0
        for c_in, c_out, stride, out_pad in zip(channels_in, channels_out, strides, output_paddings):
            layers_list.append(DecoderBlock(c_in, c_out, stride, out_pad))
            out_c = c_out
            t, h, w = (x1 * x2 for x1, x2 in zip((t, h, w), stride))

        # final conv to get 1 channels
        layers_list.append(nn.Sequential(
            nn.Conv3d(in_channels=out_c, out_channels=output_shape[0], kernel_size=1, stride=1, padding=0, bias=True),
        ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, x):
        """
        Forward function.
        """

        h = x
        h = self.tdd(h)

        # reshape
        h = torch.transpose(h, 1, 2).contiguous()
        h = h.view(len(h), *self._deepest_shape)

        h = self.conv(h)
        o = h

        return o

    def __call__(self, *args, **kwargs):
        return super(ANDDecoder, self).__call__(*args, **kwargs)


class AND(nn.Module):
    """
    Autoregressive anomaly detector model.
    """

    def __init__(self, input_shape, code_length, autoregression_bins):
        """
        Class constructor.

        Parameters
        ----------
        input_shape: tuple
            shape of input clips.
        code_length: int
            dimensionality of the compressed representation.
        autoregression_bins: int
            number of autoregression bins used for estimation.
        """

        super(AND, self).__init__()

        self._input_shape = input_shape
        self._code_length = code_length
        self._autoregression_bins = autoregression_bins
        self._flatten_input_shape = reduce(mul, input_shape)

        # Build encoder
        self._encoder = ANDEncoder(input_shape=input_shape, code_length=code_length)

        # Build decoder
        self._decoder = ANDDecoder(code_length=code_length,
                                   deepest_shape=self._encoder.deepest_shape,
                                   output_shape=input_shape)

        # Build estimator
        self._estimator = ANDEstimator2D(code_length=code_length, autoregression_bins=autoregression_bins)

    def forward(self, x):
        """
        Forward function.
        """

        h = x

        z = self._encoder(h)

        z_dist = self._estimator(z)

        x_r = self._decoder(z)
        x_r = x_r.view(-1, *self._input_shape)

        return x_r, z, z_dist

    def __call__(self, *args, **kwargs):
        return super(AND, self).__call__(*args, **kwargs)

    def __repr__(self):
        """
        String representation.
        """

        good_old = super(AND, self).__repr__()

        n_parameters = sum([reduce(mul, p.size()) for p in self.parameters()])
        addition = 'Total number of parameters: {:,}'.format(n_parameters)

        return good_old + '\n' + addition

"""
References
[1]: 'https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/4'
"""
