# Modified from https://github.com/echocatzh/conv-stft/blob/master/conv_stft/conv_stft.py

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# MIT License

# Copyright (c) 2020 Shimin Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch as th
import torch.nn.functional as F
from scipy.signal import check_COLA, get_window

support_clp_op = None
if th.__version__ >= "1.7.0":
    from torch.fft import rfft as fft

    support_clp_op = True
else:
    from torch import rfft as fft


class STFT(th.nn.Module):
    def __init__(
        self,
        win_len=1024,
        win_hop=512,
        fft_len=1024,
        enframe_mode="continue",
        win_type="hann",
        win_sqrt=False,
        pad_center=True,
    ):
        """
        Implement of STFT using 1D convolution and 1D transpose convolutions.
        Implement of framing the signal in 2 ways, `break` and `continue`.
        `break` method is a kaldi-like framing.
        `continue` method is a librosa-like framing.

        More information about `perfect reconstruction`:
        1. https://ww2.mathworks.cn/help/signal/ref/stft.html
        2. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html

        Args:
            win_len (int): Number of points in one frame.  Defaults to 1024.
            win_hop (int): Number of framing stride. Defaults to 512.
            fft_len (int): Number of DFT points. Defaults to 1024.
            enframe_mode (str, optional): `break` and `continue`. Defaults to 'continue'.
            win_type (str, optional): The type of window to create. Defaults to 'hann'.
            win_sqrt (bool, optional): using square root window. Defaults to True.
            pad_center (bool, optional): `perfect reconstruction` opts. Defaults to True.
        """
        super(STFT, self).__init__()
        assert enframe_mode in ["break", "continue"]
        assert fft_len >= win_len
        self.win_len = win_len
        self.win_hop = win_hop
        self.fft_len = fft_len
        self.mode = enframe_mode
        self.win_type = win_type
        self.win_sqrt = win_sqrt
        self.pad_center = pad_center
        self.pad_amount = self.fft_len // 2

        en_k, fft_k, ifft_k, ola_k = self.__init_kernel__()
        self.register_buffer("en_k", en_k)
        self.register_buffer("fft_k", fft_k)
        self.register_buffer("ifft_k", ifft_k)
        self.register_buffer("ola_k", ola_k)

    def __init_kernel__(self):
        """
        Generate enframe_kernel, fft_kernel, ifft_kernel and overlap-add kernel.
        ** enframe_kernel: Using conv1d layer and identity matrix.
        ** fft_kernel: Using linear layer for matrix multiplication. In fact,
        enframe_kernel and fft_kernel can be combined, But for the sake of
        readability, I took the two apart.
        ** ifft_kernel, pinv of fft_kernel.
        ** overlap-add kernel, just like enframe_kernel, but transposed.

        Returns:
            tuple: four kernels.
        """
        enframed_kernel = th.eye(self.fft_len)[:, None, :]
        if support_clp_op:
            tmp = fft(th.eye(self.fft_len))
            fft_kernel = th.stack([tmp.real, tmp.imag], dim=2)
        else:
            fft_kernel = fft(th.eye(self.fft_len), 1)
        if self.mode == "break":
            enframed_kernel = th.eye(self.win_len)[:, None, :]
            fft_kernel = fft_kernel[: self.win_len]
        fft_kernel = th.cat((fft_kernel[:, :, 0], fft_kernel[:, :, 1]), dim=1)
        ifft_kernel = th.pinverse(fft_kernel)[:, None, :]
        window = get_window(self.win_type, self.win_len)

        self.perfect_reconstruct = check_COLA(window, self.win_len, self.win_len - self.win_hop)
        window = th.FloatTensor(window)
        if self.mode == "continue":
            left_pad = (self.fft_len - self.win_len) // 2
            right_pad = left_pad + (self.fft_len - self.win_len) % 2
            window = F.pad(window, (left_pad, right_pad))
        if self.win_sqrt:
            self.padded_window = window
            window = th.sqrt(window)
        else:
            self.padded_window = window**2

        fft_kernel = fft_kernel.T * window
        ifft_kernel = ifft_kernel * window
        ola_kernel = th.eye(self.fft_len)[: self.win_len, None, :]
        if self.mode == "continue":
            ola_kernel = th.eye(self.fft_len)[:, None, : self.fft_len]
        return enframed_kernel, fft_kernel, ifft_kernel, ola_kernel

    def is_perfect(self):
        """
        Whether the parameters win_len, win_hop and win_sqrt
        obey constants overlap-add(COLA)

        Returns:
            bool: Return true if parameters obey COLA.
        """
        return self.perfect_reconstruct and self.pad_center

    def transform(self, inputs, return_type="complex"):
        """Take input data (audio) to STFT domain.

        Args:
            inputs (tensor): Tensor of floats, with shape (num_batch, num_samples)
            return_type (str, optional): return (mag, phase) when `magphase`,
            return (real, imag) when `realimag` and complex(real, imag) when `complex`.
            Defaults to 'complex'.

        Returns:
            tuple: (mag, phase) when `magphase`, return (real, imag) when
            `realimag`. Defaults to 'complex', each elements with shape
            [num_batch, num_frequencies, num_frames]
        """
        assert return_type in ["magphase", "realimag", "complex"]
        if inputs.dim() == 2:
            inputs = th.unsqueeze(inputs, 1)
        self.num_samples = inputs.size(-1)
        if self.pad_center:
            inputs = F.pad(inputs, (self.pad_amount, self.pad_amount), mode="reflect")
        enframe_inputs = F.conv1d(inputs, self.en_k, stride=self.win_hop)
        outputs = th.transpose(enframe_inputs, 1, 2)
        outputs = F.linear(outputs, self.fft_k)
        outputs = th.transpose(outputs, 1, 2)
        dim = self.fft_len // 2 + 1
        real = outputs[:, :dim, :]
        imag = outputs[:, dim:, :]
        if return_type == "realimag":
            return real, imag
        elif return_type == "complex":
            assert support_clp_op
            return th.complex(real, imag)
        else:
            mags = th.sqrt(real**2 + imag**2)
            phase = th.atan2(imag, real)
            return mags, phase

    def inverse(self, input1, input2=None, input_type="magphase"):
        """Call the inverse STFT (iSTFT), given tensors produced
        by the `transform` function.

        Args:
            input1 (tensors): Magnitude/Real-part of STFT with shape
            [num_batch, num_frequencies, num_frames]
            input2 (tensors): Phase/Imag-part of STFT with shape
            [num_batch, num_frequencies, num_frames]
            input_type (str, optional): Mathematical meaning of input tensor's.
            Defaults to 'magphase'.

        Returns:
            tensors: Reconstructed audio given magnitude and phase. Of
                shape [num_batch, num_samples]
        """
        assert input_type in ["magphase", "realimag"]
        if input_type == "realimag":
            real, imag = None, None
            if support_clp_op and th.is_complex(input1):
                real, imag = input1.real, input1.imag
            else:
                real, imag = input1, input2
        else:
            real = input1 * th.cos(input2)
            imag = input1 * th.sin(input2)
        inputs = th.cat([real, imag], dim=1)
        outputs = F.conv_transpose1d(inputs, self.ifft_k, stride=self.win_hop)
        t = (self.padded_window[None, :, None]).repeat(1, 1, inputs.size(-1))
        t = t.to(inputs.device)
        coff = F.conv_transpose1d(t, self.ola_k, stride=self.win_hop)

        num_frames = input1.size(-1)
        num_samples = num_frames * self.win_hop

        rm_start, rm_end = self.pad_amount, self.pad_amount + num_samples

        outputs = outputs[..., rm_start:rm_end]
        coff = coff[..., rm_start:rm_end]
        coffidx = th.where(coff > 1e-8)
        outputs[coffidx] = outputs[coffidx] / (coff[coffidx])
        return outputs.squeeze(dim=1)

    def forward(self, inputs):
        """Take input data (audio) to STFT domain and then back to audio.

        Args:
            inputs (tensor): Tensor of floats, with shape [num_batch, num_samples]

        Returns:
            tensor: Reconstructed audio given magnitude and phase.
            Of shape [num_batch, num_samples]
        """
        mag, phase = self.transform(inputs)
        rec_wav = self.inverse(mag, phase)
        return rec_wav
