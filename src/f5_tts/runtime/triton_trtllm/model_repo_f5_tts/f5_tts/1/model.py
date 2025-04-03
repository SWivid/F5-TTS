# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import json
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.dlpack import from_dlpack, to_dlpack
import torchaudio
import jieba
import triton_python_backend_utils as pb_utils
from pypinyin import Style, lazy_pinyin
import os
from f5_tts_trtllm import F5TTS


def get_tokenizer(vocab_file_path: str):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    with open(vocab_file_path, "r", encoding="utf-8") as f:
        vocab_char_map = {}
        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i
    vocab_size = len(vocab_char_map)
    return vocab_char_map, vocab_size


def convert_char_to_pinyin(reference_target_texts_list, polyphone=True):
    final_reference_target_texts_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return "\u3100" <= c <= "\u9fff"  # common chinese characters

    for text in reference_target_texts_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_reference_target_texts_list.append(char_list)

    return final_reference_target_texts_list


def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
):  # noqa: F722
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    return list_idx_tensors


class TritonPythonModel:
    def initialize(self, args):
        self.use_perf = True
        self.device = torch.device("cuda")
        self.target_audio_sample_rate = 24000
        self.target_rms = 0.15  # target rms for audio
        self.n_fft = 1024
        self.win_length = 1024
        self.hop_length = 256
        self.n_mel_channels = 100
        self.max_mel_len = 3000
        self.head_dim = 64

        parameters = json.loads(args["model_config"])["parameters"]
        for key, value in parameters.items():
            parameters[key] = value["string_value"]

        self.vocab_char_map, self.vocab_size = get_tokenizer(parameters["vocab_file"])
        self.reference_sample_rate = int(parameters["reference_audio_sample_rate"])
        self.resampler = torchaudio.transforms.Resample(self.reference_sample_rate, self.target_audio_sample_rate)

        self.tllm_model_dir = parameters["tllm_model_dir"]
        config_file = os.path.join(self.tllm_model_dir, "config.json")
        with open(config_file) as f:
            config = json.load(f)
        self.model = F5TTS(
            config,
            debug_mode=False,
            tllm_model_dir=self.tllm_model_dir,
            model_path=parameters["model_path"],
            vocab_size=self.vocab_size,
        )

        self.vocoder = parameters["vocoder"]
        assert self.vocoder in ["vocos", "bigvgan"]
        if self.vocoder == "vocos":
            self.mel_stft = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.target_audio_sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mel_channels,
                power=1,
                center=True,
                normalized=False,
                norm=None,
            ).to(self.device)
            self.compute_mel_fn = self.get_vocos_mel_spectrogram
        elif self.vocoder == "bigvgan":
            self.compute_mel_fn = self.get_bigvgan_mel_spectrogram

    def get_vocos_mel_spectrogram(self, waveform):
        mel = self.mel_stft(waveform)
        mel = mel.clamp(min=1e-5).log()
        return mel.transpose(1, 2)

    def forward_vocoder(self, mel):
        mel = mel.to(torch.float32).contiguous().cpu()
        input_tensor_0 = pb_utils.Tensor.from_dlpack("mel", to_dlpack(mel))

        inference_request = pb_utils.InferenceRequest(
            model_name="vocoder", requested_output_names=["waveform"], inputs=[input_tensor_0]
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            waveform = pb_utils.get_output_tensor_by_name(inference_response, "waveform")
            waveform = torch.utils.dlpack.from_dlpack(waveform.to_dlpack()).cpu()

            return waveform

    def execute(self, requests):
        (
            reference_text_list,
            target_text_list,
            reference_target_texts_list,
            estimated_reference_target_mel_len,
            reference_mel_len,
        ) = [], [], [], [], []
        mel_features_list = []
        if self.use_perf:
            torch.cuda.nvtx.range_push("preprocess")
        for request in requests:
            wav_tensor = pb_utils.get_input_tensor_by_name(request, "reference_wav")
            wav_lens = pb_utils.get_input_tensor_by_name(request, "reference_wav_len")

            reference_text = pb_utils.get_input_tensor_by_name(request, "reference_text").as_numpy()
            reference_text = reference_text[0][0].decode("utf-8")
            reference_text_list.append(reference_text)
            target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
            target_text = target_text[0][0].decode("utf-8")
            target_text_list.append(target_text)

            text = reference_text + target_text
            reference_target_texts_list.append(text)

            wav = from_dlpack(wav_tensor.to_dlpack())
            wav_len = from_dlpack(wav_lens.to_dlpack())
            wav_len = wav_len.squeeze()
            assert wav.shape[0] == 1, "Only support batch size 1 for now."
            wav = wav[:, :wav_len]

            ref_rms = torch.sqrt(torch.mean(torch.square(wav)))
            if ref_rms < self.target_rms:
                wav = wav * self.target_rms / ref_rms
            if self.reference_sample_rate != self.target_audio_sample_rate:
                wav = self.resampler(wav)
            wav = wav.to(self.device)
            if self.use_perf:
                torch.cuda.nvtx.range_push("compute_mel")
            mel_features = self.compute_mel_fn(wav)
            if self.use_perf:
                torch.cuda.nvtx.range_pop()
            mel_features_list.append(mel_features)

            reference_mel_len.append(mel_features.shape[1])
            estimated_reference_target_mel_len.append(
                int(mel_features.shape[1] * (1 + len(target_text) / len(reference_text)))
            )

        max_seq_len = min(max(estimated_reference_target_mel_len), self.max_mel_len)

        batch = len(requests)
        mel_features = torch.zeros((batch, max_seq_len, self.n_mel_channels), dtype=torch.float16).to(self.device)
        for i, mel in enumerate(mel_features_list):
            mel_features[i, : mel.shape[1], :] = mel

        reference_mel_len_tensor = torch.LongTensor(reference_mel_len).to(self.device)

        pinyin_list = convert_char_to_pinyin(reference_target_texts_list, polyphone=True)
        text_pad_sequence = list_str_to_idx(pinyin_list, self.vocab_char_map)

        for i, item in enumerate(text_pad_sequence):
            text_pad_sequence[i] = F.pad(
                item, (0, estimated_reference_target_mel_len[i] - len(item)), mode="constant", value=-1
            )
            text_pad_sequence[i] += 1  # WAR: 0 is reserved for padding token, hard coding in F5-TTS
        text_pad_sequence = pad_sequence(text_pad_sequence, padding_value=-1, batch_first=True).to(self.device)
        text_pad_sequence = F.pad(
            text_pad_sequence, (0, max_seq_len - text_pad_sequence.shape[1]), mode="constant", value=-1
        )
        if self.use_perf:
            torch.cuda.nvtx.range_pop()

        denoised, cost_time = self.model.sample(
            text_pad_sequence,
            mel_features,
            reference_mel_len_tensor,
            estimated_reference_target_mel_len,
            remove_input_padding=False,
            use_perf=self.use_perf,
        )
        if self.use_perf:
            torch.cuda.nvtx.range_push("vocoder")

        responses = []
        for i in range(batch):
            ref_me_len = reference_mel_len[i]
            estimated_mel_len = estimated_reference_target_mel_len[i]
            denoised_one_item = denoised[i, ref_me_len:estimated_mel_len, :].unsqueeze(0).transpose(1, 2)
            audio = self.forward_vocoder(denoised_one_item)
            rms = torch.sqrt(torch.mean(torch.square(audio)))
            if rms < self.target_rms:
                audio = audio * self.target_rms / rms

            audio = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(audio))
            inference_response = pb_utils.InferenceResponse(output_tensors=[audio])
            responses.append(inference_response)
        if self.use_perf:
            torch.cuda.nvtx.range_pop()
        return responses
