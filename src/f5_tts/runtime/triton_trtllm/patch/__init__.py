# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .baichuan.model import BaichuanForCausalLM
from .bert.model import (
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertModel,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaModel,
)
from .bloom.model import BloomForCausalLM, BloomModel
from .chatglm.config import ChatGLMConfig
from .chatglm.model import ChatGLMForCausalLM, ChatGLMModel
from .cogvlm.config import CogVLMConfig
from .cogvlm.model import CogVLMForCausalLM
from .commandr.model import CohereForCausalLM
from .dbrx.config import DbrxConfig
from .dbrx.model import DbrxForCausalLM
from .deepseek_v1.model import DeepseekForCausalLM
from .deepseek_v2.model import DeepseekV2ForCausalLM
from .dit.model import DiT
from .eagle.model import EagleForCausalLM
from .enc_dec.model import DecoderModel, EncoderModel, WhisperEncoder
from .falcon.config import FalconConfig
from .falcon.model import FalconForCausalLM, FalconModel
from .gemma.config import GEMMA2_ARCHITECTURE, GEMMA_ARCHITECTURE, GemmaConfig
from .gemma.model import GemmaForCausalLM
from .gpt.config import GPTConfig
from .gpt.model import GPTForCausalLM, GPTModel
from .gptj.config import GPTJConfig
from .gptj.model import GPTJForCausalLM, GPTJModel
from .gptneox.model import GPTNeoXForCausalLM, GPTNeoXModel
from .grok.model import GrokForCausalLM
from .llama.config import LLaMAConfig
from .llama.model import LLaMAForCausalLM, LLaMAModel
from .mamba.model import MambaForCausalLM
from .medusa.config import MedusaConfig
from .medusa.model import MedusaForCausalLm
from .mllama.model import MLLaMAModel
from .modeling_utils import PretrainedConfig, PretrainedModel, SpeculativeDecodingMode
from .mpt.model import MPTForCausalLM, MPTModel
from .nemotron_nas.model import DeciLMForCausalLM
from .opt.model import OPTForCausalLM, OPTModel
from .phi3.model import Phi3ForCausalLM, Phi3Model
from .phi.model import PhiForCausalLM, PhiModel
from .qwen.model import QWenForCausalLM
from .recurrentgemma.model import RecurrentGemmaForCausalLM
from .redrafter.model import ReDrafterForCausalLM
from .f5tts.model import F5TTS

__all__ = [
    "BertModel",
    "BertForQuestionAnswering",
    "BertForSequenceClassification",
    "RobertaModel",
    "RobertaForQuestionAnswering",
    "RobertaForSequenceClassification",
    "BloomModel",
    "BloomForCausalLM",
    "DiT",
    "DeepseekForCausalLM",
    "FalconConfig",
    "DeepseekV2ForCausalLM",
    "FalconForCausalLM",
    "FalconModel",
    "GPTConfig",
    "GPTModel",
    "GPTForCausalLM",
    "OPTForCausalLM",
    "OPTModel",
    "LLaMAConfig",
    "LLaMAForCausalLM",
    "LLaMAModel",
    "MedusaConfig",
    "MedusaForCausalLm",
    "ReDrafterForCausalLM",
    "GPTJConfig",
    "GPTJModel",
    "GPTJForCausalLM",
    "GPTNeoXModel",
    "GPTNeoXForCausalLM",
    "PhiModel",
    "PhiConfig",
    "Phi3Model",
    "Phi3Config",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "ChatGLMConfig",
    "ChatGLMForCausalLM",
    "ChatGLMModel",
    "BaichuanForCausalLM",
    "QWenConfigQWenForCausalLM",
    "QWenModel",
    "EncoderModel",
    "DecoderModel",
    "PretrainedConfig",
    "PretrainedModel",
    "WhisperEncoder",
    "MambaForCausalLM",
    "MambaConfig",
    "MPTForCausalLM",
    "MPTModel",
    "SkyworkForCausalLM",
    "GemmaConfig",
    "GemmaForCausalLM",
    "DbrxConfig",
    "DbrxForCausalLM",
    "RecurrentGemmaForCausalLM",
    "CogVLMConfig",
    "CogVLMForCausalLM",
    "EagleForCausalLM",
    "SpeculativeDecodingMode",
    "CohereForCausalLM",
    "MLLaMAModel",
    "F5TTS",
]

MODEL_MAP = {
    "GPT2LMHeadModel": GPTForCausalLM,
    "GPT2LMHeadCustomModel": GPTForCausalLM,
    "GPTBigCodeForCausalLM": GPTForCausalLM,
    "Starcoder2ForCausalLM": GPTForCausalLM,
    "FuyuForCausalLM": GPTForCausalLM,
    "Kosmos2ForConditionalGeneration": GPTForCausalLM,
    "JAISLMHeadModel": GPTForCausalLM,
    "GPTForCausalLM": GPTForCausalLM,
    "NemotronForCausalLM": GPTForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
    "BloomForCausalLM": BloomForCausalLM,
    "RWForCausalLM": FalconForCausalLM,
    "FalconForCausalLM": FalconForCausalLM,
    "PhiForCausalLM": PhiForCausalLM,
    "Phi3ForCausalLM": Phi3ForCausalLM,
    "Phi3VForCausalLM": Phi3ForCausalLM,
    "Phi3SmallForCausalLM": Phi3ForCausalLM,
    "PhiMoEForCausalLM": Phi3ForCausalLM,
    "MambaForCausalLM": MambaForCausalLM,
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "GPTJForCausalLM": GPTJForCausalLM,
    "MPTForCausalLM": MPTForCausalLM,
    "GLMModel": ChatGLMForCausalLM,
    "ChatGLMModel": ChatGLMForCausalLM,
    "ChatGLMForCausalLM": ChatGLMForCausalLM,
    "LlamaForCausalLM": LLaMAForCausalLM,
    "ExaoneForCausalLM": LLaMAForCausalLM,
    "MistralForCausalLM": LLaMAForCausalLM,
    "MixtralForCausalLM": LLaMAForCausalLM,
    "ArcticForCausalLM": LLaMAForCausalLM,
    "Grok1ModelForCausalLM": GrokForCausalLM,
    "InternLMForCausalLM": LLaMAForCausalLM,
    "InternLM2ForCausalLM": LLaMAForCausalLM,
    "MedusaForCausalLM": MedusaForCausalLm,
    "ReDrafterForCausalLM": ReDrafterForCausalLM,
    "BaichuanForCausalLM": BaichuanForCausalLM,
    "BaiChuanForCausalLM": BaichuanForCausalLM,
    "SkyworkForCausalLM": LLaMAForCausalLM,
    GEMMA_ARCHITECTURE: GemmaForCausalLM,
    GEMMA2_ARCHITECTURE: GemmaForCausalLM,
    "QWenLMHeadModel": QWenForCausalLM,
    "QWenForCausalLM": QWenForCausalLM,
    "Qwen2ForCausalLM": QWenForCausalLM,
    "Qwen2MoeForCausalLM": QWenForCausalLM,
    "Qwen2ForSequenceClassification": QWenForCausalLM,
    "Qwen2VLForConditionalGeneration": QWenForCausalLM,
    "WhisperEncoder": WhisperEncoder,
    "EncoderModel": EncoderModel,
    "DecoderModel": DecoderModel,
    "DbrxForCausalLM": DbrxForCausalLM,
    "RecurrentGemmaForCausalLM": RecurrentGemmaForCausalLM,
    "CogVLMForCausalLM": CogVLMForCausalLM,
    "DiT": DiT,
    "DeepseekForCausalLM": DeepseekForCausalLM,
    "DeciLMForCausalLM": DeciLMForCausalLM,
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "EagleForCausalLM": EagleForCausalLM,
    "CohereForCausalLM": CohereForCausalLM,
    "MllamaForConditionalGeneration": MLLaMAModel,
    "BertForQuestionAnswering": BertForQuestionAnswering,
    "BertForSequenceClassification": BertForSequenceClassification,
    "BertModel": BertModel,
    "RobertaModel": RobertaModel,
    "RobertaForQuestionAnswering": RobertaForQuestionAnswering,
    "RobertaForSequenceClassification": RobertaForSequenceClassification,
    "F5TTS": F5TTS,
}
