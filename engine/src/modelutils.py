import math
import os
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import transformers
from accelerate import dispatch_model
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from transformers import AutoConfig, AutoModelForCausalLM

from src.aq import QuantizedWeight

MODEL_ERROR_MSG = "Unsupported model type {} - only llama-like, opt, falcon, phi3, internvl_chat are supported"
FALCON_TYPES = ("falcon", "refinedweb", "refinedwebmodel")
INTERNVL_TYPES = ("internvl_chat",)
LLAMA_LIKE = ("llama", "Yi", "mistral", "mixtral", "gemma", "cohere", "qwen2", "qwen3")


def is_internvl(model: nn.Module) -> bool:
    return model.config.model_type in INTERNVL_TYPES


def get_llm_model(model: nn.Module) -> nn.Module:
    if is_internvl(model):
        return model.language_model
    return model


def get_llm_config(model: nn.Module):
    if is_internvl(model):
        return model.config.llm_config
    return model.config


def get_hidden_size(model: nn.Module) -> int:
    return get_llm_config(model).hidden_size


def get_use_cache(model: nn.Module) -> bool:
    return getattr(get_llm_config(model), "use_cache", False)


def set_use_cache(model: nn.Module, use_cache: bool) -> None:
    get_llm_config(model).use_cache = use_cache


def get_forward_model(model: nn.Module) -> nn.Module:
    """Text-only forward model used during AQLM calibration."""
    return get_llm_model(model)


def get_quantizer_key_prefix(model: nn.Module) -> str:
    if is_internvl(model):
        return "language_model.model.layers"
    return "model.layers"


@contextmanager
def suspend_nn_inits():
    def skip(*args, **kwargs):
        pass

    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring


def dispatch_quantized_model(model):
    num_devices = torch.cuda.device_count()
    llm = get_llm_model(model)
    if is_internvl(model):
        device_map = {
            "language_model.model.embed_tokens": 0,
            "language_model.model.norm": num_devices - 1,
            "language_model.lm_head": 0,
        }
        layers_prefix = "language_model.model.layers"
    else:
        device_map = {"model.embed_tokens": 0, "model.norm": num_devices - 1, "lm_head": 0}
        layers_prefix = "model.layers"
    num_layers = len(get_layers(model))
    layers_per_device = math.ceil(num_layers / num_devices)
    for layer_id in range(num_layers):
        device_id = layer_id // layers_per_device
        device_map[f"{layers_prefix}.{layer_id}"] = device_id
    model = dispatch_model(model, device_map)
    if is_internvl(model):
        model.language_model.model.embed_tokens = model.language_model.model.embed_tokens.to("cuda:0")
        model.language_model.lm_head = model.language_model.lm_head.to("cuda:0")
    else:
        model.model.embed_tokens = model.model.embed_tokens.to("cuda:0")
        model.lm_head = model.lm_head.to("cuda:0")
    return model


def get_model(
    model_path, load_quantized=None, dtype="auto", device_map=None, attn_implementation=None, trust_remote_code=False
):
    if dtype == "auto":
        dtype = (
            AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code).torch_dtype or "auto"
        )  # force transformers 4.29.2 to follow the same rules as 4.30.x
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    model_kwargs = {}
    # this argument is avaialbe only for transformers >= 4.38.0
    if transformers.__version__ >= "4.38.0":
        model_kwargs["attn_implementation"] = attn_implementation

    load_kwargs = dict(
        trust_remote_code=trust_remote_code,
        device_map=None if load_quantized else device_map,
        low_cpu_mem_usage=True,
        local_files_only=True,
        **model_kwargs,
    )
    if transformers.__version__ >= "4.56.0":
        load_kwargs["dtype"] = dtype
    else:
        load_kwargs["torch_dtype"] = dtype

    with suspend_nn_inits():
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            **load_kwargs,
        )
        if load_quantized:
            print("Initializing model with random weights...")
            print("Loading quantized model ...")
            model = load_quantized_model(model, load_quantized)
            if device_map == "auto":
                llm_type = get_llm_config(model).model_type
                assert llm_type in LLAMA_LIKE or is_internvl(model), (
                    "Dispatching is implemented only for Llama-like models and InternVL."
                )
                model = dispatch_quantized_model(model)
        else:
            print("Loading pretrained model ...")

    print("Model loaded sucсessfully ...")

    return model


def is_model_for_causal_lm(model: nn.Module):
    assert isinstance(model, transformers.PreTrainedModel)
    assert len(model.base_model_prefix) > 0 and hasattr(model, model.base_model_prefix)
    assert model.get_output_embeddings() is not None
    return True


def get_model_head_with_norm(model):
    head = torch.nn.ModuleList()
    llm = get_llm_model(model)
    llm_config = get_llm_config(model)
    if llm_config.model_type in (*LLAMA_LIKE, "phi3"):
        if llm.model.norm is not None:
            head.append(llm.model.norm)
        head.append(llm.lm_head)
    elif llm_config.model_type.lower() in FALCON_TYPES:
        if llm.transformer.ln_f is not None:
            head.append(llm.transformer.ln_f)
        head.append(llm.lm_head)
    elif llm_config.model_type == "opt":
        if llm.model.decoder.final_layer_norm is not None:
            head.append(llm.model.decoder.final_layer_norm)
        if llm.model.decoder.project_out is not None:
            head.append(llm.model.decoder.project_out)
        head.append(llm.lm_head)
    else:
        raise ValueError(MODEL_ERROR_MSG.format(llm_config.model_type))
    return head


def get_lm_logits(inps_, model):
    llm = get_llm_model(model)
    llm_config = get_llm_config(model)
    if llm_config.model_type in (*LLAMA_LIKE, "phi3"):
        hidden_states = inps_.unsqueeze(0)
        if llm.model.norm is not None:
            hidden_states = llm.model.norm(hidden_states)
        lm_logits = llm.lm_head(hidden_states)
    elif llm_config.model_type.lower() in FALCON_TYPES:
        hidden_states = inps_.unsqueeze(0)
        if llm.transformer.ln_f is not None:
            hidden_states = llm.transformer.ln_f(hidden_states)
        lm_logits = llm.lm_head(hidden_states)
    elif llm_config.model_type == "opt":
        hidden_states = inps_.unsqueeze(0)
        if llm.model.decoder.final_layer_norm is not None:
            hidden_states = llm.model.decoder.final_layer_norm(hidden_states)
        if llm.model.decoder.project_out is not None:
            hidden_states = llm.model.decoder.project_out(hidden_states)
        lm_logits = llm.lm_head(hidden_states)
    else:
        raise ValueError(MODEL_ERROR_MSG.format(llm_config.model_type))
    return lm_logits


def get_layers(model):
    llm = get_llm_model(model)
    llm_config = get_llm_config(model)
    if llm_config.model_type in (*LLAMA_LIKE, "phi3"):
        return llm.model.layers
    elif llm_config.model_type.lower() in FALCON_TYPES:
        return llm.transformer.h
    elif llm_config.model_type == "opt":
        return llm.model.decoder.layers
    else:
        raise ValueError(MODEL_ERROR_MSG.format(llm_config.model_type))


def find_sublayers(module, layers=(nn.Conv2d, nn.Linear)):
    res = {}
    for name, layer in module.named_modules():
        if isinstance(layer, layers):
            res[name] = layer
    return res


def get_sequential_groups(model):
    llm_config = get_llm_config(model)
    if llm_config.model_type in LLAMA_LIKE:
        assert "mixtral" not in llm_config.model_type.lower()  # check that this is not mixtral
        return [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]
    elif llm_config.model_type.lower() in FALCON_TYPES:
        return [
            ["self_attention.query_key_value"],
            ["self_attention.dense"],
            ["mlp.dense_h_to_4h"],
            ["mlp.dense_4h_to_h"],
        ]
    elif llm_config.model_type == "opt":
        return [
            ["self_attn.q_proj"],
            ["self_attn.k_proj"],
            ["self_attn.v_proj"],
            ["self_attn.out_proj"],
            ["fc1"],
            ["fc2"],
        ]
    elif llm_config.model_type == "phi3":
        return [["self_attn.qkv_proj"], ["self_attn.o_proj"], ["mlp.gate_up_proj"], ["mlp.down_proj"]]
    else:
        raise ValueError(MODEL_ERROR_MSG.format(llm_config.model_type))


def read_quant_weight_from_file(load_path, block_i, layer_name, device):
    return torch.load(load_path + "/" + str(block_i) + "/" + layer_name, map_location=device)


def load_linear_layers(layer, quant_layer, model):
    layer_ident = {}
    llm_config = get_llm_config(model)
    for submodule in layer.modules():
        for child_name, child_module in submodule.named_children():
            print(child_name, "child_name", layer_ident)
            if isinstance(child_module, (nn.Conv2d, nn.Linear)) or "norm" in child_name:
                if child_name in layer_ident:
                    layer_ident[child_name] += 1
                else:
                    layer_ident[child_name] = 1
                quant_count = 0
                print("Finding to dequantize ", child_name)
                for quant_submodule in quant_layer.modules():
                    for quant_child_name, quant_child_module in quant_submodule.named_children():
                        if quant_child_name == child_name:
                            quant_count += 1
                            if quant_count != layer_ident[child_name]:
                                continue
                            print(quant_child_name, quant_child_module)
                            if ("gate" in child_name.lower()) and ("mixtral" in llm_config.model_type.lower()):
                                print("gate", child_name)
                                child_module.weight.data = quant_child_module.weight.data.to(
                                    child_module.weight.dtype
                                ).to(child_module.weight.device)
                                continue
                            if "norm" in child_name and not isinstance(child_module, (nn.Conv2d, nn.Linear)):
                                print("norm", child_name)
                                child_module.weight.data = quant_child_module.weight.data.to(
                                    child_module.weight.dtype
                                ).to(child_module.weight.device)
                            else:
                                print(child_name)
                                child_module.weight.data = (
                                    quant_child_module.quantized_weight()
                                    .data.to(child_module.weight.dtype)
                                    .to(child_module.weight.device)
                                )
                            # Bias is not taked into account
    return layer


def load_dequantized_model(model, load_path):
    """Load quantized model by dequantizing it"""
    layers = get_layers(model)
    for layer_index in range(len(layers)):
        print("layer", layer_index)
        layer = layers[layer_index]
        quant_layer = torch.load(os.path.join(load_path, str(layer_index) + ".pth"), map_location="cpu")
        for module in quant_layer.modules():
            if isinstance(module, QuantizedWeight):
                if not hasattr(module, "codes_storage"):
                    module.codes_storage = None  # backwards compatibility
        layers[layer_index] = load_linear_layers(layer, quant_layer, model)
    model.load_state_dict(torch.load(os.path.join(load_path, "not_quantized_weights.pt")), strict=False)
    return model


def load_quantized_model(model, load_path):
    """Load quantized model"""
    layers = get_layers(model)
    for layer_index in range(len(layers)):
        layers[layer_index] = torch.load(
            os.path.join(load_path, str(layer_index) + ".pth"),
            map_location=layers[layer_index].input_layernorm.weight.device,
        )
        for module in layers[layer_index].modules():
            if isinstance(module, QuantizedWeight):
                if not hasattr(module, "codes_storage"):
                    module.codes_storage = None  # backwards compatibility

    model.load_state_dict(torch.load(os.path.join(load_path, "not_quantized_weights.pt")), strict=False)
    return model


def save_not_quantized_weights(model: nn.Module, save_dir: str):
    already_saved_weights = set()
    for layer in get_layers(model):
        for param in layer.parameters():
            already_saved_weights.add(param)
    not_quantized_weights = {
        name: param for name, param in model.named_parameters() if param not in already_saved_weights
    }
    torch.save(not_quantized_weights, os.path.join(save_dir, "not_quantized_weights.pt"))


def save_quantized_model(model: transformers.PreTrainedModel, save_dir: str):
    """Save dequantized model state in the same format as returned by AQLM calibration (main.py)"""
    os.makedirs(save_dir, exist_ok=True)
    for layer_index, layer in enumerate(get_layers(model)):
        layer_save_path = os.path.join(save_dir, f"{layer_index}.pth")
        torch.save(layer, layer_save_path)
    save_not_quantized_weights(model, save_dir)


def get_layers_prefix(config: transformers.PretrainedConfig) -> str:
    if config.model_type in INTERNVL_TYPES:
        return "language_model.model.layers"
    if config.model_type in ("llama", "mistral", "mixtral", "gemma", "phi3", "qwen2", "qwen3"):
        return "model.layers"
    raise NotImplementedError(f"Can't get layers prefix for {config.model_type}")
