import argparse
import os
import struct
import json
from pathlib import Path

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
from gguf import GGUFWriter, GGMLQuantizationType # type: ignore

# This script is inspired by the convert.py script from llama.cpp
# and aims to provide a more robust conversion for various Hugging Face models
# to the GGUF format, including proper tensor remapping and metadata.

def get_gguf_writer_for_model(model, tokenizer, output_path, arch):
    gguf_writer = GGUFWriter(output_path, arch)

    # Add model metadata
    gguf_writer.add_architecture()
    gguf_writer.add_name(model.config.model_type)

    # General metadata
    gguf_writer.add_uint32("general.file_type", 1)  # 1 for f16

    # Tokenizer metadata
    gguf_writer.add_string("tokenizer.ggml.model", "qwen")
    tokens = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
    gguf_writer.add_token_list(tokens)
    gguf_writer.add_int32("tokenizer.ggml.bos_token_id", tokenizer.bos_token_id if tokenizer.bos_token_id is not None else -1)
    gguf_writer.add_int32("tokenizer.ggml.eos_token_id", tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1)
    gguf_writer.add_int32("tokenizer.ggml.unk_token_id", tokenizer.unk_token_id if tokenizer.unk_token_id is not None else -1)
    gguf_writer.add_int32("tokenizer.ggml.pad_token_id", tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1)
    
    # Qwen3 specific metadata (using keys from llama.cpp's GGUF)
    # Safely get config attributes with defaults
    context_length = getattr(model.config, "max_position_embeddings", 2048)
    embedding_length = getattr(model.config, "hidden_size", 0)
    block_count = getattr(model.config, "num_hidden_layers", 0)
    feed_forward_length = getattr(model.config, "intermediate_size", 0)
    num_attention_heads = getattr(model.config, "num_attention_heads", 0)
    num_key_value_heads = getattr(model.config, "num_key_value_heads", num_attention_heads)
    rms_norm_eps = getattr(model.config, "rms_norm_eps", 1e-6)
    vocab_size = getattr(model.config, "vocab_size", 0)
    rope_theta = getattr(model.config, "rope_theta", 10000.0)

    # Validate and add metadata
    if embedding_length > 0 and num_attention_heads > 0:
        head_dim = embedding_length // num_attention_heads
    else:
        head_dim = 0 # Or raise an error if head_dim is critical and cannot be 0

    print(f"Adding metadata: llama.context_length, value={context_length}, type=uint32")
    gguf_writer.add_uint32("llama.context_length", context_length)
    print(f"Adding metadata: llama.embedding_length, value={embedding_length}, type=uint32")
    gguf_writer.add_uint32("llama.embedding_length", embedding_length)
    print(f"Adding metadata: llama.block_count, value={block_count}, type=uint32")
    gguf_writer.add_uint32("llama.block_count", block_count)
    print(f"Adding metadata: llama.feed_forward_length, value={feed_forward_length}, type=uint32")
    gguf_writer.add_uint32("llama.feed_forward_length", feed_forward_length)
    print(f"Adding metadata: llama.rope.dimension_count, value={head_dim}, type=uint32")
    gguf_writer.add_uint32("llama.rope.dimension_count", head_dim)
    print(f"Adding metadata: llama.rope.freq_base, value={float(rope_theta)}, type=float32")
    gguf_writer.add_float32("llama.rope.freq_base", float(rope_theta))
    print(f"Adding metadata: llama.attention.head_count, value={num_attention_heads}, type=uint32")
    gguf_writer.add_uint32("llama.attention.head_count", num_attention_heads)
    print(f"Adding metadata: llama.attention.head_count_kv, value={num_key_value_heads}, type=uint32")
    gguf_writer.add_uint32("llama.attention.head_count_kv", num_key_value_heads)
    print(f"Adding metadata: llama.attention.layer_norm_rms_epsilon, value={float(rms_norm_eps)}, type=float32")
    gguf_writer.add_float32("llama.attention.layer_norm_rms_epsilon", float(rms_norm_eps))
    print(f"Adding metadata: tokenizer.ggml.model.n_vocab, value={vocab_size}, type=uint32")
    gguf_writer.add_uint32("tokenizer.ggml.model.n_vocab", vocab_size)

    return gguf_writer


def convert_to_gguf(model_name_or_path, output_dir=".", filename="model.gguf"):
    print(f"Loading model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    print("Model and tokenizer loaded successfully.")

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output path set to: {output_path}")

    arch = model.config.model_type.lower()
    if arch == "qwen2": # Transformers uses Qwen2 for Qwen3 sometimes
        arch = "qwen3"
    print(f"Detected architecture: {arch}")

    gguf_writer = get_gguf_writer_for_model(model, tokenizer, output_path, arch)
    print("GGUFWriter initialized and metadata added.")

    tensors_to_write = []
    tensor_count = 0
    for name, data in model.named_parameters():
        original_shape = data.shape
        np_data = data.detach().float().numpy()

        new_name = name

        if "embed_tokens" in name:
            new_name = "token_embd.weight"
        elif ".q_proj.weight" in name:
            new_name = name
            np_data = np_data.T # Re-add transpose
        elif ".q_proj.bias" in name:
            new_name = name
        elif ".k_proj.weight" in name:
            new_name = name
            np_data = np_data.T # Re-add transpose
        elif ".k_proj.bias" in name:
            new_name = name
        elif ".v_proj.weight" in name:
            new_name = name
            np_data = np_data.T # Re-add transpose
        elif ".v_proj.bias" in name:
            new_name = name
        elif ".o_proj.weight" in name:
            new_name = name
            np_data = np_data.T # Re-add transpose
        elif ".o_proj.bias" in name:
            new_name = name
        elif ".gate_proj.weight" in name:
            new_name = name
            np_data = np_data.T # Re-add transpose
        elif ".gate_proj.bias" in name:
            new_name = name
        elif ".up_proj.weight" in name:
            new_name = name
            np_data = np_data.T # Re-add transpose
        elif ".up_proj.bias" in name:
            new_name = name
        elif ".down_proj.weight" in name:
            new_name = name
            np_data = np_data.T # Re-add transpose
        elif ".down_proj.bias" in name:
            new_name = name
        elif "input_layernorm.weight" in name:
            new_name = name
        elif "input_layernorm.bias" in name:
            new_name = name
        elif "post_attention_layernorm.weight" in name:
            new_name = name
        elif "post_attention_layernorm.bias" in name:
            new_name = name
        elif "norm.weight" in name and "model.norm" in name: # Final norm
            new_name = "output_norm.weight"
        elif "norm.bias" in name and "model.norm" in name:
            new_name = "output_norm.bias"
        elif "lm_head.weight" in name:
            new_name = "output.weight"
            np_data = np_data.T
        elif "lm_head.bias" in name:
            new_name = "output.bias"
        elif "rotary_emb.inv_freq" in name:
            # Rotary embeddings are usually not saved as tensors in GGUF
            print(f"Skipping tensor: {name} (rotary_emb.inv_freq)")
            continue
        
        print(f"Adding tensor: {name} (original shape {original_shape}) -> {new_name} (new shape {np_data.shape})")
        gguf_writer.add_tensor(new_name, np_data)
        tensor_count += 1
    
    print(f"Finished processing {tensor_count} tensors.")

    print("Writing header to file...")
    gguf_writer.write_header_to_file()
    print("Writing KV data to file...")
    gguf_writer.write_kv_data_to_file()
    print("Writing tensors to file...")
    gguf_writer.write_tensors_to_file()
    print("All data written.")

    gguf_writer.close()
    print(f"Model successfully converted and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hugging Face models to GGUF format.")
    parser.add_argument("model_name_or_path", type=str,
                        help="The name or path of the Hugging Face model (e.g., 'Qwen/Qwen1.5-0.5B-Chat').")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="The directory to save the GGUF file.")
    parser.add_argument("--filename", type=str, default="model.gguf",
                        help="The name of the GGUF file.")
    args = parser.parse_args()

    convert_to_gguf(args.model_name_or_path, args.output_dir, args.filename)