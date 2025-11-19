import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from gguf import GGUFWriter

def convert_to_gguf(model_name_or_path, output_dir=".", filename="model.gguf"):
    """
    Converts a Hugging Face model to GGUF format.

    Args:
        model_name_or_path (str): The name or path of the Hugging Face model.
        output_dir (str): The directory to save the GGUF file.
        filename (str): The name of the GGUF file.
    """
    print(f"Loading model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    output_path = os.path.join(output_dir, filename)

    # Dynamically determine architecture
    architecture = "unknown"
    if model.config.architectures:
        # e.g., LlamaForCausalLM -> Llama
        architecture = model.config.architectures[0].replace("ForCausalLM", "").lower()

    gguf_writer = GGUFWriter(output_path, architecture)

    # Add model metadata
    gguf_writer.add_architecture()
    gguf_writer.add_name(model_name_or_path)

    # Conditional metadata extraction based on architecture
    if architecture == "llama":
        gguf_writer.add_context_length(getattr(model.config, "max_position_embeddings", 2048))
        gguf_writer.add_embedding_length(model.config.hidden_size)
        gguf_writer.add_block_count(model.config.num_hidden_layers)
        gguf_writer.add_feed_forward_length(model.config.intermediate_size)
        gguf_writer.add_rope_dimension(getattr(model.config, "rope_theta", 10000)) # Llama 2 uses rope_theta
        gguf_writer.add_head_count(model.config.num_attention_heads)
        gguf_writer.add_head_count_kv(getattr(model.config, "num_key_value_heads", model.config.num_attention_heads))
        gguf_writer.add_layer_norm_rms_eps(model.config.rms_norm_eps)
        gguf_writer.add_tokenizer_model("llama")
        gguf_writer.add_token_list([tokenizer.decode([i]) for i in range(tokenizer.vocab_size)])
    elif architecture == "mistral":
        gguf_writer.add_context_length(getattr(model.config, "max_position_embeddings", 2048))
        gguf_writer.add_embedding_length(model.config.hidden_size)
        gguf_writer.add_block_count(model.config.num_hidden_layers)
        gguf_writer.add_feed_forward_length(model.config.intermediate_size)
        gguf_writer.add_rope_dimension(getattr(model.config, "rope_theta", 10000))
        gguf_writer.add_head_count(model.config.num_attention_heads)
        gguf_writer.add_head_count_kv(getattr(model.config, "num_key_value_heads", model.config.num_attention_heads))
        gguf_writer.add_layer_norm_rms_eps(model.config.rms_norm_eps)
        gguf_writer.add_tokenizer_model("llama") # Mistral often uses Llama tokenizer
        gguf_writer.add_token_list([tokenizer.decode([i]) for i in range(tokenizer.vocab_size)])
    elif architecture == "qwen3":
        gguf_writer.add_context_length(getattr(model.config, "max_position_embeddings", 32768))
        gguf_writer.add_embedding_length(model.config.hidden_size)
        gguf_writer.add_block_count(model.config.num_hidden_layers)
        gguf_writer.add_feed_forward_length(model.config.intermediate_size)
        gguf_writer.add_rope_dimension_count(getattr(model.config, "rope_theta", 10000)) # Qwen uses rope_theta
        gguf_writer.add_head_count(model.config.num_attention_heads)
        gguf_writer.add_head_count_kv(getattr(model.config, "num_key_value_heads", model.config.num_attention_heads))
        gguf_writer.add_layer_norm_rms_eps(model.config.rms_norm_eps)
        gguf_writer.add_tokenizer_model("qwen") # Qwen has its own tokenizer
        gguf_writer.add_token_list([tokenizer.decode([i]) for i in range(tokenizer.vocab_size)])
    # Add more architectures as needed (e.g., gemma, phi, etc.)
    else:
        print(f"Warning: Architecture '{architecture}' not explicitly handled. Using generic metadata.")
        gguf_writer.add_context_length(getattr(model.config, "max_position_embeddings", 2048))
        gguf_writer.add_embedding_length(getattr(model.config, "hidden_size", 0))
        gguf_writer.add_block_count(getattr(model.config, "num_hidden_layers", 0))
        gguf_writer.add_feed_forward_length(getattr(model.config, "intermediate_size", 0))
        gguf_writer.add_head_count(getattr(model.config, "num_attention_heads", 0))
        gguf_writer.add_head_count_kv(getattr(model.config, "num_key_value_heads", getattr(model.config, "num_attention_heads", 0)))
        gguf_writer.add_layer_norm_rms_eps(getattr(model.config, "rms_norm_eps", 1e-5))
        gguf_writer.add_tokenizer_model(getattr(tokenizer, "name_or_path", "unknown"))
        gguf_writer.add_token_list([tokenizer.decode([i]) for i in range(tokenizer.vocab_size)])

    # Add model tensors (weights)
    for name, data in model.named_parameters():
        print(f"Adding tensor: {name} with shape {data.shape}")
        # TODO: Implement specific tensor name remapping for different architectures
        # For now, directly use the Hugging Face tensor names.
        gguf_writer.add_tensor(name, data.detach().float().numpy())

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    print(f"Model successfully converted and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hugging Face models to GGUF format.")
    parser.add_argument("model_name_or_path", type=str,
                        help="The name or path of the Hugging Face model (e.g., 'gpt2', 'facebook/opt-125m').")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="The directory to save the GGUF file.")
    parser.add_argument("--filename", type=str, default="model.gguf",
                        help="The name of the GGUF file.")
    args = parser.parse_args()

    convert_to_gguf(args.model_name_or_path, args.output_dir, args.filename)
