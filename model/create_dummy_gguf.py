import argparse
import numpy as np
from gguf import GGUFWriter

def create_dummy_gguf(output_dir=".", filename="dummy_model.gguf"):
    """
    Creates a very small, dummy GGUF model file for unit testing purposes.
    """
    output_path = f"{output_dir}/{filename}"

    # Define dummy model parameters
    architecture = "dummy"
    vocab_size = 16  # Small vocabulary
    embedding_length = 32
    num_layers = 1
    num_attention_heads = 1
    num_key_value_heads = 1
    intermediate_size = 64 # Feed forward length
    rms_norm_eps = 1e-5
    rope_dimension = 4 # A small arbitrary value

    # Create a GGUF writer
    gguf_writer = GGUFWriter(output_path, architecture)

    # Add model metadata
    gguf_writer.add_architecture()
    gguf_writer.add_name("dummy-model")
    gguf_writer.add_context_length(256)
    gguf_writer.add_embedding_length(embedding_length)
    gguf_writer.add_block_count(num_layers)
    gguf_writer.add_feed_forward_length(intermediate_size)
    gguf_writer.add_rope_dimension_count(rope_dimension)
    gguf_writer.add_head_count(num_attention_heads)
    gguf_writer.add_head_count_kv(num_key_value_heads)
    gguf_writer.add_layer_norm_rms_eps(rms_norm_eps)
    
    # Add dummy tokenizer info
    gguf_writer.add_tokenizer_model("dummy")
    
    # Create a dummy token list
    dummy_tokens = [f"token_{i}" for i in range(vocab_size - 4)] # regular tokens
    dummy_tokens.extend(["<unk>", "<s>", "</s>", "<pad>"]) # special tokens
    gguf_writer.add_token_list(dummy_tokens)

    # Add dummy tensors
    # All tensors will be float32 (np.float32) and filled with small random values.
    # The shapes are simplified to match the basic expectations of an LLM.

    # 1. Token Embeddings
    token_embeddings_weight = np.random.rand(vocab_size, embedding_length).astype(np.float32)
    gguf_writer.add_tensor("model.embed_tokens.weight", token_embeddings_weight)

    # 2. Input LayerNorm for the first block
    input_layernorm_weight_0 = np.random.rand(embedding_length).astype(np.float32)
    gguf_writer.add_tensor(f"model.layers.0.input_layernorm.weight", input_layernorm_weight_0)

    # 3. Attention projections for the first block
    q_proj_weight_0 = np.random.rand(embedding_length, embedding_length).astype(np.float32)
    gguf_writer.add_tensor(f"model.layers.0.self_attn.q_proj.weight", q_proj_weight_0)

    k_proj_weight_0 = np.random.rand(embedding_length, embedding_length).astype(np.float32)
    gguf_writer.add_tensor(f"model.layers.0.self_attn.k_proj.weight", k_proj_weight_0)

    v_proj_weight_0 = np.random.rand(embedding_length, embedding_length).astype(np.float32)
    gguf_writer.add_tensor(f"model.layers.0.self_attn.v_proj.weight", v_proj_weight_0)

    o_proj_weight_0 = np.random.rand(embedding_length, embedding_length).astype(np.float32)
    gguf_writer.add_tensor(f"model.layers.0.self_attn.o_proj.weight", o_proj_weight_0)

    # 4. MLP projections for the first block (FFN)
    gate_proj_weight_0 = np.random.rand(intermediate_size, embedding_length).astype(np.float32)
    gguf_writer.add_tensor(f"model.layers.0.mlp.gate_proj.weight", gate_proj_weight_0)

    up_proj_weight_0 = np.random.rand(intermediate_size, embedding_length).astype(np.float32)
    gguf_writer.add_tensor(f"model.layers.0.mlp.up_proj.weight", up_proj_weight_0)

    down_proj_weight_0 = np.random.rand(embedding_length, intermediate_size).astype(np.float32)
    gguf_writer.add_tensor(f"model.layers.0.mlp.down_proj.weight", down_proj_weight_0)
    
    # 5. Final LayerNorm (model.norm)
    model_norm_weight = np.random.rand(embedding_length).astype(np.float32)
    gguf_writer.add_tensor("model.norm.weight", model_norm_weight)

    # 6. Output projection (language model head)
    output_weight = np.random.rand(vocab_size, embedding_length).astype(np.float32)
    gguf_writer.add_tensor("model.output.weight", output_weight)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    print(f"Dummy GGUF model successfully created and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dummy GGUF model for testing.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="The directory to save the dummy GGUF file.")
    parser.add_argument("--filename", type=str, default="test_model.gguf",
                        help="The name of the dummy GGUF file.")
    args = parser.parse_args()

    create_dummy_gguf(args.output_dir, args.filename)