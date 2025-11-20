#include "engine.h"
#include <iostream>
#include <fstream> // For file seeking
#include <stdexcept> // For std::runtime_error
#include <string> // For std::stoul, std::stof
#include <sstream> // For std::stringstream
#include <cfloat> // For FLT_MAX
#include <vector> // For std::vector
#include "kernel/matrix_multiplication.h" // Add this line

#ifdef __cplusplus
extern "C" {
#endif

void happy_phone_llm_engine_init() {
    std::cout << "Happy Phone LLM Engine Initialized!" << std::endl;
}

#ifdef __cplusplus
}
#endif

namespace engine {

using namespace happy_phone_llm::kernel; // Add this line

// Helper to convert GGUF type to Tensor data type
static tensor::DataType gguf_type_to_tensor_dtype(gguf::gguf_type type) {
    switch (type) {
        case gguf::GGUF_TYPE_FLOAT32: return tensor::F32;
        case gguf::GGUF_TYPE_INT32: return tensor::I32;
        case gguf::GGUF_TYPE_UINT8: // Temporary workaround
            std::cerr << "Warning: GGUF_TYPE_UINT8 encountered, treating as F32. This might indicate a conversion issue." << std::endl;
            return tensor::F32;
        // TODO: Add more type conversions as needed
        default: throw std::runtime_error("Unsupported GGUF data type for tensor conversion.");
    }
}

Model::Model()
    : token_embeddings({}, tensor::F32),
      rms_norm_weight({}, tensor::F32),
      wq({}, tensor::F32), wk({}, tensor::F32), wv({}, tensor::F32), wo({}, tensor::F32),
      w1({}, tensor::F32), w2({}, tensor::F32), w3({}, tensor::F32),
      output_weight({}, tensor::F32),
      n_vocab(0), n_embd(0), n_head(0), n_kv_head(0),
      n_layer(0), n_rot(0), head_dim_q(0), head_dim_kv(0), // Initialize new members
      norm_eps(1e-5f) { // Also explicitly initialize norm_eps
    // Default constructor for Model
    std::cout << "Model default constructor called." << std::endl;
}

// Move Constructor
Model::Model(Model&& other) noexcept
    : token_embeddings(std::move(other.token_embeddings)),
      rms_norm_weight(std::move(other.rms_norm_weight)),
      wq(std::move(other.wq)), wk(std::move(other.wk)), wv(std::move(other.wv)), wo(std::move(other.wo)),
      w1(std::move(other.w1)), w2(std::move(other.w2)), w3(std::move(other.w3)),
      output_weight(std::move(other.output_weight)),
      n_vocab(other.n_vocab), n_embd(other.n_embd), n_head(other.n_head), n_kv_head(other.n_kv_head),
      n_layer(other.n_layer), n_rot(other.n_rot),
      head_dim_q(other.head_dim_q), head_dim_kv(other.head_dim_kv), // Initialize new members
      norm_eps(other.norm_eps),
      m_gguf_reader(std::move(other.m_gguf_reader)),
      m_tensors(std::move(other.m_tensors)) {
    // Other's members are already nullified/emptied by their respective move constructors
}

// Move Assignment Operator
Model& Model::operator=(Model&& other) noexcept {
    if (this != &other) {
        token_embeddings = std::move(other.token_embeddings);
        rms_norm_weight = std::move(other.rms_norm_weight);
        wq = std::move(other.wq);
        wk = std::move(other.wk);
        wv = std::move(other.wv);
        wo = std::move(other.wo);
        w1 = std::move(other.w1);
        w2 = std::move(other.w2);
        w3 = std::move(other.w3);
        output_weight = std::move(other.output_weight);
        n_vocab = other.n_vocab;
        n_embd = other.n_embd;
        n_head = other.n_head;
        n_kv_head = other.n_kv_head;
        n_layer = other.n_layer;
        n_rot = other.n_rot;
        head_dim_q = other.head_dim_q; // Assign new members
        head_dim_kv = other.head_dim_kv; // Assign new members
        norm_eps = other.norm_eps;
        m_gguf_reader = std::move(other.m_gguf_reader);
        m_tensors = std::move(other.m_tensors);

        // Other's members are already nullified/emptied by their respective move assignment operators
    }
    return *this;
}

bool Model::load(const std::string& filepath) {
    std::cout << "Model: Attempting to load GGUF file from " << filepath << std::endl;
    if (!m_gguf_reader.load_from_file(filepath)) {
        std::cerr << "Error: Failed to load GGUF file: " << filepath << std::endl;
        return false;
    }
    std::cout << "GGUF file loaded successfully. Version: " << m_gguf_reader.get_version()
              << ", Tensors: " << m_gguf_reader.get_tensor_infos().size()
              << ", Metadata: " << m_gguf_reader.get_metadata().size() << std::endl;

    const uint8_t* file_data_ptr = m_gguf_reader.get_file_data();
    uint64_t tensor_data_start_offset = m_gguf_reader.get_tensor_data_offset();
    
    // Process tensor infos from m_gguf_reader and create tensor::Tensor objects
    for (const auto& gguf_tensor_info : m_gguf_reader.get_tensor_infos()) {
        try {
            std::vector<uint64_t> shape;
            // GGUF stores dimensions in reverse order [d3, d2, d1, d0] for row-major.
            // Tensor expects [d0, d1, d2, d3] for row-major for C-style arrays (last dim contiguous).
            // We need to reverse them here for compatibility with common ML frameworks.
            for (uint32_t i = 0; i < gguf_tensor_info.n_dims; ++i) {
                // Reverse dimensions for tensor creation
                shape.insert(shape.begin(), gguf_tensor_info.ne[i]);
            }

            tensor::DataType dtype = gguf_type_to_tensor_dtype(gguf_tensor_info.type);
            
            // Calculate absolute offset for tensor data within the memory-mapped file
            uint64_t absolute_tensor_offset = tensor_data_start_offset + gguf_tensor_info.offset;
            const uint8_t* tensor_data_ptr = file_data_ptr + absolute_tensor_offset;

            // Construct tensor using the memory-mapped data. Set ownership to false.
            tensor::Tensor new_tensor(shape, dtype, const_cast<void*>(static_cast<const void*>(tensor_data_ptr)), false);


            // Assign to specific model tensors based on name
            std::cout << "  Processing tensor: " << gguf_tensor_info.name << ", Shape: [";
            for (size_t k = 0; k < shape.size(); ++k) {
                std::cout << shape[k] << (k == shape.size() - 1 ? "" : ", ");
            }
            std::cout << "], Type: " << gguf_tensor_info.type << std::endl;

            if (gguf_tensor_info.name == "token_embd.weight") { // Example from Llama.cpp, or "model.embed_tokens.weight"
                token_embeddings = std::move(new_tensor);
                std::cout << "  Assigned " << gguf_tensor_info.name << " to token_embeddings." << std::endl;
            } else {
                // Store all tensors in the map, including layer-specific ones
                m_tensors.emplace(gguf_tensor_info.name, std::move(new_tensor));
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing tensor " << gguf_tensor_info.name << ": " << e.what() << std::endl;
            return false;
        }
    }
    std::cout << "Model: All tensors processed." << std::endl;

    // Extract model parameters from GGUF metadata
    const auto& metadata = m_gguf_reader.get_metadata();
    std::cout << "Model: Extracting metadata..." << std::endl;

    auto get_metadata_value = [&](const std::string& key, auto default_value) {
        if (metadata.count(key)) {
            try {
                // Use std::get to safely access variant value
                if constexpr (std::is_same_v<decltype(default_value), uint32_t>) {
                    uint32_t value = std::get<uint32_t>(metadata.at(key));
                    return value;
                } else if constexpr (std::is_same_v<decltype(default_value), float>) {
                    float value = std::get<float>(metadata.at(key));
                    return value;
                } else if constexpr (std::is_same_v<decltype(default_value), int32_t>) { // Added for completeness, if needed
                    int32_t value = std::get<int32_t>(metadata.at(key));
                    return value;
                }
                 else {
                    // This case should ideally not be hit if all types are handled
                    std::cerr << "Warning: Unhandled metadata type for key '" << key << "'." << std::endl;
                    return default_value;
                }
            } catch (const std::bad_variant_access& e) {
                std::cerr << "Warning: Could not convert metadata key '" << key << "' value: " << e.what() << std::endl;
                return default_value;
            } catch (const std::exception& e) { // Catch-all for other potential issues
                std::cerr << "Warning: Error accessing metadata key '" << key << "': " << e.what() << std::endl;
                return default_value;
            }
        }
        return default_value;
    };

    n_vocab = get_metadata_value("tokenizer.ggml.model.n_vocab", (uint32_t)0); // Corrected key for n_vocab
    n_embd = get_metadata_value("llama.embedding_length", (uint32_t)0);
    n_head = get_metadata_value("llama.attention.head_count", (uint32_t)0);
    n_kv_head = get_metadata_value("llama.attention.head_count_kv", n_head);
    n_layer = get_metadata_value("llama.block_count", (uint32_t)0);
    n_rot = get_metadata_value("llama.rope.dimension_count", (uint32_t)0); // Rotary dimension
    norm_eps = get_metadata_value("llama.attention.layer_norm_rms_epsilon", 1e-5f); // Example key

    // Calculate head dimensions after n_head, n_kv_head, and n_embd are known.
    // Assuming layer 0 is representative for projection tensor shapes.
    if (n_head > 0) {
        // Query projection output dimension / n_head
        head_dim_q = m_tensors.at("model.layers.0.self_attn.q_proj.weight").shape()[1] / n_head;
    }
    if (n_kv_head > 0) {
        // Key/Value projection output dimension / n_kv_head
        head_dim_kv = m_tensors.at("model.layers.0.self_attn.k_proj.weight").shape()[1] / n_kv_head;
    }


    // Additional logging for model parameters
    std::cout << "Model parameters loaded: n_vocab=" << n_vocab << ", n_embd=" << n_embd
              << ", n_head=" << n_head << ", n_layer=" << n_layer << ", n_rot=" << n_rot
              << ", head_dim_q=" << head_dim_q << ", head_dim_kv=" << head_dim_kv
              << ", norm_eps=" << norm_eps << std::endl;

    // Implement weight tying: output_weight is the transpose of token_embeddings
    // This is a common practice in many LLMs to reduce parameter count.
    if (token_embeddings.shape().size() == 2 && token_embeddings.shape()[0] == n_vocab && token_embeddings.shape()[1] == n_embd) {
        output_weight = token_embeddings.transpose();
        std::cout << "  Assigned transposed token_embeddings to output_weight (weight tying)." << std::endl;
    } else {
        std::cerr << "Warning: Could not perform weight tying. token_embeddings has unexpected shape." << std::endl;
    }

    std::cout << "Model loaded successfully." << std::endl;
    return true;

}


LLM::LLM() : m_max_seq_len(2048), // Initialize m_max_seq_len first
               m_bos_token_id(-1), m_eos_token_id(-1), m_unk_token_id(-1), m_pad_token_id(-1),
               m_kv_cache_k({2048, 1}, tensor::F32), // Use literal for initialization
               m_kv_cache_v({2048, 1}, tensor::F32) { // Use literal for initialization
    std::cout << "LLM instance created." << std::endl;
}

bool LLM::load_model(const std::string& filepath) {
    std::cout << "LLM: Loading model from " << filepath << std::endl;
    if (!m_model.load(filepath)) {
        std::cerr << "Error: Failed to load model from " << filepath << std::endl;
        return false;
    }

    // Populate tokenizer data from GGUF metadata
    const auto& metadata = m_model.get_gguf_reader().get_metadata();

    auto get_metadata_val_or_default = [&](const std::string& key, auto default_val) {
        if (metadata.count(key)) {
            try {
                return std::get<decltype(default_val)>(metadata.at(key));
            } catch (const std::bad_variant_access& e) {
                std::cerr << "Warning: Metadata key '" << key << "' has unexpected type: " << e.what() << std::endl;
            }
        }
        return default_val;
    };

    // Extract tokens
    if (metadata.count("tokenizer.ggml.tokens")) {
        try {
            m_tokenizer_id_to_token = std::get<std::vector<std::string>>(metadata.at("tokenizer.ggml.tokens"));
            for (int i = 0; i < m_tokenizer_id_to_token.size(); ++i) {
                m_tokenizer_token_to_id[m_tokenizer_id_to_token[i]] = i;
            }
            std::cout << "LLM: Loaded " << m_tokenizer_id_to_token.size() << " tokenizer tokens." << std::endl;
        } catch (const std::bad_variant_access& e) {
            std::cerr << "Error: 'tokenizer.ggml.tokens' metadata has unexpected type: " << e.what() << std::endl;
            return false;
        }
    } else {
        std::cerr << "Error: 'tokenizer.ggml.tokens' not found in GGUF metadata." << std::endl;
        return false;
    }

    // Extract scores
    if (metadata.count("tokenizer.ggml.scores")) {
        try {
            m_tokenizer_scores = std::get<std::vector<float>>(metadata.at("tokenizer.ggml.scores"));
        } catch (const std::bad_variant_access& e) {
            std::cerr << "Error: 'tokenizer.ggml.scores' metadata has unexpected type: " << e.what() << std::endl;
            // Not critical, can continue without scores
        }
    }

    // Extract token types
    if (metadata.count("tokenizer.ggml.token_type")) {
        try {
            // GGUF token_type can be uint32_t, try to get it
            m_tokenizer_token_types = std::get<std::vector<uint32_t>>(metadata.at("tokenizer.ggml.token_type"));
        } catch (const std::bad_variant_access& e) {
            std::cerr << "Error: 'tokenizer.ggml.token_type' metadata has unexpected type: " << e.what() << std::endl;
            // Not critical, can continue without token types
        }
    }

    // Extract special token IDs
    m_bos_token_id = get_metadata_val_or_default("tokenizer.ggml.bos_token_id", -1);
    m_eos_token_id = get_metadata_val_or_default("tokenizer.ggml.eos_token_id", -1);
    m_unk_token_id = get_metadata_val_or_default("tokenizer.ggml.unk_token_id", -1);
    m_pad_token_id = get_metadata_val_or_default("tokenizer.ggml.pad_token_id", -1);

    std::cout << "LLM: Tokenizer loaded with BOS=" << m_bos_token_id
              << ", EOS=" << m_eos_token_id
              << ", UNK=" << m_unk_token_id
              << ", PAD=" << m_pad_token_id << std::endl;
    
    // Resize KV cache after n_embd is known
    m_kv_cache_k.resize({m_max_seq_len, m_model.n_embd});
    m_kv_cache_v.resize({m_max_seq_len, m_model.n_embd});
    std::cout << "LLM: KV Cache resized to [" << m_max_seq_len << ", " << m_model.n_embd << "]." << std::endl;

    return true;
}

std::vector<int> LLM::encode(const std::string& text) {
    std::vector<int> tokens;
    if (text.empty()) {
        return tokens;
    }

    // Add BOS token if available
    if (m_bos_token_id != -1) {
        tokens.push_back(m_bos_token_id);
    }

    std::string current_piece;
    for (char c : text) {
        current_piece += c;
        // Simple greedy tokenization: try to find the longest match
        // This is a very basic tokenization and not a full BPE/SentencePiece.
        // For a proper LLM, a more sophisticated tokenizer is needed.
        bool found = false;
        std::string longest_match;
        int longest_match_id = m_unk_token_id;

        for (size_t i = 0; i < current_piece.length(); ++i) {
            std::string sub_piece = current_piece.substr(0, current_piece.length() - i);
            if (m_tokenizer_token_to_id.count(sub_piece)) {
                longest_match = sub_piece;
                longest_match_id = m_tokenizer_token_to_id.at(sub_piece);
                found = true;
                break;
            }
        }

        if (found) {
            tokens.push_back(longest_match_id);
            current_piece = current_piece.substr(longest_match.length());
        }
    }

    // If there's any remaining piece, tokenize it as UNK or by characters
    if (!current_piece.empty()) {
        for (char c : current_piece) {
            std::string char_str(1, c);
            if (m_tokenizer_token_to_id.count(char_str)) {
                tokens.push_back(m_tokenizer_token_to_id.at(char_str));
            } else {
                tokens.push_back(m_unk_token_id);
            }
        }
    }
    
    return tokens;
}

std::string LLM::decode(const std::vector<int>& tokens) {
    std::string decoded_text;
    for (int token_id : tokens) {
        if (token_id == m_eos_token_id) {
            break; // Stop decoding at EOS token
        }
        if (token_id >= 0 && token_id < m_tokenizer_id_to_token.size()) {
            decoded_text += m_tokenizer_id_to_token[token_id];
        } else {
            // Handle unknown tokens during decoding, e.g., by adding a placeholder
            decoded_text += "[UNK]";
        }
    }
    return decoded_text;
}

std::string LLM::generate(const std::string& prompt, int max_tokens) {
    std::cout << "LLM: Starting text generation for prompt: \"" << prompt << "\"." << std::endl;

    if (max_tokens <= 0) {
        max_tokens = 128;
        std::cout << "LLM: max_tokens was invalid or not provided, defaulting to " << max_tokens << std::endl;
    } else {
        std::cout << "LLM: Generating up to " << max_tokens << " tokens." << std::endl;
    }

    if (m_model.token_embeddings.size() == 0) {
        std::cerr << "Error: Model not loaded or token embeddings missing." << std::endl;
        return "Error: Model not loaded.";
    }
    if (m_model.n_embd == 0 || m_model.n_head == 0 || m_model.n_layer == 0 || m_model.n_vocab == 0) {
        std::cerr << "Error: Model parameters not loaded correctly." << std::endl;
        return "Error: Model parameters missing.";
    }

    std::vector<int> current_sequence_tokens = encode(prompt);
    if (current_sequence_tokens.empty()) {
        std::cerr << "Error: Prompt tokenization failed." << std::endl;
        return "Error: Prompt tokenization failed.";
    }

    std::cout << "LLM: Tokenized prompt into " << current_sequence_tokens.size() << " tokens." << std::endl;
    std::cout << "LLM: Initial input tokens: ";
    for (int token_id : current_sequence_tokens) {
        std::cout << token_id << " ";
    }
    std::cout << std::endl;

    std::string generated_text_so_far = prompt;
    int tokens_generated_count = 0;
    uint32_t current_pos = 0; // Initialize current_pos for the loop

    // Declare tensors outside the loop to avoid repeated construction
    tensor::Tensor hidden_state({1, m_model.n_embd}, tensor::F32);
    tensor::Tensor residual({1, m_model.n_embd}, tensor::F32);
    tensor::Tensor attn_output({1, m_model.n_embd}, tensor::F32); // Corrected size: output of attention is n_embd
    tensor::Tensor ffn_output({1, m_model.n_embd}, tensor::F32);
    tensor::Tensor norm_input({1, m_model.n_embd}, tensor::F32);
    tensor::Tensor logits({1, m_model.n_vocab}, tensor::F32);

    // Main generation loop
    while (tokens_generated_count < max_tokens) {
        // Determine the current token to process
        int current_token_id_to_process;
        if (current_pos < current_sequence_tokens.size()) {
            // Process tokens from the initial prompt
            current_token_id_to_process = current_sequence_tokens[current_pos];
        } else {
            // If we are generating new tokens, the last token in current_sequence_tokens
            // is the one that was just generated and needs to be processed.
            current_token_id_to_process = current_sequence_tokens.back();
        }

        // 1. Get token embedding
        if (current_token_id_to_process < 0 || current_token_id_to_process >= m_model.n_vocab) {
             std::cerr << "Error: Invalid token ID: " << current_token_id_to_process << std::endl;
             return "Error: Invalid token ID.";
        }
        
        // This is a view into the token_embeddings weight tensor
        const float* embedding_ptr = static_cast<const float*>(m_model.token_embeddings.data()) + current_token_id_to_process * m_model.n_embd;
        std::memcpy(static_cast<float*>(hidden_state.data()), embedding_ptr, m_model.n_embd * sizeof(float));

        std::cout << "LLM: Processing token at position " << current_pos << " (ID: " << current_token_id_to_process << ")" << std::endl;

        // Propagate through N transformer layers
        for (uint32_t layer = 0; layer < m_model.n_layer; ++layer) {
            // Copy hidden_state to residual for residual connection
            std::memcpy(static_cast<float*>(residual.data()), static_cast<float*>(hidden_state.data()), m_model.n_embd * sizeof(float));

            // a. Pre-attention RMSNorm
            std::string attn_norm_weight_name = "model.layers." + std::to_string(layer) + ".input_layernorm.weight";
            const tensor::Tensor& attn_norm_weight = m_model.get_tensor(attn_norm_weight_name);
            rmsnorm(norm_input, hidden_state, attn_norm_weight, m_model.norm_eps);

            // b. Multi-head Attention
            std::string wq_name = "model.layers." + std::to_string(layer) + ".self_attn.q_proj.weight";
            std::string wk_name = "model.layers." + std::to_string(layer) + ".self_attn.k_proj.weight";
            std::string wv_name = "model.layers." + std::to_string(layer) + ".self_attn.v_proj.weight";
            std::string wo_name = "model.layers." + std::to_string(layer) + ".self_attn.o_proj.weight";

            const tensor::Tensor& wq = m_model.get_tensor(wq_name);
            const tensor::Tensor& wk = m_model.get_tensor(wk_name);
            const tensor::Tensor& wv = m_model.get_tensor(wv_name);
            const tensor::Tensor& wo = m_model.get_tensor(wo_name);
            
            multi_head_attention(
                attn_output, norm_input, wq, wk, wv, wo,
                m_model.n_head, m_model.n_embd, m_model.n_rot,
                m_model.head_dim_q, m_model.head_dim_kv,
                m_model.n_kv_head,
                current_pos,
                m_kv_cache_k, m_kv_cache_v
            );

            // c. Residual connection: hidden_state += attn_output
            float* hidden_data = static_cast<float*>(hidden_state.data());
            const float* residual_data = static_cast<const float*>(residual.data());
            const float* attn_output_data = static_cast<const float*>(attn_output.data());
            for (uint32_t i = 0; i < m_model.n_embd; ++i) {
                hidden_data[i] = residual_data[i] + attn_output_data[i];
            }
            
            // Copy hidden_state (after attention) to residual for next residual connection
            std::memcpy(static_cast<float*>(residual.data()), static_cast<float*>(hidden_state.data()), m_model.n_embd * sizeof(float));

            // d. Pre-FFN RMSNorm
            std::string ffn_norm_weight_name = "model.layers." + std::to_string(layer) + ".post_attention_layernorm.weight";
            const tensor::Tensor& ffn_norm_weight = m_model.get_tensor(ffn_norm_weight_name);
            rmsnorm(norm_input, hidden_state, ffn_norm_weight, m_model.norm_eps);

            // e. Feed-Forward Network
            std::string w1_name = "model.layers." + std::to_string(layer) + ".mlp.gate_proj.weight"; // Also called w1
            std::string w2_name = "model.layers." + std::to_string(layer) + ".mlp.down_proj.weight"; // Also called w2
            std::string w3_name = "model.layers." + std::to_string(layer) + ".mlp.up_proj.weight";   // Also called w3

            const tensor::Tensor& w1 = m_model.get_tensor(w1_name);
            const tensor::Tensor& w2 = m_model.get_tensor(w2_name);
            const tensor::Tensor& w3 = m_model.get_tensor(w3_name);

            ffn(ffn_output, norm_input, w1, w2, w3);

            // f. Residual connection: hidden_state += ffn_output
            const float* ffn_output_data = static_cast<const float*>(ffn_output.data()); // Declared here
            for (uint32_t i = 0; i < m_model.n_embd; ++i) {
                hidden_data[i] = residual_data[i] + ffn_output_data[i];
            }
        } // end for layer

        // Final RMSNorm
        std::string final_norm_weight_name = "output_norm.weight"; // Now correctly refers to the assigned name
        const tensor::Tensor& final_norm_weight = m_model.get_tensor(final_norm_weight_name);
        rmsnorm(norm_input, hidden_state, final_norm_weight, m_model.norm_eps);

        // Output Projection to Logits
        matmul(norm_input, m_model.output_weight, logits);
        
        // This part needs to be outside the if(current_pos >= current_sequence_tokens.size()) block
        // because the new token is predicted and added BEFORE current_pos is incremented to process it.
        // It's the logic for adding a *newly generated* token.
        if (current_pos >= current_sequence_tokens.size() - 1) { // If it's the last prompt token or a generated token
             // 3. Generate the next token (Argmax sampling)
            int next_token_id = 0;
            float max_logit = -FLT_MAX; // Use FLT_MAX from <cfloat>
            const float* logits_data = static_cast<const float*>(logits.data());

            for (uint32_t j = 0; j < m_model.n_vocab; ++j) {
                if (logits_data[j] > max_logit) {
                    max_logit = logits_data[j];
                    next_token_id = j;
                }
            }

            // Only add new token if we are in the generation phase OR if it's the very first token to generate
            if (current_pos == current_sequence_tokens.size() - 1 || tokens_generated_count < max_tokens) {
                 current_sequence_tokens.push_back(next_token_id);
                 tokens_generated_count++;

                 // Decode and append to generated text
                 std::string next_token_str = decode({next_token_id});
                 generated_text_so_far += next_token_str;
                 std::cout << "LLM: Generated token: " << next_token_str << std::endl;

                 // Stop condition: EOS token
                 if (next_token_id == m_eos_token_id) {
                     std::cout << "LLM: Stop condition met (EOS token)." << std::endl;
                     break;
                 }
            }
        }
        current_pos++; // Move to the next token position
    } // end while (tokens_generated_count < max_tokens)
    std::cout << "LLM: Generation finished." << std::endl;

    return generated_text_so_far;
} // closes LLM::generate
} // closes namespace engine
