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
        // TODO: Add more type conversions as needed
        default: throw std::runtime_error("Unsupported GGUF data type for tensor conversion.");
    }
}

Model::Model()
    : token_embeddings({}, tensor::F32),
      rms_norm_weight({}, tensor::F32),
      wq({}, tensor::F32), wk({}, tensor::F32), wv({}, tensor::F32), wo({}, tensor::F32),
      w1({}, tensor::F32), w2({}, tensor::F32), w3({}, tensor::F32),
      output_weight({}, tensor::F32) {
    // Default constructor for Model
    std::cout << "Model default constructor called." << std::endl;
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
    std::cout << "Model: Starting tensor processing. Tensor data starts at offset: " << tensor_data_start_offset << std::endl;

    // Process tensor infos from m_gguf_reader and create tensor::Tensor objects
    for (const auto& gguf_tensor_info : m_gguf_reader.get_tensor_infos()) {
        try {
            std::vector<uint64_t> shape;
            // GGUF stores dimensions in reverse order [d3, d2, d1, d0] for row-major.
            // Tensor expects [d0, d1, d2, d3] for row-major for C-style arrays (last dim contiguous).
            // We need to reverse them here for compatibility with common ML frameworks.
            std::cout << "  Processing tensor: " << gguf_tensor_info.name
                      << ", Dims: " << gguf_tensor_info.n_dims
                      << ", GGUF Type: " << gguf_tensor_info.type;

            std::cout << ", Shape: [";
            for (uint32_t i = 0; i < gguf_tensor_info.n_dims; ++i) {
                // Reverse dimensions for tensor creation
                shape.insert(shape.begin(), gguf_tensor_info.ne[i]);
                std::cout << gguf_tensor_info.ne[i] << (i < gguf_tensor_info.n_dims - 1 ? ", " : "");
            }
            std::cout << "]" << std::endl;

            tensor::DataType dtype = gguf_type_to_tensor_dtype(gguf_tensor_info.type);
            
            // Calculate absolute offset for tensor data within the memory-mapped file
            uint64_t absolute_tensor_offset = tensor_data_start_offset + gguf_tensor_info.offset;
            const uint8_t* tensor_data_ptr = file_data_ptr + absolute_tensor_offset;

            // Construct tensor using the memory-mapped data. Set ownership to false.
            tensor::Tensor new_tensor(shape, dtype, const_cast<void*>(static_cast<const void*>(tensor_data_ptr)), false);
            std::cout << "  Created tensor::Tensor for " << gguf_tensor_info.name
                      << " with size " << new_tensor.size() << " elements ("
                      << new_tensor.size() * tensor::Tensor::get_element_size(dtype) << " bytes)" << std::endl;

            // Assign to specific model tensors based on name
            if (gguf_tensor_info.name == "token_embd.weight") { // Example from Llama.cpp, or "model.embed_tokens.weight"
                token_embeddings = std::move(new_tensor);
                std::cout << "  Assigned " << gguf_tensor_info.name << " to token_embeddings." << std::endl;
            } else if (gguf_tensor_info.name == "output.weight") {
                output_weight = std::move(new_tensor);
                std::cout << "  Assigned " << gguf_tensor_info.name << " to output_weight." << std::endl;
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
                    std::cout << "  Metadata: " << key << " = " << value << std::endl;
                    return value;
                } else if constexpr (std::is_same_v<decltype(default_value), float>) {
                    float value = std::get<float>(metadata.at(key));
                    std::cout << "  Metadata: " << key << " = " << value << std::endl;
                    return value;
                } else if constexpr (std::is_same_v<decltype(default_value), int32_t>) { // Added for completeness, if needed
                    int32_t value = std::get<int32_t>(metadata.at(key));
                    std::cout << "  Metadata: " << key << " = " << value << std::endl;
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
        std::cout << "  Metadata: " << key << " not found, using default value." << std::endl;
        return default_value;
    };

    n_vocab = get_metadata_value("tokenizer.ggml.model.n_vocab", (uint32_t)0); // Corrected key for n_vocab
    n_embd = get_metadata_value("llama.embedding_length", (uint32_t)0);
    n_head = get_metadata_value("llama.attention.head_count", (uint32_t)0);
    n_layer = get_metadata_value("llama.block_count", (uint32_t)0);
    n_rot = get_metadata_value("llama.context_length", (uint32_t)0); // Often n_rot is context_length
    norm_eps = get_metadata_value("llama.attention.layer_norm_rms_epsilon", 1e-5f); // Example key


    // Additional logging for model parameters
    std::cout << "Model parameters loaded: n_vocab=" << n_vocab << ", n_embd=" << n_embd
              << ", n_head=" << n_head << ", n_layer=" << n_layer << ", n_rot=" << n_rot
              << ", norm_eps=" << norm_eps << std::endl;

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

    std::vector<int> input_tokens = encode(prompt);
    if (input_tokens.empty()) {
        std::cerr << "Error: Prompt tokenization failed." << std::endl;
        return "Error: Prompt tokenization failed.";
    }

    std::cout << "LLM: Tokenized prompt into " << input_tokens.size() << " tokens." << std::endl;
    std::cout << "LLM: Initial input tokens: ";
    for (int token_id : input_tokens) {
        std::cout << token_id << " ";
    }
    std::cout << std::endl;

    std::string generated_text_so_far = prompt;
    
    // Hidden state for the current token being processed
    tensor::Tensor hidden_state({1, m_model.n_embd}, tensor::F32);
    // Residual connection tensor
    tensor::Tensor residual({1, m_model.n_embd}, tensor::F32);
    // Attention output tensor
    tensor::Tensor attn_output({1, m_model.n_embd}, tensor::F32);
    // FFN output tensor
    tensor::Tensor ffn_output({1, m_model.n_embd}, tensor::F32);
    // Normalized input for attention/ffn
    tensor::Tensor norm_input({1, m_model.n_embd}, tensor::F32);
    // Logits for next token prediction
    tensor::Tensor logits({1, m_model.n_vocab}, tensor::F32);

    for (uint32_t current_pos = 0; current_pos < input_tokens.size() + max_tokens; ++current_pos) {
        int current_token_id;
        if (current_pos < input_tokens.size()) {
            current_token_id = input_tokens[current_pos];
        } else {
            // If beyond initial prompt, use the previously generated token
            current_token_id = input_tokens.back();
            if (current_pos - input_tokens.size() >= max_tokens) {
                 std::cout << "LLM: Maximum generation length reached." << std::endl;
                 break; // Stop if generated max_tokens beyond prompt
            }
        }
        
        // 1. Get token embedding
        // Copy embedding for current_token_id into hidden_state
        if (current_token_id < 0 || current_token_id >= m_model.n_vocab) {
             std::cerr << "Error: Invalid token ID: " << current_token_id << std::endl;
             return "Error: Invalid token ID.";
        }
        
        // This is a view into the token_embeddings weight tensor
        const float* embedding_ptr = static_cast<const float*>(m_model.token_embeddings.data()) + current_token_id * m_model.n_embd;
        std::memcpy(static_cast<float*>(hidden_state.data()), embedding_ptr, m_model.n_embd * sizeof(float));

        std::cout << "LLM: Processing token at position " << current_pos << " (ID: " << current_token_id << ")" << std::endl;

        // Propagate through N transformer layers
        for (uint32_t layer = 0; layer < m_model.n_layer; ++layer) {
            // Copy hidden_state to residual for residual connection
            std::memcpy(static_cast<float*>(residual.data()), static_cast<float*>(hidden_state.data()), m_model.n_embd * sizeof(float));

            // a. Pre-attention RMSNorm
            std::string attn_norm_weight_name = "blk." + std::to_string(layer) + ".attn_norm.weight";
            const tensor::Tensor& attn_norm_weight = m_model.get_tensor(attn_norm_weight_name);
            rmsnorm(norm_input, hidden_state, attn_norm_weight, m_model.norm_eps);

            // b. Multi-head Attention
            std::string wq_name = "blk." + std::to_string(layer) + ".attn_q.weight";
            std::string wk_name = "blk." + std::to_string(layer) + ".attn_k.weight";
            std::string wv_name = "blk." + std::to_string(layer) + ".attn_v.weight";
            std::string wo_name = "blk." + std::to_string(layer) + ".attn_output.weight";

            const tensor::Tensor& wq = m_model.get_tensor(wq_name);
            const tensor::Tensor& wk = m_model.get_tensor(wk_name);
            const tensor::Tensor& wv = m_model.get_tensor(wv_name);
            const tensor::Tensor& wo = m_model.get_tensor(wo_name);
            
            multi_head_attention(
                attn_output, norm_input, wq, wk, wv, wo,
                m_model.n_head, m_model.n_embd, m_model.n_rot,
                current_pos, m_kv_cache_k, m_kv_cache_v
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
            std::string ffn_norm_weight_name = "blk." + std::to_string(layer) + ".ffn_norm.weight";
            const tensor::Tensor& ffn_norm_weight = m_model.get_tensor(ffn_norm_weight_name);
            rmsnorm(norm_input, hidden_state, ffn_norm_weight, m_model.norm_eps);

            // e. Feed-Forward Network
            std::string w1_name = "blk." + std::to_string(layer) + ".ffn_gate.weight"; // Also called w1
            std::string w2_name = "blk." + std::to_string(layer) + ".ffn_down.weight"; // Also called w2
            std::string w3_name = "blk." + std::to_string(layer) + ".ffn_up.weight";   // Also called w3

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
        std::string final_norm_weight_name = "output_norm.weight"; // Example name for final norm
        const tensor::Tensor& final_norm_weight = m_model.get_tensor(final_norm_weight_name);
        rmsnorm(norm_input, hidden_state, final_norm_weight, m_model.norm_eps);

        // Output Projection to Logits
        matmul(norm_input, m_model.output_weight, logits);
        
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

        // Add the new token to the sequence
        input_tokens.push_back(next_token_id);

        // Decode and append to generated text
        if (current_pos >= input_tokens.size() - 1) { // Only append if it's a newly generated token
            std::string next_token_str = decode({next_token_id});
            generated_text_so_far += next_token_str;
            std::cout << "LLM: Generated token: " << next_token_str << std::endl;
        }

        // Stop condition: EOS token or max tokens reached
        if (next_token_id == m_eos_token_id) {
            std::cout << "LLM: Stop condition met (EOS token)." << std::endl;
            break;
        }
    } // end for current_pos
    std::cout << "LLM: Generation finished." << std::endl;

    return generated_text_so_far;
} // closes LLM::generate
} // closes namespace engine
