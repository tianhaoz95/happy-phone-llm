#include "engine.h"
#include <iostream>
#include <fstream> // For file seeking
#include <stdexcept> // For std::runtime_error
#include <string> // For std::stoul, std::stof
#include <sstream> // For std::stringstream

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


    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return false;
    }

    uint64_t tensor_data_start_offset = m_gguf_reader.get_tensor_data_offset();
    std::cout << "Model: Starting tensor processing. Tensor data starts at offset: " << tensor_data_start_offset << std::endl;

    // Process tensor infos from m_gguf_reader and create tensor::Tensor objects
    for (const auto& gguf_tensor_info : m_gguf_reader.get_tensor_infos()) {
        try {
            std::vector<uint64_t> shape;
            std::cout << "  Processing tensor: " << gguf_tensor_info.name
                      << ", Dims: " << gguf_tensor_info.n_dims
                      << ", GGUF Type: " << gguf_tensor_info.type;

            std::cout << ", Shape: [";
            for (uint32_t i = 0; i < gguf_tensor_info.n_dims; ++i) {
                shape.push_back(gguf_tensor_info.ne[i]);
                std::cout << gguf_tensor_info.ne[i] << (i < gguf_tensor_info.n_dims - 1 ? ", " : "");
            }
            std::cout << "]" << std::endl;

            tensor::DataType dtype = gguf_type_to_tensor_dtype(gguf_tensor_info.type);
            tensor::Tensor new_tensor(shape, dtype);
            std::cout << "  Created tensor::Tensor for " << gguf_tensor_info.name
                      << " with size " << new_tensor.size() << " elements ("
                      << new_tensor.size() * tensor::Tensor::get_element_size(dtype) << " bytes)" << std::endl;

            // Calculate absolute offset for tensor data
            uint64_t absolute_tensor_offset = tensor_data_start_offset + gguf_tensor_info.offset;
            std::cout << "  Reading data for " << gguf_tensor_info.name << " from file offset: " << absolute_tensor_offset << std::endl;

            // Seek to the tensor data in the file
            file.seekg(absolute_tensor_offset, std::ios::beg);

            // Read tensor data into the Tensor object
            file.read(static_cast<char*>(new_tensor.data()), new_tensor.size() * tensor::Tensor::get_element_size(dtype));
            std::cout << "  Successfully read data for " << gguf_tensor_info.name << std::endl;

            // Assign to specific model tensors based on name
            if (gguf_tensor_info.name == "model.embed_tokens.weight") { // Dummy model name
                token_embeddings = std::move(new_tensor);
                std::cout << "  Assigned " << gguf_tensor_info.name << " to token_embeddings." << std::endl;
            } else if (gguf_tensor_info.name == "blk.0.attn_norm.weight") { // Example name
                rms_norm_weight = std::move(new_tensor);
                std::cout << "  Assigned " << gguf_tensor_info.name << " to rms_norm_weight." << std::endl;
            }
            // TODO: Add more assignments for wq, wk, wv, wo, w1, w2, w3, output_weight
            // based on actual GGUF tensor names.

            m_tensors.emplace(gguf_tensor_info.name, std::move(new_tensor));
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
                if constexpr (std::is_same_v<decltype(default_value), uint32_t>) {
                    uint32_t value = static_cast<uint32_t>(std::stoul(metadata.at(key)));
                    std::cout << "  Metadata: " << key << " = " << value << std::endl;
                    return value;
                } else if constexpr (std::is_same_v<decltype(default_value), float>) {
                    float value = std::stof(metadata.at(key));
                    std::cout << "  Metadata: " << key << " = " << value << std::endl;
                    return value;
                } else {
                    auto value = metadata.at(key);
                    std::cout << "  Metadata: " << key << " = " << value << std::endl;
                    return default_value; // Return default if type not handled for logging
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not convert metadata key '" << key << "' value '" << metadata.at(key) << "': " << e.what() << std::endl;
                return default_value;
            }
        }
        std::cout << "  Metadata: " << key << " not found, using default value." << std::endl;
        return default_value;
    };

    n_vocab = get_metadata_value("vocab_size", (uint32_t)0);
    n_embd = get_metadata_value("embedding_length", (uint32_t)0);
    n_head = get_metadata_value("head_count", (uint32_t)0);
    n_layer = get_metadata_value("block_count", (uint32_t)0);
    n_rot = get_metadata_value("rope_dimension_count", (uint32_t)0);
    norm_eps = get_metadata_value("layer_norm_rms_epsilon", 1e-5f);

    std::cout << "Model parameters loaded: n_vocab=" << n_vocab << ", n_embd=" << n_embd
              << ", n_head=" << n_head << ", n_layer=" << n_layer << ", n_rot=" << n_rot
              << ", norm_eps=" << norm_eps << std::endl;

    file.close();
    std::cout << "Model loaded successfully." << std::endl;
    return true;

}

LLM::LLM() {
    std::cout << "LLM instance created." << std::endl;
}

bool LLM::load_model(const std::string& filepath) {
    std::cout << "LLM: Loading model from " << filepath << std::endl;
    if (!m_model.load(filepath)) {
        std::cerr << "Error: Failed to load model from " << filepath << std::endl;
        return false;
    }

    // // Populate tokenizer maps from GGUF metadata
    // const auto& gguf_reader = m_model.get_gguf_reader();
    // const auto& metadata = gguf_reader.get_metadata();

    // if (metadata.count("tokenizer.ggml.tokens")) {
    //     std::cout << "LLM: Processing tokenizer metadata (tokenizer.ggml.tokens detected)." << std::endl;
    //     // The tokenizer tokens are stored as an array of strings in GGUF.
    //     // The current GGUFReader only exposes metadata as map<string, string>.
    //     // This part needs a more robust GGUF metadata parsing for arrays.
    //     // For now, we'll simulate by looking for individual token entries if they exist,
    //     // or assume a simple character-based tokenizer for demonstration.
    //     std::cout << "Warning: Tokenizer loading from GGUF metadata is simplified. "
    //               << "Full GGUF array parsing for tokens is not yet implemented." << std::endl;

    //     // Dummy tokenizer for now if actual tokens are not easily parsed
    //     if (m_tokenizer_id_to_token.empty()) {
    //         for (int i = 0; i < 256; ++i) { // ASCII characters
    //             m_tokenizer_id_to_token[i] = std::string(1, static_cast<char>(i));
    //             m_tokenizer_token_to_id[std::string(1, static_cast<char>(i))] = i;
    //         }
    //         m_tokenizer_id_to_token[0] = "<UNK>"; // Example for unknown token
    //         m_tokenizer_token_to_id["<UNK>"] = 0;
    //         // Add tokens from the dummy GGUF
    //         if (metadata.count("token_list")) {
    //             // This assumes token_list is a single string with space-separated tokens for simplicity
    //             // In a real scenario, this would be an array of strings.
    //             std::string token_list_str = metadata.at("token_list");
    //             std::stringstream ss(token_list_str);
    //             std::string token;
    //             int current_id = m_tokenizer_id_to_token.size(); // Start from existing size
    //             while (ss >> token) {
    //                 if (m_tokenizer_token_to_id.find(token) == m_tokenizer_token_to_id.end()) {
    //                     m_tokenizer_id_to_token[current_id] = token;
    //                     m_tokenizer_token_to_id[token] = current_id;
    //                     current_id++;
    //                 }
    //             }
    //         }
    //         std::cout << "LLM: Initialized dummy character tokenizer with " << m_tokenizer_id_to_token.size() << " tokens." << std::endl;
    //     }
    // } else {
    //     std::cout << "LLM: No 'tokenizer.ggml.tokens' metadata found. Initializing default dummy tokenizer." << std::endl;
        if (m_tokenizer_id_to_token.empty()) {
            for (int i = 0; i < 256; ++i) { // ASCII characters
                m_tokenizer_id_to_token[i] = std::string(1, static_cast<char>(i));
                m_tokenizer_token_to_id[std::string(1, static_cast<char>(i))] = i;
            }
            m_tokenizer_id_to_token[0] = "<UNK>"; // Example for unknown token
            m_tokenizer_token_to_id["<UNK>"] = 0;

            // Try to use "token_list" metadata from the dummy GGUF
            // if (metadata.count("token_list")) {
            //     std::string token_list_str = metadata.at("token_list");
            //     std::stringstream ss(token_list_str);
            //     std::string token;
            //     int current_id = m_tokenizer_id_to_token.size();
            //     while (ss >> token) {
            //         if (m_tokenizer_token_to_id.find(token) == m_tokenizer_token_to_id.end()) {
            //             m_tokenizer_id_to_token[current_id] = token;
            //             m_tokenizer_token_to_id[token] = current_id;
            //             current_id++;
            //         }
            //     }
            // }
            std::cout << "LLM: Initialized default dummy tokenizer with " << m_tokenizer_id_to_token.size() << " tokens." << std::endl;
        }



    return true;
}

std::string LLM::generate(const std::string& prompt, int max_tokens) {
    std::cout << "LLM: Starting text generation for prompt: \"" << prompt << "\"." << std::endl;

    // Default max_tokens to 128 if not provided or invalid
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

    // 1. Tokenize the input prompt
    std::vector<int> input_tokens;
    std::string current_token_str;
    for (char c : prompt) {
        current_token_str += c;
        if (m_tokenizer_token_to_id.count(current_token_str)) {
            input_tokens.push_back(m_tokenizer_token_to_id[current_token_str]);
            current_token_str = "";
        }
    }
    // Add any remaining part of the prompt as an unknown token or handle as needed
    if (!current_token_str.empty()) {
        // For simplicity, just add the first character of the remaining string as a token
        // In a real tokenizer, this would be more sophisticated (e.g., byte-pair encoding)
        if (m_tokenizer_token_to_id.count(std::string(1, current_token_str[0]))) {
             input_tokens.push_back(m_tokenizer_token_to_id[std::string(1, current_token_str[0])]);
        } else {
            input_tokens.push_back(0); // Unknown token ID
        }
    }

    std::cout << "LLM: Tokenized prompt into " << input_tokens.size() << " tokens." << std::endl;
    std::cout << "LLM: Initial input tokens: ";
    for (int token_id : input_tokens) {
        std::cout << token_id << " ";
    }
    std::cout << std::endl;


    std::string generated_text = prompt;

    // Placeholder for current hidden state (embedding of the last token)
    // In a real LLM, this would be a sequence of hidden states
    tensor::Tensor current_hidden_state({1, m_model.n_embd}, tensor::F32);
    // Initialize with dummy data for now
    for (size_t i = 0; i < current_hidden_state.size(); ++i) {
        static_cast<float*>(current_hidden_state.data())[i] = 0.1f;
    }


    for (int i = 0; i < max_tokens; ++i) {
        // Simulate a single transformer block forward pass
        std::cout << "LLM: Performing forward pass (step " << i + 1 << "/" << max_tokens << ")" << std::endl;

        // 2. Perform forward pass through the model (simplified placeholder)
        // This would involve calling various LLM layers (attention, FFN, etc.)
        // using the loaded tensors (m_model.m_tensors) and input_tokens.
        // The output would be logits for the next token.

        // Dummy Tensors for kernel calls
        tensor::Tensor dummy_output({1, m_model.n_embd}, tensor::F32);
        tensor::Tensor dummy_query({1, m_model.n_embd}, tensor::F32);
        tensor::Tensor dummy_key({1, m_model.n_embd}, tensor::F32);
        tensor::Tensor dummy_value({1, m_model.n_embd}, tensor::F32);
        tensor::Tensor dummy_weight({m_model.n_embd}, tensor::F32);
        tensor::Tensor dummy_bias({m_model.n_embd}, tensor::F32);

        // Example calls to kernel functions with dummy data
        // In a real implementation, these would use actual model weights and intermediate activations
        happy_phone_llm::kernel::rmsnorm(dummy_output, current_hidden_state, dummy_weight, m_model.norm_eps);
        happy_phone_llm::kernel::attention(dummy_output, dummy_query, dummy_key, dummy_value, 1.0f);
        happy_phone_llm::kernel::ffn(dummy_output, current_hidden_state, dummy_weight, dummy_weight, dummy_bias, dummy_bias);

        // Simulate output logits
        tensor::Tensor logits({1, m_model.n_vocab}, tensor::F32);
        // Fill with some dummy logits
        for (size_t j = 0; j < logits.size(); ++j) {
            static_cast<float*>(logits.data())[j] = static_cast<float>(rand()) / RAND_MAX;
        }

        // 3. Generate the next token (placeholder - e.g., simple sampling or argmax)
        int next_token_id = 0;
        float max_logit = -1.0f;
        for (size_t j = 0; j < m_model.n_vocab; ++j) {
            if (static_cast<float*>(logits.data())[j] > max_logit) {
                max_logit = static_cast<float*>(logits.data())[j];
                next_token_id = j;
            }
        }

        std::string next_char_str = "";
        if (m_tokenizer_id_to_token.count(next_token_id)) {
            next_char_str = m_tokenizer_id_to_token[next_token_id];
        } else {
            next_char_str = "[UNK]"; // Unknown token
        }

        generated_text += next_char_str;
        input_tokens.push_back(next_token_id); // Add to context for next step

        std::cout << "LLM: Generated token: " << next_char_str << std::endl;

        // Simple stop condition (e.g., end of sentence token or max tokens reached)
        if (next_char_str == "." || next_char_str == "<EOS>") {
            std::cout << "LLM: Stop condition met (token: " << next_char_str << ")." << std::endl;
            break;
        }
    }
    std::cout << "LLM: Generation finished." << std::endl;

    return generated_text;
} // closes LLM::generate
} // closes namespace engine