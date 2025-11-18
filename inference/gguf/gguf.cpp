#include "gguf.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>

namespace gguf {

// Helper to read a uint64_t from the file
static uint64_t read_u64(std::ifstream& file) {
    uint64_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

// Helper to read a string from the file
static std::string read_string(std::ifstream& file) {
    uint64_t length = read_u64(file);
    std::string str(length, ' ');
    file.read(&str[0], length);
    return str;
}

// Helper to read a gguf_type from the file
static gguf_type read_gguf_type(std::ifstream& file) {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    return static_cast<gguf_type>(value);
}

GGUFReader::GGUFReader() {}

bool GGUFReader::load_from_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return false;
    }

    // Read header
    file.read(reinterpret_cast<char*>(&m_header.magic), sizeof(m_header.magic));
    file.read(reinterpret_cast<char*>(&m_header.version), sizeof(m_header.version));
    m_header.tensor_count = read_u64(file);
    m_header.metadata_kv_count = read_u64(file);

    if (m_header.magic != GGUF_MAGIC) {
        std::cerr << "Error: Invalid GGUF magic number." << std::endl;
        return false;
    }

    if (m_header.version != GGUF_VERSION) {
        std::cerr << "Error: Unsupported GGUF version." << std::endl;
        return false;
    }

    std::cout << "GGUF file loaded successfully. Version: " << m_header.version
              << ", Tensors: " << m_header.tensor_count
              << ", Metadata: " << m_header.metadata_kv_count << std::endl;

    // Read metadata
    for (uint64_t i = 0; i < m_header.metadata_kv_count; ++i) {
        std::string key = read_string(file);
        gguf_type type = read_gguf_type(file);

        // For now, we only care about string metadata values
        if (type == GGUF_TYPE_STRING) {
            std::string value = read_string(file);
            m_metadata[key] = value;
        } else {
            // Skip other types for now
            switch (type) {
                case GGUF_TYPE_UINT8: file.seekg(sizeof(uint8_t), std::ios_base::cur); break;
                case GGUF_TYPE_INT8: file.seekg(sizeof(int8_t), std::ios_base::cur); break;
                case GGUF_TYPE_UINT16: file.seekg(sizeof(uint16_t), std::ios_base::cur); break;
                case GGUF_TYPE_INT16: file.seekg(sizeof(int16_t), std::ios_base::cur); break;
                case GGUF_TYPE_UINT32: file.seekg(sizeof(uint32_t), std::ios_base::cur); break;
                case GGUF_TYPE_INT32: file.seekg(sizeof(int32_t), std::ios_base::cur); break;
                case GGUF_TYPE_FLOAT32: file.seekg(sizeof(float), std::ios_base::cur); break;
                case GGUF_TYPE_BOOL: file.seekg(sizeof(bool), std::ios_base::cur); break;
                case GGUF_TYPE_UINT64: file.seekg(sizeof(uint64_t), std::ios_base::cur); break;
                case GGUF_TYPE_INT64: file.seekg(sizeof(int64_t), std::ios_base::cur); break;
                case GGUF_TYPE_FLOAT64: file.seekg(sizeof(double), std::ios_base::cur); break;
                case GGUF_TYPE_ARRAY: {
                    gguf_type array_type = read_gguf_type(file);
                    uint64_t array_len = read_u64(file);
                    // This is a simplified skip. A proper implementation would need to know the size of each element.
                    // For now, we'll just skip a fixed amount, which is incorrect for variable-sized elements.
                    // This needs to be improved if we need to support array metadata.
                    std::cerr << "Warning: Skipping array metadata. Full implementation needed." << std::endl;
                    // For now, just skip a placeholder amount. This is NOT robust.
                    file.seekg(array_len * sizeof(uint32_t), std::ios_base::cur); // Assuming array of uint32 for skipping
                    break;
                }
                default:
                    std::cerr << "Warning: Unknown GGUF metadata type encountered: " << type << std::endl;
                    // Attempt to skip a reasonable amount to continue parsing
                    file.seekg(sizeof(uint64_t), std::ios_base::cur);
                    break;
            }
        }
    }

    // Read tensor info
    for (uint64_t i = 0; i < m_header.tensor_count; ++i) {
        gguf_tensor_info tensor_info;
        tensor_info.name = read_string(file);
        file.read(reinterpret_cast<char*>(&tensor_info.n_dims), sizeof(tensor_info.n_dims));
        for (uint32_t j = 0; j < tensor_info.n_dims; ++j) {
            tensor_info.ne[j] = read_u64(file);
        }
        tensor_info.type = read_gguf_type(file);
        tensor_info.offset = read_u64(file); // Offset to tensor data

        m_tensor_infos.push_back(tensor_info);
    }

    // Store the current file position as the start of tensor data
    m_tensor_data_offset = file.tellg();

    file.close();
    return true;
}

} // namespace gguf