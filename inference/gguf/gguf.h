#ifndef HAPPY_PHONE_LLM_GGUF_H
#define HAPPY_PHONE_LLM_GGUF_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace gguf {

// GGUF magic number
const uint32_t GGUF_MAGIC = 0x46554747; // GGUF

// GGUF version
const uint32_t GGUF_VERSION = 3;

enum gguf_type {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

struct gguf_header {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

struct gguf_tensor_info {
    std::string name;
    uint32_t n_dims;
    uint64_t ne[4]; // number of elements
    gguf_type type;
    uint64_t offset;
};

class GGUFReader {
public:
    GGUFReader();
    bool load_from_file(const std::string& filepath);

    const std::vector<gguf_tensor_info>& get_tensor_infos() const { return m_tensor_infos; }
    const std::map<std::string, std::string>& get_metadata() const { return m_metadata; }
    uint64_t get_tensor_data_offset() const { return m_tensor_data_offset; }
    uint32_t get_version() const { return m_header.version; }

private:
    gguf_header m_header;
    std::map<std::string, std::string> m_metadata;
    std::vector<gguf_tensor_info> m_tensor_infos;
    uint64_t m_tensor_data_offset; // Stores the file offset where tensor data begins
};

} // namespace gguf

#endif // HAPPY_PHONE_LLM_GGUF_H
