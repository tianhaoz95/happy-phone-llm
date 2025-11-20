#ifndef HAPPY_PHONE_LLM_GGUF_H
#define HAPPY_PHONE_LLM_GGUF_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <variant> // For std::variant

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

// Define a variant to hold different types of metadata values
using GGUFMetadataValue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float, bool, std::string, uint64_t, int64_t, double,
    std::vector<uint8_t>, std::vector<int8_t>, std::vector<uint16_t>, std::vector<int16_t>, std::vector<uint32_t>, std::vector<int32_t>, std::vector<float>, std::vector<bool>, std::vector<std::string>, std::vector<uint64_t>, std::vector<int64_t>, std::vector<double>
>;

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
    const std::map<std::string, GGUFMetadataValue>& get_metadata() const { return m_metadata; } // Changed type
    uint64_t get_tensor_data_offset() const { return m_tensor_data_offset; }
    uint32_t get_version() const { return m_header.version; }

    // Add for memory mapping
    ~GGUFReader(); // Destructor to unmap memory
    const uint8_t* get_file_data() const { return m_file_data; }
    size_t get_file_size() const { return m_file_size; }
    size_t get_mapped_size() const { return m_mapped_size; }

    // Rule of Five for resource management:
    // Delete copy constructor and copy assignment operator
    GGUFReader(const GGUFReader&) = delete;
    GGUFReader& operator=(const GGUFReader&) = delete;

    // Declare move constructor and move assignment operator
    GGUFReader(GGUFReader&& other) noexcept;
    GGUFReader& operator=(GGUFReader&& other) noexcept;

private:
    gguf_header m_header;
    std::map<std::string, GGUFMetadataValue> m_metadata; // Changed type
    std::vector<gguf_tensor_info> m_tensor_infos;
    uint64_t m_tensor_data_offset; // Stores the file offset where tensor data begins
    uint64_t m_metadata_data_offset; // Stores the file offset where metadata begins

    // Memory mapping members
    int m_file_descriptor;
    size_t m_file_size;
    uint8_t* m_file_data; // Pointer to the memory-mapped file data
    size_t m_mapped_size; // The actual size of the mapped region
};

} // namespace gguf

#endif // HAPPY_PHONE_LLM_GGUF_H
