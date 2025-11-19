#include "gguf.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <sys/mman.h>   // For mmap, munmap
#include <sys/stat.h>   // For fstat
#include <fcntl.h>      // For open
#include <unistd.h>     // For close
#include <cstring>      // For std::memcpy

namespace gguf {

// Helper to read a uint8_t from the current memory pointer and advance it
static uint8_t read_u8_mmap(const uint8_t*& ptr) {
    uint8_t value;
    std::memcpy(&value, ptr, sizeof(value));
    ptr += sizeof(value);
    return value;
}

// Helper to read an int8_t from the current memory pointer and advance it
static int8_t read_i8_mmap(const uint8_t*& ptr) {
    int8_t value;
    std::memcpy(&value, ptr, sizeof(value));
    ptr += sizeof(value);
    return value;
}

// Helper to read a uint16_t from the current memory pointer and advance it
static uint16_t read_u16_mmap(const uint8_t*& ptr) {
    uint16_t value;
    std::memcpy(&value, ptr, sizeof(value));
    ptr += sizeof(value);
    return value;
}

// Helper to read an int16_t from the current memory pointer and advance it
static int16_t read_i16_mmap(const uint8_t*& ptr) {
    int16_t value;
    std::memcpy(&value, ptr, sizeof(value));
    ptr += sizeof(value);
    return value;
}

// Helper to read a uint32_t from the current memory pointer and advance it
static uint32_t read_u32_mmap(const uint8_t*& ptr) {
    uint32_t value;
    std::memcpy(&value, ptr, sizeof(value));
    ptr += sizeof(value);
    return value;
}

// Helper to read an int32_t from the current memory pointer and advance it
static int32_t read_i32_mmap(const uint8_t*& ptr) {
    int32_t value;
    std::memcpy(&value, ptr, sizeof(value));
    ptr += sizeof(value);
    return value;
}

// Helper to read a float32 from the current memory pointer and advance it
static float read_f32_mmap(const uint8_t*& ptr) {
    float value;
    std::memcpy(&value, ptr, sizeof(value));
    ptr += sizeof(value);
    return value;
}

// Helper to read a bool from the current memory pointer and advance it
static bool read_bool_mmap(const uint8_t*& ptr) {
    bool value;
    std::memcpy(&value, ptr, sizeof(value));
    ptr += sizeof(value);
    return value;
}

// Helper to read a uint64_t from the current memory pointer and advance it
static uint64_t read_u64_mmap(const uint8_t*& ptr) {
    uint64_t value;
    std::memcpy(&value, ptr, sizeof(value));
    ptr += sizeof(value);
    return value;
}

// Helper to read an int64_t from the current memory pointer and advance it
static int64_t read_i64_mmap(const uint8_t*& ptr) {
    int64_t value;
    std::memcpy(&value, ptr, sizeof(value));
    ptr += sizeof(value);
    return value;
}

// Helper to read a float64 from the current memory pointer and advance it
static double read_f64_mmap(const uint8_t*& ptr) {
    double value;
    std::memcpy(&value, ptr, sizeof(value));
    ptr += sizeof(value);
    return value;
}

// Helper to read a string from the current memory pointer and advance it
static std::string read_string_mmap(const uint8_t*& ptr) {
    uint64_t length = read_u64_mmap(ptr);
    std::string str(reinterpret_cast<const char*>(ptr), length);
    ptr += length;
    return str;
}

// Helper to read a gguf_type from the current memory pointer and advance it
static gguf_type read_gguf_type_mmap(const uint8_t*& ptr) {
    uint32_t value;
    std::memcpy(&value, ptr, sizeof(value));
    ptr += sizeof(value);
    return static_cast<gguf_type>(value);
}

// Helper to read a GGUFMetadataValue from the current memory pointer and advance it
static GGUFMetadataValue read_gguf_value_mmap(const uint8_t*& ptr, gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8: return read_u8_mmap(ptr);
        case GGUF_TYPE_INT8: return read_i8_mmap(ptr);
        case GGUF_TYPE_UINT16: return read_u16_mmap(ptr);
        case GGUF_TYPE_INT16: return read_i16_mmap(ptr);
        case GGUF_TYPE_UINT32: return read_u32_mmap(ptr);
        case GGUF_TYPE_INT32: return read_i32_mmap(ptr);
        case GGUF_TYPE_FLOAT32: return read_f32_mmap(ptr);
        case GGUF_TYPE_BOOL: return read_bool_mmap(ptr);
        case GGUF_TYPE_STRING: return read_string_mmap(ptr);
        case GGUF_TYPE_UINT64: return read_u64_mmap(ptr);
        case GGUF_TYPE_INT64: return read_i64_mmap(ptr);
        case GGUF_TYPE_FLOAT64: return read_f64_mmap(ptr);
        case GGUF_TYPE_ARRAY: {
            gguf_type array_type = read_gguf_type_mmap(ptr);
            uint64_t array_len = read_u64_mmap(ptr);

            // Read array elements into a vector based on array_type
            switch (array_type) {
                case GGUF_TYPE_UINT8: {
                    std::vector<uint8_t> vec;
                    vec.reserve(array_len);
                    for (uint64_t i = 0; i < array_len; ++i) vec.push_back(read_u8_mmap(ptr));
                    return vec;
                }
                case GGUF_TYPE_INT8: {
                    std::vector<int8_t> vec;
                    vec.reserve(array_len);
                    for (uint64_t i = 0; i < array_len; ++i) vec.push_back(read_i8_mmap(ptr));
                    return vec;
                }
                case GGUF_TYPE_UINT16: {
                    std::vector<uint16_t> vec;
                    vec.reserve(array_len);
                    for (uint64_t i = 0; i < array_len; ++i) vec.push_back(read_u16_mmap(ptr));
                    return vec;
                }
                case GGUF_TYPE_INT16: {
                    std::vector<int16_t> vec;
                    vec.reserve(array_len);
                    for (uint64_t i = 0; i < array_len; ++i) vec.push_back(read_i16_mmap(ptr));
                    return vec;
                }
                case GGUF_TYPE_UINT32: {
                    std::vector<uint32_t> vec;
                    vec.reserve(array_len);
                    for (uint64_t i = 0; i < array_len; ++i) vec.push_back(read_u32_mmap(ptr));
                    return vec;
                }
                case GGUF_TYPE_INT32: {
                    std::vector<int32_t> vec;
                    vec.reserve(array_len);
                    for (uint64_t i = 0; i < array_len; ++i) vec.push_back(read_i32_mmap(ptr));
                    return vec;
                }
                case GGUF_TYPE_FLOAT32: {
                    std::vector<float> vec;
                    vec.reserve(array_len);
                    for (uint64_t i = 0; i < array_len; ++i) vec.push_back(read_f32_mmap(ptr));
                    return vec;
                }
                case GGUF_TYPE_BOOL: {
                    std::vector<bool> vec;
                    vec.reserve(array_len);
                    for (uint64_t i = 0; i < array_len; ++i) vec.push_back(read_bool_mmap(ptr));
                    return vec;
                }
                case GGUF_TYPE_STRING: {
                    std::vector<std::string> vec;
                    vec.reserve(array_len);
                    for (uint64_t i = 0; i < array_len; ++i) vec.push_back(read_string_mmap(ptr));
                    return vec;
                }
                case GGUF_TYPE_UINT64: {
                    std::vector<uint64_t> vec;
                    vec.reserve(array_len);
                    for (uint64_t i = 0; i < array_len; ++i) vec.push_back(read_u64_mmap(ptr));
                    return vec;
                }
                case GGUF_TYPE_INT64: {
                    std::vector<int64_t> vec;
                    vec.reserve(array_len);
                    for (uint64_t i = 0; i < array_len; ++i) vec.push_back(read_i64_mmap(ptr));
                    return vec;
                }
                case GGUF_TYPE_FLOAT64: {
                    std::vector<double> vec;
                    vec.reserve(array_len);
                    for (uint64_t i = 0; i < array_len; ++i) vec.push_back(read_f64_mmap(ptr));
                    return vec;
                }
                default:
                    std::cerr << "Error: Unknown GGUF array metadata type encountered: " << array_type << std::endl;
                    return GGUFMetadataValue(); // Return an empty variant
            }
        }
        default:
            std::cerr << "Error: Unknown GGUF metadata type encountered: " << type << std::endl;
            return GGUFMetadataValue(); // Return an empty variant
    }
}

GGUFReader::GGUFReader() : m_file_descriptor(-1), m_file_size(0), m_file_data(nullptr), m_mapped_size(0) {}

GGUFReader::~GGUFReader() {
    if (m_file_data != nullptr) {
        if (munmap(m_file_data, m_mapped_size) == -1) {
            std::cerr << "Error: Failed to munmap memory." << std::endl;
        }
        m_file_data = nullptr;
    }
    if (m_file_descriptor != -1) {
        if (close(m_file_descriptor) == -1) {
            std::cerr << "Error: Failed to close file descriptor." << std::endl;
        }
        m_file_descriptor = -1;
    }
}

bool GGUFReader::load_from_file(const std::string& filepath) {
    m_file_descriptor = open(filepath.c_str(), O_RDONLY);
    if (m_file_descriptor == -1) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return false;
    }

    struct stat sb;
    if (fstat(m_file_descriptor, &sb) == -1) {
        std::cerr << "Error: Could not get file size for " << filepath << std::endl;
        close(m_file_descriptor);
        m_file_descriptor = -1;
        return false;
    }
    m_file_size = sb.st_size;

    // Ensure the mapped size is a multiple of the page size, or simply map the entire file.
    // For simplicity, we map the entire file.
    m_mapped_size = m_file_size;
    m_file_data = static_cast<uint8_t*>(mmap(NULL, m_mapped_size, PROT_READ, MAP_PRIVATE, m_file_descriptor, 0));
    if (m_file_data == MAP_FAILED) {
        std::cerr << "Error: Could not memory map file " << filepath << std::endl;
        close(m_file_descriptor);
        m_file_descriptor = -1;
        m_file_data = nullptr;
        return false;
    }

    // A pointer that will advance through the mapped memory
    const uint8_t* current_ptr = m_file_data;

    // Read header
    std::memcpy(&m_header.magic, current_ptr, sizeof(m_header.magic));
    current_ptr += sizeof(m_header.magic);
    std::memcpy(&m_header.version, current_ptr, sizeof(m_header.version));
    current_ptr += sizeof(m_header.version);
    m_header.tensor_count = read_u64_mmap(current_ptr);
    m_header.metadata_kv_count = read_u64_mmap(current_ptr);

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
        std::string key = read_string_mmap(current_ptr);
        gguf_type type = read_gguf_type_mmap(current_ptr);
        m_metadata[key] = read_gguf_value_mmap(current_ptr, type);
    }

    // Read tensor info
    for (uint64_t i = 0; i < m_header.tensor_count; ++i) {
        gguf_tensor_info tensor_info;
        tensor_info.name = read_string_mmap(current_ptr);
        std::memcpy(&tensor_info.n_dims, current_ptr, sizeof(tensor_info.n_dims));
        current_ptr += sizeof(tensor_info.n_dims);
        for (uint32_t j = 0; j < tensor_info.n_dims; ++j) {
            tensor_info.ne[j] = read_u64_mmap(current_ptr);
        }
        tensor_info.type = read_gguf_type_mmap(current_ptr);
        tensor_info.offset = read_u64_mmap(current_ptr); // Offset to tensor data

        m_tensor_infos.push_back(tensor_info);
    }

    // Store the current file position as the start of tensor data
    m_tensor_data_offset = current_ptr - m_file_data;

    return true;
}

} // namespace gguf