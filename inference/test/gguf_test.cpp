#include "gguf/gguf.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <numeric>

// Function to create a dummy GGUF file for testing
void create_dummy_gguf_file(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create dummy GGUF file " << filepath << std::endl;
        return;
    }

    gguf::gguf_header header;
    header.magic = gguf::GGUF_MAGIC;
    header.version = gguf::GGUF_VERSION;
    header.tensor_count = 1;
    header.metadata_kv_count = 1;

    file.write(reinterpret_cast<char*>(&header.magic), sizeof(header.magic));
    file.write(reinterpret_cast<char*>(&header.version), sizeof(header.version));
    file.write(reinterpret_cast<char*>(&header.tensor_count), sizeof(header.tensor_count));
    file.write(reinterpret_cast<char*>(&header.metadata_kv_count), sizeof(header.metadata_kv_count));

    // Write metadata
    std::string key = "test.metadata";
    uint64_t key_len = key.length();
    file.write(reinterpret_cast<char*>(&key_len), sizeof(key_len));
    file.write(key.c_str(), key_len);

    uint32_t type = gguf::GGUF_TYPE_STRING;
    file.write(reinterpret_cast<char*>(&type), sizeof(type));

    std::string value = "test_value";
    uint64_t value_len = value.length();
    file.write(reinterpret_cast<char*>(&value_len), sizeof(value_len));
    file.write(value.c_str(), value_len);

    // Write tensor info
    gguf::gguf_tensor_info tensor_info;
    tensor_info.name = "test.tensor";
    tensor_info.n_dims = 2;
    tensor_info.ne[0] = 2;
    tensor_info.ne[1] = 3;
    tensor_info.type = gguf::GGUF_TYPE_FLOAT32;
    tensor_info.offset = 0; // Relative to start of tensor data

    uint64_t name_len = tensor_info.name.length();
    file.write(reinterpret_cast<char*>(&name_len), sizeof(name_len));
    file.write(tensor_info.name.c_str(), name_len);
    file.write(reinterpret_cast<char*>(&tensor_info.n_dims), sizeof(tensor_info.n_dims));
    file.write(reinterpret_cast<char*>(&tensor_info.ne[0]), sizeof(uint64_t) * tensor_info.n_dims);
    file.write(reinterpret_cast<char*>(&tensor_info.type), sizeof(tensor_info.type));
    file.write(reinterpret_cast<char*>(&tensor_info.offset), sizeof(tensor_info.offset));

    // Write dummy tensor data (2x3 float matrix)
    float tensor_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    file.write(reinterpret_cast<char*>(tensor_data), sizeof(tensor_data));

    file.close();
}

void test_gguf_reader() {
    std::cout << "Running test_gguf_reader..." << std::endl;
    const std::string dummy_filepath = "dummy.gguf";
    create_dummy_gguf_file(dummy_filepath);

    gguf::GGUFReader reader;
    assert(reader.load_from_file(dummy_filepath));

    // Check metadata
    const auto& metadata = reader.get_metadata();
    assert(metadata.count("test.metadata") == 1);
    assert(metadata.at("test.metadata") == "test_value");

    // Check tensor info
    const auto& tensor_infos = reader.get_tensor_infos();
    assert(tensor_infos.size() == 1);
    assert(tensor_infos[0].name == "test.tensor");
    assert(tensor_infos[0].n_dims == 2);
    assert(tensor_infos[0].ne[0] == 2);
    assert(tensor_infos[0].ne[1] == 3);
    assert(tensor_infos[0].type == gguf::GGUF_TYPE_FLOAT32);
    assert(tensor_infos[0].offset == 0);

    std::cout << "test_gguf_reader passed." << std::endl;
}

int main() {
    test_gguf_reader();
    std::cout << "All GGUF tests passed!" << std::endl;
    return 0;
}
