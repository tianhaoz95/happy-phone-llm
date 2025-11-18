cmake_minimum_required(VERSION 3.10)

project(happy_phone_llm_tests CXX)

add_executable(tensor_test tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE happy_phone_llm_tensor)

add_executable(gguf_test gguf_test.cpp)
target_link_libraries(gguf_test PRIVATE happy_phone_llm_gguf)

# Add include directories for all modules
target_include_directories(tensor_test PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(gguf_test PRIVATE ${CMAKE_SOURCE_DIR})
