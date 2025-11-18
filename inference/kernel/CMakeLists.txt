cmake_minimum_required(VERSION 3.10)

project(happy_phone_llm_kernel CXX)

add_library(happy_phone_llm_kernel STATIC
    kernel.cpp
    matrix_multiplication.cpp
)

target_include_directories(happy_phone_llm_kernel PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR} # Add the root inference directory for tensor includes
)

target_link_libraries(happy_phone_llm_kernel PUBLIC
    happy_phone_llm_tensor
)
