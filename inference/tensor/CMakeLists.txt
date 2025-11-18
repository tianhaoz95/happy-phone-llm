cmake_minimum_required(VERSION 3.10)

project(happy_phone_llm_tensor CXX)

add_library(happy_phone_llm_tensor STATIC
    tensor.cpp
)

target_include_directories(happy_phone_llm_tensor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
