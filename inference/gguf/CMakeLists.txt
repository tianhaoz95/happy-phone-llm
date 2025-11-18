cmake_minimum_required(VERSION 3.10)

project(happy_phone_llm_gguf CXX)

add_library(happy_phone_llm_gguf STATIC
    gguf.cpp
)

target_include_directories(happy_phone_llm_gguf PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
