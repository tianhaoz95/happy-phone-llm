cmake_minimum_required(VERSION 3.10)

project(happy_phone_llm_engine CXX)

add_library(happy_phone_llm_engine SHARED
    engine.cpp
    capi.cpp
)

target_include_directories(happy_phone_llm_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR} # For capi.h
)

target_link_libraries(happy_phone_llm_engine PUBLIC
    happy_phone_llm_gguf
    happy_phone_llm_tensor
)
