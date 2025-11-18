cmake_minimum_required(VERSION 3.10)

project(happy_phone_llm_cli CXX)

add_executable(happy_phone_llm_cli main.cpp)

target_include_directories(happy_phone_llm_cli PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/.. # For engine/engine.h
    ${CMAKE_SOURCE_DIR}/tensor # For tensor.h
    ${CMAKE_SOURCE_DIR}/gguf # For gguf.h
)

target_link_libraries(happy_phone_llm_cli PUBLIC
    happy_phone_llm_engine
)
