cmake_minimum_required(VERSION 3.10)

project(happy_phone_llm_inference CXX)

add_subdirectory(engine)
add_subdirectory(kernel)
add_subdirectory(gguf)
add_subdirectory(tensor)
add_subdirectory(test)
