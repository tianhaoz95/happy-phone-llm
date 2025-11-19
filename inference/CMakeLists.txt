cmake_minimum_required(VERSION 3.10)
project(happy_phone_llm_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(engine)
add_subdirectory(kernel)
add_subdirectory(gguf)
add_subdirectory(tensor)
add_subdirectory(test)
add_subdirectory(cli)
