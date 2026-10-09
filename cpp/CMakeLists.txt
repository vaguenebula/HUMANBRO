cmake_minimum_required(VERSION 3.16)
project(humanbro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_library(humanbro STATIC
  src/model.cpp
  src/pipeline.cpp
  src/humanizer.cpp
  src/timing.cpp
  src/midi_file.cpp
)
target_include_directories(humanbro PUBLIC include PRIVATE src)
target_link_libraries(humanbro PUBLIC Threads::Threads)

# Features must match the Python pipeline bit for bit: strict IEEE arithmetic,
# no fused multiply-add contraction, no fast-math reassociation.
if(MSVC)
  target_compile_options(humanbro PRIVATE /fp:precise /W4 /permissive-)
else()
  target_compile_options(humanbro PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
endif()

add_executable(humanbro_cli tools/humanbro_cli.cpp)
target_link_libraries(humanbro_cli PRIVATE humanbro)
set_target_properties(humanbro_cli PROPERTIES OUTPUT_NAME humanbro)
