cmake_minimum_required(VERSION 3.14)
project(CrossVenueArbitrage)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# C++20 is required, not preferred: signals.hpp and obi_config.hpp use std::span
# for the non-owning weight view, and predictor.hpp uses [[likely]]/[[unlikely]].
# This was 17 and did not compile -- `std::span has not been declared`.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Discover Boost from the toolchain/package manager, not a machine-specific path.
find_package(Boost REQUIRED)
find_package(Threads REQUIRED)

# Find Python headers and libraries
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Find nlohmann_json
find_package(nlohmann_json 3.2.0 REQUIRED)

# Find OpenSSL (required for Beast WebSocket TLS)
find_package(OpenSSL REQUIRED)

# Locate Pybind11 (installed via your environment)
find_package(pybind11 REQUIRED)

# Create a shared library from the source files (for linking with executable)
add_library(arbitrage_engine_lib SHARED
    src/predictor.cpp
    src/models.cpp
    src/bindings.cpp
    src/ingestion_engine.cpp
    src/execution_manager.cpp
    src/feed_handler.cpp
)

# Ensure header search paths are included
target_include_directories(arbitrage_engine_lib PRIVATE
    include
    ${Python3_INCLUDE_DIRS}
    ${NLOHMANN_JSON_INCLUDE_DIR}
)

# Link imported targets so include and link requirements travel with the targets.
target_link_libraries(arbitrage_engine_lib PRIVATE
    Boost::boost
    nlohmann_json::nlohmann_json
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
    ${Python3_LIBRARIES}
)

# Create a Python module version of the same code
pybind11_add_module(arbitrage_engine
    src/predictor.cpp
    src/models.cpp
    src/bindings.cpp
)

# Ensure header search paths are included for the Python module too
target_include_directories(arbitrage_engine PRIVATE
    include
    ${Python3_INCLUDE_DIRS}
    ${NLOHMANN_JSON_INCLUDE_DIR}
)

# Link imported dependencies for the Python module.
target_link_libraries(arbitrage_engine PRIVATE
    Boost::boost
    nlohmann_json::nlohmann_json
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
    ${Python3_LIBRARIES}
)

# Install .so next to the Python source for easy import (Python module)
set_target_properties(arbitrage_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

# Also output the regular shared library to the same location
set_target_properties(arbitrage_engine_lib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

# Optimize for speed (-O3).
#
# -ffp-contract=off is deliberate and is about reproducibility, not speed. The
# weighted OBI accumulates `num += w * (b - a)`, which the compiler is otherwise
# free to fuse into an FMA -- one rounding instead of two. That makes the C++
# signal disagree with src/obi_weights.py by ~1 ULP, which was measured: 339 of
# 3756 values differed, all of them on the decay_75 profile (the only preset
# whose weights are not powers of two, so the only one where that multiply
# rounds at all). The gap is ~1e-17 against a 0.3 threshold and cannot change a
# trade decision, but it does mean the backtest and the live engine compute
# different numbers from identical input, and it makes the cross-language parity
# check untestable except through a tolerance. Turning contraction off costs
# nothing measurable on a 5-iteration loop.
#
# Avoid -march=native so binaries remain buildable on other supported CPUs.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(arbitrage_engine_lib PRIVATE -O3 -ffp-contract=off)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(arbitrage_engine PRIVATE -O3 -ffp-contract=off)
endif()