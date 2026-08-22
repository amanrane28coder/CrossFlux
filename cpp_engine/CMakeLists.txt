cmake_minimum_required(VERSION 3.14)
project(CrossVenueArbitrage)

# C++20 is required, not preferred: signals.hpp and obi_config.hpp use std::span
# for the non-owning weight view, and predictor.hpp uses [[likely]]/[[unlikely]].
# This was 17 and did not compile -- `std::span has not been declared`.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Manually specify Boost paths since find_package is problematic
set(BOOST_INCLUDE_DIR /opt/homebrew/opt/boost/include)
set(BOOST_LIBRARY_DIR /opt/homebrew/opt/boost/lib)

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
    ${BOOST_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${NLOHMANN_JSON_INCLUDE_DIR}
    ${OPENSSL_INCLUDE_DIR}
)

# Link libraries - manually specify the Boost libraries we need
target_link_libraries(arbitrage_engine_lib PRIVATE
    ${BOOST_LIBRARY_DIR}/libboost_thread.dylib
    ${BOOST_LIBRARY_DIR}/libboost_context.dylib
    ${BOOST_LIBRARY_DIR}/libboost_coroutine.dylib
    ${BOOST_LIBRARY_DIR}/libboost_fiber.dylib
    nlohmann_json::nlohmann_json
    ${OPENSSL_LIBRARIES}
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
    ${BOOST_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${NLOHMANN_JSON_INCLUDE_DIR}
    ${OPENSSL_INCLUDE_DIR}
)

# Link libraries for the Python module
target_link_libraries(arbitrage_engine PRIVATE
    ${BOOST_LIBRARY_DIR}/libboost_thread.dylib
    ${BOOST_LIBRARY_DIR}/libboost_context.dylib
    ${BOOST_LIBRARY_DIR}/libboost_coroutine.dylib
    ${BOOST_LIBRARY_DIR}/libboost_fiber.dylib
    nlohmann_json::nlohmann_json
    ${OPENSSL_LIBRARIES}
    ${Python3_LIBRARIES}
)

# Install .so next to the Python source for easy import (Python module)
set_target_properties(arbitrage_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
)

# Also output the regular shared library to the same location
set_target_properties(arbitrage_engine_lib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
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
# Note that -march=native has the same class of problem and is left as-is: a
# binary built here is not guaranteed to reproduce bit-for-bit on a different
# microarchitecture. Remove it if that guarantee is ever needed.
target_compile_options(arbitrage_engine_lib PRIVATE -O3 -march=native -ffp-contract=off)
target_compile_options(arbitrage_engine PRIVATE -O3 -march=native -ffp-contract=off)