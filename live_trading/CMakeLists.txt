cmake_minimum_required(VERSION 3.14)
project(LiveTrading)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# C++20 is required: live_trading.cpp includes signals.hpp and predictor.hpp,
# which use std::span and [[likely]]/[[unlikely]]. This was 17 and did not build.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Discover Boost from the toolchain/package manager, not a machine-specific path.
find_package(Boost REQUIRED)
find_package(Threads REQUIRED)

# Find nlohmann_json
find_package(nlohmann_json 3.2.0 REQUIRED)

# Find OpenSSL (required for Beast WebSocket TLS)
find_package(OpenSSL REQUIRED)

# Add executable
add_executable(live_trading_binance_kraken
    ../live_trading.cpp
)

# Include directories
target_include_directories(live_trading_binance_kraken PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_engine/include
    ${NLOHMANN_JSON_INCLUDE_DIR}
)

# Link imported targets so include and link requirements travel with the targets.
target_link_libraries(live_trading_binance_kraken PRIVATE
    Boost::boost
    nlohmann_json::nlohmann_json
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
)

# Link with the arbitrage engine library
target_link_libraries(live_trading_binance_kraken PRIVATE
    Boost::boost
    arbitrage_engine_lib
)

# Set runtime library output directory to find the .so file
set_target_properties(live_trading_binance_kraken PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
)

# Optimize for speed.
#
# -ffp-contract=off matches cpp_engine/CMakeLists.txt, and matters more here:
# this is the binary whose signals are compared against backtest output. Letting
# the compiler fuse `num += w * (b - a)` into an FMA makes the live engine
# disagree with src/obi_weights.py by ~1 ULP on the decay_75 profile, so the same
# book would produce two different signal values depending on which side of the
# system computed it. See the longer note in cpp_engine/CMakeLists.txt.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(live_trading_binance_kraken PRIVATE -O3 -ffp-contract=off)
endif()