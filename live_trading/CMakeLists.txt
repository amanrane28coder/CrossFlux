cmake_minimum_required(VERSION 3.14)
project(LiveTrading)

# C++20 is required: live_trading.cpp includes signals.hpp and predictor.hpp,
# which use std::span and [[likely]]/[[unlikely]]. This was 17 and did not build.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Manually specify Boost paths since find_package is problematic
set(BOOST_INCLUDE_DIR /opt/homebrew/opt/boost/include)
set(BOOST_LIBRARY_DIR /opt/homebrew/opt/boost/lib)

# Find nlohmann_json
find_package(nlohmann_json 3.2.0 REQUIRED)

# Find OpenSSL (required for Beast WebSocket TLS)
find_package(OpenSSL REQUIRED)

# Add executable
add_executable(live_trading_binance_kraken
    ../live_trading.cpp
)

# Include directories
# Explicitly set Boost include path since we know where it's installed
target_include_directories(live_trading_binance_kraken PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_engine/include
    ${BOOST_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR}
    ${OPENSSL_INCLUDE_DIR}
)

# Link libraries - manually specify the Boost libraries we need
target_link_libraries(live_trading_binance_kraken PRIVATE
    ${BOOST_LIBRARY_DIR}/libboost_thread.dylib
    ${BOOST_LIBRARY_DIR}/libboost_context.dylib
    ${BOOST_LIBRARY_DIR}/libboost_coroutine.dylib
    ${BOOST_LIBRARY_DIR}/libboost_fiber.dylib
    nlohmann_json::nlohmann_json
    ${OPENSSL_LIBRARIES}
    pthread
    dl
)

# Link with the arbitrage engine library
target_link_libraries(live_trading_binance_kraken PRIVATE
    /Users/amanavinashrane/Projects/libarbitrage_engine_lib.dylib
)

# Set runtime library output directory to find the .so file
set_target_properties(live_trading_binance_kraken PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../
)

# Optimize for speed.
#
# -ffp-contract=off matches cpp_engine/CMakeLists.txt, and matters more here:
# this is the binary whose signals are compared against backtest output. Letting
# the compiler fuse `num += w * (b - a)` into an FMA makes the live engine
# disagree with src/obi_weights.py by ~1 ULP on the decay_75 profile, so the same
# book would produce two different signal values depending on which side of the
# system computed it. See the longer note in cpp_engine/CMakeLists.txt.
target_compile_options(live_trading_binance_kraken PRIVATE -O3 -march=native -ffp-contract=off)