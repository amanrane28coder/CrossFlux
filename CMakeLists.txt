cmake_minimum_required(VERSION 3.14)
project(HFTTrader LANGUAGES CXX)

# Prefer Boost's package config when supported by the installed CMake.
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# C++20 for std::span (obi_config.hpp / signals.hpp weight views) and the
# [[likely]]/[[unlikely]] branch hints in predictor.hpp. Both subdirectories set
# this themselves; it is repeated here so a target added directly to this file
# does not silently get 17.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add subdirectories
add_subdirectory(cpp_engine)
add_subdirectory(live_trading)