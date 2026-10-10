cmake_minimum_required(VERSION 3.25)
project(vulkan_en_cpp LANGUAGES CXX)

# C++23 only because std::print/std::println live in <print> (MSVC: /std:c++latest);
# everything else in the examples is plain C++20.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The Vulkan SDK provides the loader (vulkan-1.lib) and the C++ headers
# (vulkan/vulkan_raii.hpp). VULKAN_SDK must be set by the SDK installer.
find_package(Vulkan REQUIRED)

# One executable per lesson step; every file is self-contained.
set(STEPS
    01_instance
    02_validation
    03_physical_devices
    04_queue_families
    05_logical_device)

foreach(step IN LISTS STEPS)
    add_executable(${step} src/${step}.cpp)
    target_link_libraries(${step} PRIVATE Vulkan::Vulkan)

    # Vulkan-Hpp structs become plain aggregates, so we can write
    # designated initializers: vk::ApplicationInfo{ .pApplicationName = "x" }.
    target_compile_definitions(${step} PRIVATE VULKAN_HPP_NO_STRUCT_CONSTRUCTORS)

    if(MSVC)
        target_compile_options(${step} PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(${step} PRIVATE -Wall -Wextra)
    endif()
endforeach()

# No LTO anywhere: it only slows down a teaching project. Release gets /O2 by default.
