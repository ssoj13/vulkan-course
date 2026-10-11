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

# GLFW (windows + Vulkan surface, steps 06+) comes from vcpkg in manifest mode:
# vcpkg.json lists it, the presets point CMAKE_TOOLCHAIN_FILE at vcpkg.
find_package(glfw3 CONFIG REQUIRED)

# Steps 08+ read the SPIR-V compiled from ../../shaders/*.slang (bun run sync:shaders)
# at run time; the absolute path of shaders/spv is baked in at build time.
cmake_path(ABSOLUTE_PATH CMAKE_CURRENT_SOURCE_DIR OUTPUT_VARIABLE src_dir)
cmake_path(APPEND src_dir "../../shaders/spv" OUTPUT_VARIABLE shader_dir)
cmake_path(NORMAL_PATH shader_dir)

# One executable per lesson step; every file is self-contained.
set(STEPS
    01_instance
    02_validation
    03_physical_devices
    04_queue_families
    05_logical_device
    06_window_surface
    07_swapchain
    08_pipeline
    09_triangle
    10_frames_in_flight
    11_resize
    12_timeline)
# Steps that open a window.
set(WINDOW_STEPS 06_window_surface 07_swapchain 08_pipeline 09_triangle
                 10_frames_in_flight 11_resize 12_timeline)

foreach(step IN LISTS STEPS)
    add_executable(${step} src/${step}.cpp)
    target_link_libraries(${step} PRIVATE Vulkan::Vulkan)
    if(step IN_LIST WINDOW_STEPS)
        target_link_libraries(${step} PRIVATE glfw)
        target_compile_definitions(${step} PRIVATE SHADER_DIR="${shader_dir}")
    endif()

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
