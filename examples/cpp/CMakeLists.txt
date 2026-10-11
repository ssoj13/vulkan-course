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

# VMA (Vulkan Memory Allocator, step 16) also comes from vcpkg: header-only, 3.4.0.
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

# Steps 08+ read the SPIR-V compiled from ../../shaders/*.slang (bun run sync:shaders)
# at run time; the absolute path of shaders/spv is baked in at build time (as
# SHADER_DIR, defined for every step: the ones without shaders ignore it).
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
    12_timeline
    13_memory_types
    14_vertex_buffer
    15_staging
    16_allocator
    17_uniforms
    18_push_constants
    19_buffer_device_address
    20_bindless
    21_texture
    22_depth
    23_mipmaps
    24_msaa
    25_compute
    26_particles
    27_async_compute)
# Steps that open a window.
set(WINDOW_STEPS 06_window_surface 07_swapchain 08_pipeline 09_triangle
                 10_frames_in_flight 11_resize 12_timeline 14_vertex_buffer
                 15_staging 17_uniforms 18_push_constants
                 19_buffer_device_address 20_bindless
                 21_texture 22_depth 23_mipmaps 24_msaa
                 26_particles 27_async_compute)

foreach(step IN LISTS STEPS)
    add_executable(${step} src/${step}.cpp)
    target_link_libraries(${step} PRIVATE Vulkan::Vulkan)
    if(step STREQUAL "16_allocator")
        # Adds .../include/vma: the code includes <vk_mem_alloc.h>. (The Vulkan SDK
        # ships an older VMA as <vma/vk_mem_alloc.h>, which we do not use.)
        target_link_libraries(${step} PRIVATE GPUOpen::VulkanMemoryAllocator)
    endif()
    target_compile_definitions(${step} PRIVATE SHADER_DIR="${shader_dir}")
    if(step IN_LIST WINDOW_STEPS)
        target_link_libraries(${step} PRIVATE glfw)
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
