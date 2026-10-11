# Module 12 "advanced" — plan (2026-10-11)

## Decisions (orchestrator)
1. Ray tracing = two lessons: advanced/ray-query (33), advanced/ray-tracing-pipeline (34).
2. Debugging = two programs: 28_debug_labels, 29_gpu_queries (one lesson debugging-and-profiling, or two if it gets long).
3. Negative demos stay "temporary edit, reverted" (course practice), no --bug flags.
4. Capstone model: commit MetalRoughSpheresNoTextures.glb (291 KB, CC0 + LICENSE); `bun run fetch:models` downloads WaterBottle.glb (8.97 MB, CC0) with pinned sha256 into a gitignored dir; `--model` flag. NOT DamagedHelmet (CC-BY-NC part).
5. pipeline_binary: full `--binary` path (Rust via raw fp()).
6. Tools (2026-10-11, user): RenderDoc updated to 1.46 portable in place of 1.39 (same path, the HKLM implicit-layer
   registration still points to ...\ide\renderdoc\renderdoc.json; layer api_version 1.4.324; step 28 now runs under it).
   The debugging lesson links the official downloads: RenderDoc https://renderdoc.org/ (builds:
   https://renderdoc.org/builds), Nsight Systems https://developer.nvidia.com/nsight-systems, Nsight Graphics
   https://developer.nvidia.com/nsight-graphics. Lessons state only the requirement and the reason: RenderDoc 1.46+,
   because the course needs Vulkan 1.4 support in the capture layer. Never mention or compare older tool versions.
   Tools: use installed RenderDoc (portable, C:\Programs\Ntutil\apps\prog\ide\renderdoc) and Nsight Systems 2026.1.3; Nsight Graphics not installed -> described only. No software downloads without the user.

## Verified facts (RTX 3080 Ti, driver 616.64, SDK 1.4.341.1; vulkaninfoSDK --text)
- Extensions present: VK_EXT_mesh_shader, VK_KHR_acceleration_structure (rev 13), VK_KHR_ray_query, VK_KHR_ray_tracing_pipeline, ray_tracing_maintenance1, ray_tracing_position_fetch, deferred_host_operations, VK_KHR_pipeline_binary, calibrated_timestamps (KHR+EXT), pipeline_executable_properties, pipeline_creation_feedback/cache_control, graphics_pipeline_library, shader_object, device_fault, NV device_diagnostic_checkpoints, shader_non_semantic_info, maintenance1-11. Absent: VK_KHR_performance_query. Instance: VK_EXT_debug_utils.
- gpu.json has no extension list (features/limits only).
- Features: pipelineCreationCacheControl, pipelineStatisticsQuery, inheritedQueries, hostQueryReset true; taskShader/meshShader/multiviewMeshShader/meshShaderQueries true; accelerationStructure + CaptureReplay true, HostCommands/IndirectBuild false; rayQuery, rayTracingPipeline, TraceRaysIndirect, rayTraversalPrimitiveCulling true; pipelineBinaries true.
- Mesh: max task/mesh WG invocations 128, preferred 32; maxMeshOutputVertices/Primitives 256; maxTaskPayloadSize 16384; prefersCompactPrimitiveOutput true.
- RT pipeline: shaderGroupHandleSize 32, HandleAlignment 32, BaseAlignment 64, maxRayRecursionDepth 31, maxRayHitAttributeSize 32. AS: minAccelerationStructureScratchOffsetAlignment 128.
- Pipeline binary: internalCache true, internalCacheControl true (VkDevicePipelineBinaryInternalCacheControlKHR disableInternalCache allowed, VUID-...-09602), PrefersInternalCache false, CompressedData true. pipelineCacheUUID becd218b-ed85-b945-fd53-2c2a423d6fe9. NVIDIA disk cache %LOCALAPPDATA%\NVIDIA\DXCache (whether disableInternalCache covers it: measure).
- timestampPeriod 1; timestampValidBits 64 on families 0,1,2,5; 32 on 3,4. Calibrateable domains: query at runtime.
- Validation settings: printf_enable (default false), printf_to_stdout, printf_buffer_size 1024; gpuav_enable etc.
- ash f4c2ca3: wrappers ext::mesh_shader, khr::acceleration_structure, khr::ray_tracing_pipeline, khr::deferred_host_operations, khr::calibrated_timestamps, khr::ray_tracing_maintenance1, khr::pipeline_executable_properties, ext::debug_utils. pipeline_binary: only generated ash::khr::pipeline_binary::Device (raw fp()). Types present: AccelerationStructureInstanceKHR, Packed24_8, DevicePipelineBinaryInternalCacheControlKHR, CommandBufferInheritanceRenderingInfo. Vulkan-Hpp raii has all.
- Ext deps (vk.xml): acceleration_structure needs 1.2 + deferred_host_operations; ray_query / ray_tracing_pipeline need acceleration_structure; mesh_shader 1.2; pipeline_binary 1.4 (+maintenance5).
- Slang 2026.1: atoms mesh/amplification/raytracing/rayquery/printf exist; NOT compile-tested yet (do it first, see below).
- Rust: gltf 1.4.1 (2024-05-10; features utils; import pulls image), png 0.18.1, image 0.25.10, zune-jpeg 0.5.15. vcpkg: cgltf 1.15, fastgltf 0.9.0, tinygltf 3.0.0, stb 2024-07-29, glm 1.0.3, ktx 4.4.2.

## Tools section (user request 2026-10-11): "how to use them" lessons
A separate group of lessons (own module `tools` after advanced, or inside advanced — decide when writing), each
built on real captures/measurements of course steps on the dev machine:
- tools/renderdoc: capture (UI + renderdoccmd + in-app API), Event Browser with our debug labels (step 28), Pipeline
  State, resource/texture viewer (depth, mips of 21-24), mesh viewer, shader debugging (step-through a pixel),
  what it hides (extension whitelist).
- tools/nsight-systems: CPU/GPU timeline of frames in flight (10, 27 async compute), vulkan-annotations, queue
  overlap, present waits; reading the report.
- tools/nsight-graphics: frame debugger + GPU Trace (unit throughput, SM occupancy, warp stalls) + Shader Profiler
  (per-line hotspots). NOT installed yet: ask the user before downloading (may need an NVIDIA developer login).
  Nsight Compute = CUDA only (NVIDIA forum: Vulkan/GL compute -> Nsight Graphics); the lesson says so.
- tools/occupancy-and-registers: theory (warp scheduler, register file per SM, occupancy = resident warps / max;
  limits from registers, shared memory, workgroup size; GA102 numbers from NVIDIA docs, sourced), register spilling
  (local memory traffic). Measure: VK_KHR_pipeline_executable_properties (present on this GPU) with
  CAPTURE_STATISTICS/INTERNAL_REPRESENTATIONS on course pipelines -> print what the NVIDIA driver actually reports
  (statistic names unknown until probed); a kernel with rising register pressure (e.g. unrolled local arrays) to
  show registers/occupancy/spills vs measured time; confirm in Nsight Graphics if installed. Trainer idea: pick
  workgroup size / registers / shared memory -> computed occupancy (rules from vendor docs, labeled).
- tools/validation-and-capture: vkconfig (layer settings instead of env vars), GPU-AV, printf, sync validation +
  heuristic, GFXReconstruct capture/replay (in the SDK), crash diagnostics (VK_EXT_device_fault, NV checkpoints,
  crash_diagnostic layer).
- Mention, not teach: AMD RGP/RGA, Intel GPA, Tracy (CPU+GPU zones via timestamps).

## Examples 28-35
- 28_debug_labels (window, scene of 22): object names on every object (+ a validation message quoting a name, temp edit); cmd/queue labels nested per pass; balanced labels (VUID-vkEndCommandBuffer-commandBuffer-01815, VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-01912); `--printf`: Slang printf in fragment for one pixel with VK_KHRONOS_VALIDATION_PRINTF_ENABLE=true (quote real output); GPU-AV run; lesson: RenderDoc (ENABLE_VULKAN_RENDERDOC_CAPTURE=1, renderdoccmd capture, renderdoc_app.h), nsys profile --trace=vulkan; check what extensions RenderDoc 1.39 hides.
- 29_gpu_queries (window): timestamps per pass; pipeline statistics (IA verts/prims, VS inv, clipping, FS inv) vs computed expectations (measure); host query reset; calibrated timestamps (print domains, DEVICE+QUERY_PERFORMANCE_COUNTER, maxDeviation) -> CPU submit -> GPU start latency. --frames --seconds --no-stats.
- 30_pipeline_cache (no window): N variants via specialization constants of a heavy fragment (capstone PBR), cold / in-memory cache / file cache in new process; VkPipelineCreationFeedback per pipeline; --probe FAIL_ON_PIPELINE_COMPILE_REQUIRED -> VK_PIPELINE_COMPILE_REQUIRED; header check (32 bytes, version 1, vendor 0x10DE, device id, UUID; VUIDs 04967, initialDataSize-00768/00769), corrupted header/payload experiments; --no-driver-cache via pipeline_binary internal-cache control; --binary capture/reload via pipeline_binary. Timings with validation OFF, medians.
- 31_multithreaded (window): ~20k cubes, persistent workers + barrier (std::thread+Barrier / jthread+std::barrier); pool per (thread x frame slot) TRANSIENT + vkResetCommandPool; --mode secondary (RENDER_PASS_CONTINUE + CommandBufferInheritanceRenderingInfo, VUID-VkCommandBufferBeginInfo-flags-06002; primary BeginRendering with CONTENTS_SECONDARY_COMMAND_BUFFERS, VUID-vkCmdExecuteCommands-flags-06024; no state inheritance) / --mode resume (suspend/resume primaries); --threads 1..16; measure CPU record time + GPU time.
- 32_mesh_shader (window): procedural surface in 7x7-quad meshlets (64 v / 98 t), --pipeline vertex|mesh|task; task shader frustum + normal-cone culling, payload, DispatchMesh; WG size 32; queries MESH_PRIMITIVES_GENERATED_EXT + TASK/MESH invocations; VUIDs vkCmdDrawMeshTasksEXT 07322..07329, RuntimeSpirv-MeshEXT-07115. Shader file name meshlet.slang (mesh.slang is taken).
- 33_ray_query (window): BLAS cube+floor, TLAS; build sizes; scratch aligned 128 (VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03710); AS offset multiple of 256 (VUID-VkAccelerationStructureCreateInfoKHR-offset-03734); barrier AS_BUILD/AS_WRITE -> FRAGMENT/AS_READ; compaction (COMPACTED_SIZE query + copy, sizes printed); build timing; RayQuery shadows in fragment, --no-shadows.
- 34_ray_tracing_pipeline (window): raygen -> storage image -> blit; misses (sky, shadow), closest hit with shadow ray, recursion 2; SBT layout printed, raygen size==stride (VUID-vkCmdTraceRaysKHR-size-04023), base aligned 64 (-03682), stride multiple of handleAlignment (-03686/-03690); SHADER_BINDING_TABLE usage; stack size query.
- 35_capstone (multi-file: rust src/bin/35_capstone/{main,gltf,...}.rs; cpp src/35_capstone/{main.cpp,gltf.*}; Step.astro file="35_capstone/main" — verify; CMake special case): frame loop 17/22/24, staging 15, texture+mips 21/23, depth 22, MSAA 24, bindless textures (material index from push constant = dynamically uniform), materials SSBO; glTF metallic-roughness BRDF (GGX D, height-correlated Smith V, Schlick F), directional light + ambient, SRGB for baseColor/emissive, UNORM others, tangents or derivative TBN, tone map; Rust gltf =1.4.1 (default-features off, utils) + png =0.18.1; C++ cgltf + stb_image via vcpkg; PNG only (identical decode -> FNV hash in stdout).

## First step before 32-34
Compile-test tiny Slang shaders (mesh+task, ray query, RT pipeline raygen/miss/chit, printf) with slangc -> spirv-val vulkan1.4 -> spirv-dis capabilities; look for codegen bugs (payload storage class, PerPrimitiveEXT, printf under -O2). Add new .spv to REQUIRED in tests/shaders.test.ts.

## Order
28 -> 29 -> 30 -> 31 -> 32 -> 33 -> 34 -> 35. Lessons after each pair's code review.

## Risks
RT: many new objects, syncval coverage of AS accesses uncertain. Mesh: Slang untested. RenderDoc 1.39 may hide 1.4 features. NVIDIA disk cache contaminates cold numbers. MT speedup may be small (report as measured). gltf 1.4.1 not built yet against this toolchain (scratch try under the lock). New deps (gltf, png; vcpkg cgltf, stb) are additions to examples/rust/Cargo.toml and examples/cpp/vcpkg.json (this repo owns them).
