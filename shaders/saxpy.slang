// Step 25: the first compute kernel, SAXPY: result[i] = a * x[i] + y[i].
//
// A compute shader has no vertices, no pixels and no fixed-function stages
// around it: vkCmdDispatch launches a GRID of WORKGROUPS, each workgroup is
// `numthreads` INVOCATIONS (threads), and every invocation runs this function
// once with its own SV_DispatchThreadID = group id * group size + index in the
// group. It is the same SIMT machine that runs vertex and fragment shaders
// (warps of 32 lanes on NVIDIA), driven directly - like a Houdini wrangle that
// runs over points, where each invocation handles element i.
//
// The data reaches the kernel through STORAGE BUFFERS (StructuredBuffer =
// read-only, RWStructuredBuffer = read-write; both are STORAGE_BUFFER
// descriptors, std430 layout, so a float array has a 4-byte stride) and the
// two scalars through push constants.
//
// Compiled by `bun run sync:shaders` into shaders/spv/saxpy.spv.

// Invocations per workgroup. 256 = 8 subgroups of 32 (NVIDIA) or 4 of 64
// (AMD wave64): no partially filled subgroup. Vulkan 1.4 guarantees
// maxComputeWorkGroupInvocations >= 256 and maxComputeWorkGroupSize.x >= 256
// (Vulkan 1.0: 128); the host reads this size back from the SPIR-V
// (OpExecutionMode LocalSize) to compute the number of groups.
static const uint GROUP_SIZE = 256;

// The push constant block: 8 bytes.
struct Params
{
    float a;     // the scale
    uint count;  // number of elements; the grid is rounded up to whole groups
};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0, 0)]] StructuredBuffer<float> x;
[[vk::binding(1, 0)]] StructuredBuffer<float> y;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> result;

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void saxpy(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    // The grid has whole groups, so the last group may run past the end of
    // the arrays: those invocations must not touch them. No robustness feature
    // (robustBufferAccess and friends) is enabled here, so nothing makes an
    // out-of-bounds storage buffer access harmless.
    if (i >= params.count)
        return;
    result[i] = params.a * x[i] + y[i];
}
