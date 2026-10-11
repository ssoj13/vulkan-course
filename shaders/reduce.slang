// Step 25: a workgroup reduction - every workgroup sums its 256 values.
//
// saxpy.slang let each invocation work alone. Here the invocations of one
// workgroup cooperate through GROUPSHARED memory (SPIR-V Workgroup storage):
// on-chip memory shared by the workgroup only (NVIDIA: shared memory, the same
// SRAM as the L1 cache), invisible to other workgroups. The sum is a tree:
// 256 values -> 128 partial sums -> 64 -> ... -> 1, eight rounds instead of
// 255 sequential additions. Between rounds every invocation must see what the
// others wrote: GroupMemoryBarrierWithGroupSync() is OpControlBarrier with
// Workgroup scope - a workgroup-wide "wait until all arrive" plus a memory
// barrier on workgroup memory. There is no such barrier between workgroups;
// cooperation across the whole grid needs a second dispatch (or atomics).
//
// Integers keep the result exact, so the CPU can check it with ==.
//
// Compiled by `bun run sync:shaders` into shaders/spv/reduce.spv.

// One invocation per value; must equal the size of `partial` below.
static const uint GROUP_SIZE = 256;

// The push constant block: 4 bytes.
struct Params
{
    uint count; // number of values
};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0, 0)]] StructuredBuffer<uint> values;
// One sum per workgroup: groupSums[g] = values[256 g] + ... + values[256 g + 255].
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> groupSums;

// 256 x 4 = 1024 bytes of workgroup memory (limit: maxComputeSharedMemorySize,
// at least 16384 bytes on every Vulkan device).
groupshared uint partial[GROUP_SIZE];

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void reduce(uint3 id : SV_DispatchThreadID, uint3 local : SV_GroupThreadID, uint3 group : SV_GroupID)
{
    // Invocations past the end load 0 instead of returning early: every
    // invocation of the workgroup must reach every barrier below (a barrier
    // in control flow that not all of them execute is undefined behavior).
    partial[local.x] = id.x < params.count ? values[id.x] : 0;
    GroupMemoryBarrierWithGroupSync();

    // The loop bound is the same for all invocations, so all of them meet
    // each barrier; only the additions are skipped by the upper half.
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride /= 2)
    {
        if (local.x < stride)
            partial[local.x] += partial[local.x + stride];
        GroupMemoryBarrierWithGroupSync();
    }

    if (local.x == 0)
        groupSums[group.x] = partial[0];
}
