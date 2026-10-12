// Register-pressure series for the pipeline-stats probe (tools/vk-probe, binary pipeline-stats).
//
// Each entry point keeps N floats alive per invocation (N = 8 ... 256): a local array whose loops are fully unrolled,
// so every element has a constant index and can live in a register instead of indexed local memory. ROUNDS passes
// mix each element with its neighbor, and the result sums all of them, so the compiler can neither drop an element
// nor recompute it cheaply instead of keeping it: after the first pass every value depends on the previous pass.
// The driver's statistics (registers, spills) are then read per entry point; timing the same kernels relates them to
// speed. NVIDIA allows at most 255 registers per thread, so the largest N must spill.
//
// d64 is the control: the same 64 floats and the same arithmetic, but indexed with a value known only at run time,
// so the array cannot be split into registers and must live in addressable (local) memory.
//
// Compiled by `bun run sync:pipeline-stats` (scripts/sync-pipeline-stats.ts) into pressure.spv next to this file,
// with the flags of scripts/sync-shaders.ts; the .spv is committed because the probe embeds it with include_bytes!.

static const uint GROUP_SIZE = 256;
static const int ROUNDS = 8;

struct Params
{
    uint count; // number of invocations that do work (the grid is rounded up to whole groups)
};

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0, 0)]] StructuredBuffer<float> seeds;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> result;

// N live floats per invocation. Values stay in [0.25, 0.75] (a * b * 0.5 + 0.25 for a, b in [0, 1]): no overflow,
// no denormals, so the timing measures the arithmetic and the register file, not special cases.
float work<let N : int>(float seed)
{
    float v[N];
    [ForceUnroll]
    for (int k = 0; k < N; k++)
        v[k] = frac(seed + float(k) * 0.618034);
    [loop]
    for (int r = 0; r < ROUNDS; r++)
    {
        [ForceUnroll]
        for (int k = 0; k < N; k++)
            v[k] = v[k] * v[(k + 1) % N] * 0.5 + 0.25;
    }
    float sum = 0.0;
    [ForceUnroll]
    for (int k = 0; k < N; k++)
        sum += v[k];
    return sum;
}

void run<let N : int>(uint i)
{
    if (i >= params.count)
        return;
    result[i] = work<N>(seeds[i]);
}

// Same work as work<N>, but the index depends on the invocation: a run-time index into a local array.
float workDynamic<let N : int>(float seed, uint i)
{
    float v[N];
    [ForceUnroll]
    for (int k = 0; k < N; k++)
        v[k] = frac(seed + float(k) * 0.618034);
    [loop]
    for (int r = 0; r < ROUNDS; r++)
    {
        [loop]
        for (uint k = 0; k < N; k++)
        {
            uint a = (k + i) % N;
            v[a] = v[a] * v[(a + 1) % N] * 0.5 + 0.25;
        }
    }
    float sum = 0.0;
    [ForceUnroll]
    for (int k = 0; k < N; k++)
        sum += v[k];
    return sum;
}

void runDynamic<let N : int>(uint i)
{
    if (i >= params.count)
        return;
    result[i] = workDynamic<N>(seeds[i], i);
}

[shader("compute")] [numthreads(GROUP_SIZE, 1, 1)] void p8(uint3 id : SV_DispatchThreadID) { run<8>(id.x); }
[shader("compute")] [numthreads(GROUP_SIZE, 1, 1)] void p16(uint3 id : SV_DispatchThreadID) { run<16>(id.x); }
[shader("compute")] [numthreads(GROUP_SIZE, 1, 1)] void p32(uint3 id : SV_DispatchThreadID) { run<32>(id.x); }
[shader("compute")] [numthreads(GROUP_SIZE, 1, 1)] void p64(uint3 id : SV_DispatchThreadID) { run<64>(id.x); }
[shader("compute")] [numthreads(GROUP_SIZE, 1, 1)] void p128(uint3 id : SV_DispatchThreadID) { run<128>(id.x); }
[shader("compute")] [numthreads(GROUP_SIZE, 1, 1)] void p192(uint3 id : SV_DispatchThreadID) { run<192>(id.x); }
[shader("compute")] [numthreads(GROUP_SIZE, 1, 1)] void p256(uint3 id : SV_DispatchThreadID) { run<256>(id.x); }
[shader("compute")] [numthreads(GROUP_SIZE, 1, 1)] void d64(uint3 id : SV_DispatchThreadID) { runDynamic<64>(id.x); }
