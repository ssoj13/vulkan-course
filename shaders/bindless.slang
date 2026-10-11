// Step 20: "bindless" - one big descriptor array, indexed in the shader.
//
// Instead of binding a different descriptor set per object, the program binds
// ONE set whose binding 0 is an array of storage buffer descriptors, and the
// shader picks an element by index. Here the index is the instance number, so
// every instance of the single draw reads its own buffer. Textures work the
// same way (an array of sampled images); they come in module 10.
//
// Descriptor indexing (core 1.2) features this needs:
//   runtimeDescriptorArray                   - `objects[]` has no size in the shader
//   shaderStorageBufferArrayNonUniformIndexing - the index may differ between
//                                              invocations of one draw (see below)
// and, on the API side, descriptorBindingPartiallyBound,
// descriptorBindingVariableDescriptorCount,
// descriptorBindingStorageBufferUpdateAfterBind and
// descriptorBindingUpdateUnusedWhilePending (see 20_bindless).
//
// Compiled by `bun run sync:shaders` into shaders/spv/bindless.spv.

// Slang's core module declares NonUniformResourceIndex with the whole HLSL
// shader-model capability set (image queries, derivatives, sparse residency,
// ...), so slangc warns (41012) that it widened the spirv_1_6 profile with
// them. The widening is the compiler's own bookkeeping: none of those
// capabilities reaches the SPIR-V (checked with spirv-dis; Slang 2026.1 from
// SDK 1.4.341.1 and Slang 2026.19 both warn and both emit none of them), so
// the warning is silenced for this file. The capabilities the module DOES
// declare are pinned in tests/shaders.test.ts.
#pragma warning(disable: 41012)

// One hexagon instance: 32 bytes, std430 (StructuredBuffer): offset 0,
// scale 8, phase 12, color 16, spin 28. The same struct as step 19.
struct Instance
{
    float2 offset; // center, in units where the window is 2 tall
    float scale;   // multiplies the mesh corners
    float phase;   // starting angle, radians
    float3 color;  // multiplies the vertex colors
    float spin;    // radians per second, the sign is the direction
};

// The push constant block: 8 bytes.
struct PushData
{
    float time;   // seconds since start
    float aspect; // window height / width: keeps hexagons regular
};

[[vk::push_constant]]
ConstantBuffer<PushData> push;

// Set 0, binding 0: an unsized array of storage buffers, each holding one
// Instance. Its real length is the descriptor set's variable descriptor count.
[[vk::binding(0, 0)]]
StructuredBuffer<Instance> objects[];

// Vertex input exactly as in mesh.slang (binding 0, stride 20).
struct VSInput
{
    [[vk::location(0)]] float2 position : POSITION;
    [[vk::location(1)]] float3 color : COLOR;
};

struct VSOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

// Reads element 0 of buffer `index` of the array, where `index` may differ
// between the invocations of one draw.
Instance loadInstance(uint index)
{
    // A module may index an array of storage buffers non-uniformly only if it
    // declares the StorageBufferArrayNonUniformIndexing capability (allowed by
    // the shaderStorageBufferArrayNonUniformIndexing feature); without it the
    // index must be dynamically uniform
    // (VUID-RuntimeSpirv-StorageBufferArrayNonUniformIndexing-10136).
    // Slang 2026.1 neither emits it nor has a capability atom for it, so it is
    // stated here in inline SPIR-V; slangc moves OpCapability to the module's
    // capability list.
    spirv_asm { OpCapability StorageBufferArrayNonUniformIndexing; };
    // NonUniformResourceIndex makes slangc decorate the access chain that
    // selects the descriptor NonUniform. Vulkan wants the decoration on the
    // pointer the load itself uses (VUID-RuntimeSpirv-subgroupSize-10149),
    // and here that is a second access chain derived from the first (into the
    // buffer's element 0). Slang 2026.1 leaves it undecorated, so
    // `bun run sync:shaders` decorates every pointer derived from a NonUniform
    // one (decorateNonUniform in scripts/sync-shaders.ts).
    return objects[NonUniformResourceIndex(index)][0];
}

[shader("vertex")]
VSOutput vertexMain(VSInput input, uint instance : SV_VulkanInstanceID)
{
    // Vertices of different instances can run in the same subgroup (warp), so
    // the index is not the same for all of them. Without the NonUniform
    // decoration, a GPU that keeps descriptors in scalar (per-subgroup)
    // registers may use one lane's descriptor for all lanes.
    Instance inst = loadInstance(instance);
    float a = inst.phase + push.time * inst.spin;
    float2 p = input.position * inst.scale;
    p = float2(p.x * cos(a) - p.y * sin(a), p.x * sin(a) + p.y * cos(a)) + inst.offset;
    VSOutput output;
    output.position = float4(p.x * push.aspect, p.y, 0.0, 1.0);
    output.color = input.color * inst.color;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}
