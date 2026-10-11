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

// Slang 2026.1 (SDK 1.4.341.1) declares NonUniformResourceIndex with a broad
// capability set and warns (41012) that it widened the spirv_1_6 profile. The
// SPIR-V it emits needs only RuntimeDescriptorArray + ShaderNonUniform
// (checked with spirv-dis), so the warning is silenced for this file.
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

[shader("vertex")]
VSOutput vertexMain(VSInput input, uint instance : SV_VulkanInstanceID)
{
    // Vertices of different instances can run in the same subgroup (warp), so
    // the index is not the same for all of them: NonUniformResourceIndex
    // decorates the access NonUniform. Without it, a GPU that keeps
    // descriptors in scalar (per-subgroup) registers may use one lane's
    // descriptor for all lanes.
    Instance inst = objects[NonUniformResourceIndex(instance)][0];
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
