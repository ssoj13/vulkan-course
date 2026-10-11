// Steps 14-15: the vertices come from a vertex buffer.
//
// triangle.slang kept its corners in the shader. Here the vertex shader gets
// one vertex per invocation from vertex input: the pipeline's vertex input
// state says where each attribute lives in the bound buffer (binding 0,
// stride 20 bytes: position at offset 0, color at offset 8), and the
// input assembler fetches it. vkCmdDrawIndexed reads the index buffer to pick
// which vertex each invocation gets, so a vertex shared by several triangles
// is stored once.
//
// Compiled to SPIR-V by `bun run sync:shaders` (scripts/sync-shaders.ts) into
// shaders/spv/mesh.spv; both the Rust and the C++ examples load that file.

// One vertex as it lies in the vertex buffer. `vk::location(N)` is the
// SPIR-V Location the pipeline's VkVertexInputAttributeDescription::location
// refers to; the semantics (POSITION, COLOR) mean nothing to Vulkan.
struct VSInput
{
    [[vk::location(0)]] float2 position : POSITION; // R32G32_SFLOAT, clip space
    [[vk::location(1)]] float3 color : COLOR;       // R32G32B32_SFLOAT, linear RGB
};

// What the vertex stage hands to the rasterizer, as in triangle.slang.
struct VSOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

// SV_Target is color attachment 0 of the rendering (vkCmdBeginRendering).
[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}
