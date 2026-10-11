// The first triangle (steps 08-09): one file, two entry points.
//
// There is no vertex buffer yet (module 8): the three corners live in the shader
// and the vertex shader picks one by its index. vkCmdDraw(3, 1, 0, 0) runs it for
// vertex indices 0, 1, 2; the rasterizer interpolates `color` across the triangle
// and runs the fragment shader once per covered pixel.
//
// Compiled to SPIR-V by `bun run sync:shaders` (scripts/sync-shaders.ts) into
// shaders/spv/triangle.spv; both the Rust and the C++ examples load that file.

// Clip-space positions. Vulkan's clip space has +Y pointing DOWN (unlike OpenGL),
// so y = -0.5 is the top corner. Seen on screen the order top -> bottom-right ->
// bottom-left is clockwise, which the pipeline declares as the front face.
static const float2 positions[3] = {
    float2(0.0, -0.5),
    float2(0.5, 0.5),
    float2(-0.5, 0.5),
};

// Linear RGB. The swapchain uses an *_SRGB format, so the hardware encodes these
// values to sRGB when it writes the pixel (pure red/green/blue look the same
// either way; the interpolated mix in the middle does not).
static const float3 colors[3] = {
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0),
};

// What the vertex stage hands to the rasterizer. SV_Position is the clip-space
// position (SPIR-V `Position` built-in); `color` becomes a Location 0 varying.
struct VSOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

// SV_VulkanVertexID is SPIR-V `VertexIndex` as is. (Slang's SV_VertexID follows the
// D3D meaning, `VertexIndex - BaseVertex`, which needs the DrawParameters
// capability and the shaderDrawParameters feature; here they are equal anyway.)
[shader("vertex")]
VSOutput vertexMain(uint vertexId : SV_VulkanVertexID)
{
    VSOutput output;
    output.position = float4(positions[vertexId], 0.0, 1.0);
    output.color = colors[vertexId];
    return output;
}

// SV_Target is color attachment 0 of the rendering (vkCmdBeginRendering).
[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}
