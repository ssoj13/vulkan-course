// Steps 21 and 23: a textured surface.
//
// The vertex shader is step 17's transform with a full model-view-projection
// matrix (one matrix = the whole camera, like a DCC viewport's MVP). The new
// part is the fragment shader: it SAMPLES an image through a sampler.
//
//   Sampler2D = image view + sampler in ONE descriptor
//     (VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, set 0, binding 1).
//   .Sample(uv) = implicit-LOD sampling: the GPU derives the mip level from
//     how fast uv changes between neighboring pixels (derivatives of a 2x2
//     pixel quad), filters with the sampler's filters, and, for an _SRGB
//     image format, converts the texels to linear before filtering.
//
// UVs: (0, 0) is the image's first texel row (its top-left corner in memory),
// (1, 1) the opposite corner; values outside [0, 1] are handled by the
// sampler's address modes (repeat, mirror, clamp, border).
//
// Compiled by `bun run sync:shaders` into shaders/spv/textured.spv.

struct Camera
{
    float4x4 mvp; // offset 0, 64 bytes, column-major (step 17)
};

// Set 0, binding 0: the per-frame matrix (a uniform buffer, step 17).
[[vk::binding(0, 0)]]
ConstantBuffer<Camera> camera;

// Set 0, binding 1: the texture with its sampler (one combined descriptor).
[[vk::binding(1, 0)]]
Sampler2D albedo;

// Vertex input: binding 0, stride 20 (float3 position + float2 uv).
struct VSInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
};

struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput output;
    output.position = mul(camera.mvp, float4(input.position, 1.0));
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return albedo.Sample(input.uv);
}
