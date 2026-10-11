// Steps 22 and 24: two interpenetrating cubes.
//
// One cube mesh, drawn as two INSTANCES (instanceCount 2): instance i takes
// its model-view-projection matrix from mvp[i]. Where the cubes cut through
// each other, only a depth test decides which surface is in front; draw order
// cannot, because each cube is partly in front of the other.
//
// SV_VulkanInstanceID = InstanceIndex, which counts from firstInstance (0
// here). SV_InstanceID would be InstanceIndex - BaseInstance and needs the
// DrawParameters capability (see bindless.slang).
//
// Compiled by `bun run sync:shaders` into shaders/spv/cube.spv.

struct Scene
{
    float4x4 mvp[2]; // offsets 0 and 64 (std140 array stride 64), column-major
};

// Set 0, binding 0: the per-frame matrices (a uniform buffer, step 17).
[[vk::binding(0, 0)]]
ConstantBuffer<Scene> scene;

// Vertex input: binding 0, stride 24 (float3 position + float3 color).
struct VSInput
{
    [[vk::location(0)]] float3 position : POSITION;
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
    VSOutput output;
    output.position = mul(scene.mvp[instance], float4(input.position, 1.0));
    // The second cube gets the same face colors with the channels rotated,
    // so the two are easy to tell apart.
    output.color = instance == 0 ? input.color : input.color.brg;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}
