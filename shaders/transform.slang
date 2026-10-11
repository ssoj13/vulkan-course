// Steps 17-18: one transform block per frame, reaching the shader two ways.
//
// mesh.slang read only vertex attributes. Here every vertex is also multiplied
// by a matrix the CPU changes every frame (a rotation and the window's aspect
// ratio), the way a DCC viewport feeds a model-view-projection matrix to its
// shaders. The data is the same struct, FrameUniforms; only the road differs:
//
//   vertexUniform (step 17, and step 18 --push-descriptor): a UNIFORM BUFFER
//     the descriptor at set 0, binding 0 points to.
//   vertexPush (step 18): PUSH CONSTANTS, bytes recorded straight into the
//     command buffer by vkCmdPushConstants; no buffer, no descriptor.
//
// Layout rules decide where each member lies in memory, and the C++/Rust
// struct must lay its fields out identically:
//   ConstantBuffer (Uniform storage) -> std140 (Slang names the SPIR-V struct
//     FrameUniforms_std140). A float3 is 16-byte aligned but 12 bytes long,
//     so a following float packs into its last 4 bytes: tint at 64, time at 76.
//   [[vk::push_constant]] (PushConstant storage) -> std430
//     (FrameUniforms_std430). Same offsets for THIS struct. std140 differs
//     from std430 in three places (checked with slangc + spirv-dis):
//     - arrays: std140 rounds the stride up to 16 bytes, so float[2] and
//       float2[2] take 16 bytes per element (std430: 4 and 8);
//     - nested structs, inside an array or not: std140 aligns a struct member
//       to 16 and rounds its size up to 16, so a member after it moves too
//       (struct { float a; } then a float: std140 +16, std430 +4);
//     - matrices whose stored vectors are 2 floats (float2x2, float2x3 here:
//       Slang stores columns, a float2xN column has 2 floats): MatrixStride
//       16 in std140, 8 in std430. Columns of 3 or 4 floats are 16 in both.
// Matrices: Slang's default is column-major in memory (consecutive floats walk
// down a column, like GLM/glam), so the host uploads columns. In SPIR-V that
// shows up as the RowMajor decoration, because Slang maps an HLSL row to a
// SPIR-V column. mul(M, v) multiplies the column vector v by the matrix M.
//
// Compiled by `bun run sync:shaders` into shaders/spv/transform.spv.

struct FrameUniforms
{
    float4x4 transform; // offset 0, 64 bytes, column-major
    float3 tint;        // offset 64, 12 bytes (linear RGB multiplier)
    float time;         // offset 76: fills the 4 bytes after tint
};                      // 80 bytes

// Step 17: a uniform buffer; set 0, binding 0 of the pipeline layout.
[[vk::binding(0, 0)]]
ConstantBuffer<FrameUniforms> frame;

// Step 18: the same struct as push constants (the pipeline layout's push
// constant range: offset 0, 80 bytes, VERTEX stage).
[[vk::push_constant]]
ConstantBuffer<FrameUniforms> pushed;

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

// The work both entry points share: transform the corner, tint the color and
// let it pulse with time.
VSOutput shade(VSInput input, FrameUniforms u)
{
    VSOutput output;
    output.position = mul(u.transform, float4(input.position, 0.0, 1.0));
    output.color = input.color * u.tint * (0.85 + 0.15 * sin(u.time * 3.0));
    return output;
}

[shader("vertex")]
VSOutput vertexUniform(VSInput input)
{
    return shade(input, frame);
}

[shader("vertex")]
VSOutput vertexPush(VSInput input)
{
    return shade(input, pushed);
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}
