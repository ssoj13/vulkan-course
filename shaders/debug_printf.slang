// Step 28 (--printf): a fragment shader that prints from the GPU.
//
// It replaces cube.slang's fragmentMain in a second pipeline; the vertex stage
// still comes from cube.spv. The interface matches by Location: cube.slang's
// vertex output `color` and this input are both Location 0 (see spirv-dis).
//
// printf compiles to the NonSemantic.DebugPrintf extended instruction set
// (OpExtInst ... DebugPrintf, SPV_KHR_non_semantic_info). A driver ignores it;
// the validation layer, with its DebugPrintf setting on, instruments the shader
// to write the arguments into a buffer and prints them after the submission
// completes. Without the layer setting the program runs unchanged and prints
// nothing.
//
// Only ONE pixel prints: every fragment covering pixel (400, 300), the window
// center. SV_Position.xy of a fragment is its pixel center (400.5, 300.5), so
// the integer part names the pixel. One pixel can print several lines, one per
// fragment shaded there: at time 0 both cubes' front faces cover it, so it
// prints cube 0's fragment (drawn first) and cube 1's (nearer, drawn second,
// it replaces cube 0's color).
//
// Compiled by `bun run sync:shaders` into shaders/spv/debug_printf.spv.

struct FSInput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 color : COLOR0;
};

[shader("fragment")]
float4 fragmentPrintf(FSInput input) : SV_Target
{
    uint2 pixel = uint2(input.position.xy);
    if (pixel.x == 400 && pixel.y == 300)
    {
        printf("pixel (%u, %u): depth %f, color (%f, %f, %f)\n",
               pixel.x, pixel.y, input.position.z, input.color.x, input.color.y, input.color.z);
    }
    return float4(input.color, 1.0);
}
