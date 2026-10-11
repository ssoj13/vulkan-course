// Step 19: instance data read through a 64-bit GPU pointer.
//
// No descriptor at all: the program asks Vulkan for the buffer's device
// address (vkGetBufferDeviceAddress, core 1.2 feature bufferDeviceAddress)
// and pushes that number as a push constant. In the shader it is a plain
// pointer, `Instance*`, and `instances[i]` is pointer arithmetic: address +
// i * 32 bytes. SPIR-V calls this storage class PhysicalStorageBuffer
// (capability PhysicalStorageBufferAddresses).
//
// The pointer carries no size: an index past the end reads whatever lies
// there (or faults the device). Descriptors have a range the driver knows;
// raw addresses trade that for flexibility (pointers inside buffers, trees,
// linked lists).
//
// Compiled by `bun run sync:shaders` into shaders/spv/device_address.spv.

// One hexagon instance: 32 bytes. Slang lays out the pointee in its "natural"
// (C-like) layout here: offset 0, scale 8, phase 12, color 16, spin 28;
// std430 would give the same offsets, so the host struct is unambiguous.
struct Instance
{
    float2 offset; // center, in units where the window is 2 tall
    float scale;   // multiplies the mesh corners
    float phase;   // starting angle, radians
    float3 color;  // multiplies the vertex colors
    float spin;    // radians per second, the sign is the direction
};

// The push constant block: 16 bytes (std430: the 8-byte pointer at 0).
struct PushData
{
    Instance* instances; // the buffer's device address
    float time;          // seconds since start
    float aspect;        // window height / width: keeps hexagons regular
};

[[vk::push_constant]]
ConstantBuffer<PushData> push;

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

// SV_VulkanInstanceID is SPIR-V InstanceIndex: firstInstance + the instance
// number within the draw (SV_InstanceID would subtract firstInstance, which
// needs the DrawParameters capability; firstInstance is 0 here anyway).
[shader("vertex")]
VSOutput vertexMain(VSInput input, uint instance : SV_VulkanInstanceID)
{
    Instance inst = push.instances[instance];
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
