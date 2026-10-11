// Steps 26-27: particles simulated by a compute shader and drawn as points.
//
// One storage buffer holds the particles; the compute shader updates it every
// frame, and the graphics pipeline reads THE SAME buffer as its vertex buffer
// (usage STORAGE_BUFFER | VERTEX_BUFFER): one vertex per particle, topology
// POINT_LIST. Nothing goes back to the CPU - the data lives on the GPU, like a
// Houdini POP network whose points never leave the solver.
//
//   init      (compute) writes the starting state from the particle index
//   simulate  (compute, step 26) updates `state` in place
//   advance   (compute, step 27) reads `previous`, writes `state`: ping-pong
//   vertexMain / fragmentMain: one point per particle, colored by speed
//
// Compiled by `bun run sync:shaders` into shaders/spv/particles.spv.

static const uint GROUP_SIZE = 256;

// One particle: 16 bytes, std430 (StructuredBuffer): pos at 0, vel at 8. The
// vertex input reads the same bytes (stride 16, attributes at 0 and 8).
struct Particle
{
    float2 pos; // in units where the window is 2 tall: x in [-1/aspect, 1/aspect], y in [-1, 1]
    float2 vel; // units per second
};

// The push constant block: 16 bytes, shared by all entry points (each
// pipeline layout declares the range for the stages that read it).
struct Params
{
    float dt;     // seconds to advance (compute)
    float time;   // seconds since start (compute: moves the attractor)
    uint count;   // number of particles (compute: bounds check)
    float aspect; // window height / width (vertex)
};

[[vk::push_constant]]
ConstantBuffer<Params> params;

// The particles this dispatch writes (step 26: and reads).
[[vk::binding(0, 0)]] RWStructuredBuffer<Particle> state;
// Step 27 only: the previous state, read-only.
[[vk::binding(1, 0)]] StructuredBuffer<Particle> previous;

// A 32-bit integer hash (Chris Wellons' "lowbias32"): spreads consecutive
// indices into unrelated bit patterns, so the starting state needs no random
// numbers from the CPU.
uint hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// The top 24 bits of a hash as a float in [0, 1).
float unit(uint h)
{
    return float(h >> 8) * (1.0 / 16777216.0);
}

// The attractor: circles the window center; particles fall towards it.
float2 attractor(float time)
{
    return 0.45 * float2(cos(time * 0.6), sin(time * 0.9));
}

// One explicit Euler step: acceleration towards the attractor (softened
// inverse square, so a particle passing through the center is not flung
// away), light drag, and the window's edges reflect.
Particle update(Particle p, float dt, float time)
{
    float2 d = attractor(time) - p.pos;
    float r2 = dot(d, d) + 0.01;
    float2 acc = d * (0.08 / (r2 * sqrt(r2)));
    p.vel = (p.vel + acc * dt) * (1.0 - 0.3 * dt);
    p.pos += p.vel * dt;
    float xMax = 1.0 / params.aspect;
    if (abs(p.pos.x) > xMax)
    {
        p.pos.x = clamp(p.pos.x, -xMax, xMax);
        p.vel.x = -p.vel.x;
    }
    if (abs(p.pos.y) > 1.0)
    {
        p.pos.y = clamp(p.pos.y, -1.0, 1.0);
        p.vel.y = -p.vel.y;
    }
    return p;
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void init(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.count)
        return;
    // A disk of radius 0.8 (sqrt: uniform density), every particle orbiting
    // counterclockwise at a speed that depends on its radius.
    float r = 0.8 * sqrt(unit(hash(2 * i)));
    float a = 6.2831853 * unit(hash(2 * i + 1));
    float2 dir = float2(cos(a), sin(a));
    Particle p;
    p.pos = r * dir;
    p.vel = float2(-dir.y, dir.x) * (0.35 * sqrt(r));
    state[i] = p;
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void simulate(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.count)
        return;
    // In place: invocation i reads and writes only element i, so no
    // invocation can see a half-updated neighbor.
    state[i] = update(state[i], params.dt, params.time);
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void advance(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.count)
        return;
    state[i] = update(previous[i], params.dt, params.time);
}

// Vertex input: binding 0 = the particle buffer, stride 16.
struct VSInput
{
    [[vk::location(0)]] float2 pos : POSITION; // R32G32_SFLOAT at offset 0
    [[vk::location(1)]] float2 vel : VELOCITY; // R32G32_SFLOAT at offset 8
};

struct VSOutput
{
    float4 position : SV_Position;
    // A POINT_LIST pipeline whose vertex shader does not write PointSize is
    // invalid unless maintenance5 is enabled, which this course does not do
    // (VUID-VkGraphicsPipelineCreateInfo-topology-08773). 1.0 is the only
    // size that needs no largePoints feature.
    float pointSize : SV_PointSize;
    float3 color : COLOR0;
};

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput output;
    output.position = float4(input.pos.x * params.aspect, input.pos.y, 0.0, 1.0);
    output.pointSize = 1.0;
    // Slow particles blue, fast ones orange (linear RGB).
    float speed = saturate(length(input.vel) * 1.2);
    output.color = lerp(float3(0.1, 0.25, 1.0), float3(1.0, 0.45, 0.1), speed);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}
