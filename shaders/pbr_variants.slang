// Step 30: one fragment shader, many pipelines.
//
// Five specialization constants turn this source into a family of shader
// variants, the way an engine's material system does: the host sets them per
// pipeline (VkSpecializationInfo), and the driver compiles each combination
// separately, with the constants folded in - loops of a constant trip count
// unrolled, the branches not taken removed. That compilation is what a
// pipeline cache saves. Step 30 creates the pipelines and never draws with
// them: the image the shader would produce does not matter there, only how
// long the driver takes to turn it into GPU code.
//
// The shading is a small physically based model (Cook-Torrance: GGX normal
// distribution, height-correlated Smith visibility, Schlick Fresnel) over
// procedural inputs, so the compiler has real work:
// value-noise octaves for the surface, a loop over point lights, each with a
// few soft-shadow taps.
//
// Compiled by `bun run sync:shaders` into shaders/spv/pbr_variants.spv.

// Specialization constants: SpecId 0..4. The values here are the defaults a
// pipeline gets when VkSpecializationInfo does not set the constant.
[vk::constant_id(0)] const int LIGHTS = 1;      // point lights, 1..8
[vk::constant_id(1)] const int OCTAVES = 1;     // noise octaves, 1..8
[vk::constant_id(2)] const int BRDF = 0;        // 0 = Lambert + Blinn-Phong, 1 = GGX / Smith / Schlick
[vk::constant_id(3)] const int TONEMAP = 0;     // 0 = clamp, 1 = ACES fit
[vk::constant_id(4)] const int VARIANT = 0;     // seeds the noise: makes every pipeline unique

static const float PI = 3.14159265;
static const int SHADOW_TAPS = 4;

struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// A full-screen triangle from the vertex index alone: no vertex buffer.
[shader("vertex")]
VSOutput vertexMain(uint vertex : SV_VulkanVertexID)
{
    float2 uv = float2((vertex << 1) & 2, vertex & 2);
    VSOutput output;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv = uv;
    return output;
}

float hash(float2 p)
{
    float3 q = frac(float3(p.xyx) * 0.1031 + float(VARIANT) * 0.0137);
    q += dot(q, q.yzx + 33.33);
    return frac((q.x + q.y) * q.z);
}

float valueNoise(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + float2(1.0, 0.0));
    float c = hash(i + float2(0.0, 1.0));
    float d = hash(i + float2(1.0, 1.0));
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

// Fractal noise: OCTAVES layers, each twice the frequency and half the weight.
float fbm(float2 p)
{
    float sum = 0.0;
    float weight = 0.5;
    for (int octave = 0; octave < OCTAVES; ++octave)
    {
        sum += weight * valueNoise(p);
        p = p * 2.03 + float2(17.0, 9.0);
        weight *= 0.5;
    }
    return sum;
}

// GGX / Trowbridge-Reitz normal distribution.
float distributionGGX(float nh, float alpha)
{
    float a2 = alpha * alpha;
    float d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Height-correlated Smith visibility term V = G / (4 n.l n.v).
float visibilitySmith(float nl, float nv, float alpha)
{
    float a2 = alpha * alpha;
    float gv = nl * sqrt(nv * nv * (1.0 - a2) + a2);
    float gl = nv * sqrt(nl * nl * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

float3 fresnelSchlick(float vh, float3 f0)
{
    return f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
}

// Soft shadow: a few taps of the height field between the point and the light.
float shadow(float2 p, float2 toLight)
{
    float lit = 0.0;
    for (int tap = 1; tap <= SHADOW_TAPS; ++tap)
    {
        float t = float(tap) / float(SHADOW_TAPS);
        lit += step(fbm(p + toLight * t * 0.2), 0.55 + 0.1 * t);
    }
    return lit / float(SHADOW_TAPS);
}

float3 acesFit(float3 x)
{
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    float2 p = input.uv * 8.0;
    // The height field gives the normal (central differences) and the material.
    float h = fbm(p);
    float hx = fbm(p + float2(0.01, 0.0)) - h;
    float hy = fbm(p + float2(0.0, 0.01)) - h;
    float3 n = normalize(float3(-hx * 40.0, -hy * 40.0, 1.0));
    float3 v = float3(0.0, 0.0, 1.0);
    float roughness = lerp(0.15, 0.9, h);
    float metallic = step(0.6, h);
    float3 baseColor = lerp(float3(0.8, 0.5, 0.3), float3(0.9, 0.9, 0.95), h);
    float3 f0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);

    float3 color = 0.03 * baseColor;
    for (int light = 0; light < LIGHTS; ++light)
    {
        float angle = 2.0 * PI * float(light) / float(LIGHTS);
        float2 lightPos = float2(4.0 + 3.0 * cos(angle), 4.0 + 3.0 * sin(angle));
        float3 toLight = float3(lightPos - p, 1.5);
        float distance2 = dot(toLight, toLight);
        float3 l = toLight * rsqrt(distance2);
        float3 hv = normalize(l + v);
        float nl = saturate(dot(n, l));
        float nv = saturate(dot(n, v)) + 1e-4;
        float nh = saturate(dot(n, hv));
        float vh = saturate(dot(v, hv));
        float3 radiance = float3(1.0, 0.95, 0.9) * (8.0 / distance2);
        float3 lightColor;
        if (BRDF == 1)
        {
            float alpha = roughness * roughness;
            float3 f = fresnelSchlick(vh, f0);
            float3 specular = distributionGGX(nh, alpha) * visibilitySmith(nl, nv, alpha) * f;
            float3 diffuse = (1.0 - f) * (1.0 - metallic) * baseColor / PI;
            lightColor = (diffuse + specular) * radiance * nl;
        }
        else
        {
            float shininess = 2.0 / max(roughness * roughness, 1e-3);
            lightColor = (baseColor / PI + pow(nh, shininess) * f0) * radiance * nl;
        }
        color += lightColor * shadow(p, toLight.xy);
    }
    color = TONEMAP == 1 ? acesFit(color) : saturate(color);
    return float4(color, 1.0);
}
