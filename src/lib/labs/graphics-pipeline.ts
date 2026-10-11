/**
 * Model of the graphics-pipeline lab: the stages of the Vulkan graphics pipeline in spec order (chapter "Pipelines",
 * block diagram), which VkGraphicsPipelineCreateInfo member configures each, which of that state may be dynamic in
 * core Vulkan 1.4 (src/data/dynamic-states.json, generated from vk.xml by scripts/sync-dynamic-states.ts), how many
 * pipelines a sample scene needs with baked vs dynamic state, and one triangle walked through the stages.
 * Pure module: no DOM, no React, tested with `bun test`.
 */
import dynJson from '../../data/dynamic-states.json';

export const STAGE_IDS = [
	'vertex-input',
	'input-assembly',
	'vertex-shader',
	'tessellation',
	'geometry',
	'rasterization',
	'fragment-shader',
	'fragment-ops',
	'color-blend',
	'attachments',
] as const;
export type StageId = (typeof STAGE_IDS)[number];

/** A member of VkGraphicsPipelineCreateInfo and the structure it points at. */
export interface Config {
	member: string;
	struct: string;
}

export interface Stage {
	id: StageId;
	/** Fixed function (configured by state) or a shader stage (runs your SPIR-V). */
	kind: 'fixed' | 'shader';
	/** A pipeline may leave it out; the lab shows it dimmed. */
	optional: boolean;
	config: readonly Config[];
}

const SHADER: Config = { member: 'pStages[]', struct: 'VkPipelineShaderStageCreateInfo' };

/** Stages in the order of the spec's block diagram. */
export const STAGES: readonly Stage[] = [
	{ id: 'vertex-input', kind: 'fixed', optional: false, config: [{ member: 'pVertexInputState', struct: 'VkPipelineVertexInputStateCreateInfo' }] },
	{ id: 'input-assembly', kind: 'fixed', optional: false, config: [{ member: 'pInputAssemblyState', struct: 'VkPipelineInputAssemblyStateCreateInfo' }] },
	{ id: 'vertex-shader', kind: 'shader', optional: false, config: [SHADER] },
	{ id: 'tessellation', kind: 'shader', optional: true, config: [SHADER, { member: 'pTessellationState', struct: 'VkPipelineTessellationStateCreateInfo' }] },
	{ id: 'geometry', kind: 'shader', optional: true, config: [SHADER] },
	{
		id: 'rasterization',
		kind: 'fixed',
		optional: false,
		config: [
			{ member: 'pViewportState', struct: 'VkPipelineViewportStateCreateInfo' },
			{ member: 'pRasterizationState', struct: 'VkPipelineRasterizationStateCreateInfo' },
			{ member: 'pMultisampleState', struct: 'VkPipelineMultisampleStateCreateInfo' },
		],
	},
	{ id: 'fragment-shader', kind: 'shader', optional: false, config: [SHADER] },
	{ id: 'fragment-ops', kind: 'fixed', optional: false, config: [{ member: 'pDepthStencilState', struct: 'VkPipelineDepthStencilStateCreateInfo' }] },
	{ id: 'color-blend', kind: 'fixed', optional: false, config: [{ member: 'pColorBlendState', struct: 'VkPipelineColorBlendStateCreateInfo' }] },
	// Dynamic rendering (core 1.3): attachment formats go in the pNext chain and renderPass stays VK_NULL_HANDLE.
	{ id: 'attachments', kind: 'fixed', optional: false, config: [{ member: 'pNext', struct: 'VkPipelineRenderingCreateInfo' }] },
];

/** Members of VkGraphicsPipelineCreateInfo that belong to the whole pipeline, not one stage. */
export const WHOLE: readonly Config[] = [
	{ member: 'layout', struct: 'VkPipelineLayout' },
	{ member: 'pDynamicState', struct: 'VkPipelineDynamicStateCreateInfo' },
];

export function stage(id: StageId): Stage {
	const s = STAGES.find((x) => x.id === id);
	if (!s) throw new Error(`graphics-pipeline: unknown stage ${id}`);
	return s;
}

/** Core dynamic states with the core version that introduced each (generated from vk.xml). */
export const CORE_DYNAMIC: readonly { name: string; version: string }[] = dynJson;

export interface DynUse {
	/** VK_DYNAMIC_STATE_... */
	state: string;
	stage: StageId;
	/** The pipeline state it replaces ("struct::member"), ignored at pipeline creation when the state is dynamic. */
	replaces: string;
	/** The command that sets it while recording. */
	cmd: string;
}

const RS = 'VkPipelineRasterizationStateCreateInfo';
const DS = 'VkPipelineDepthStencilStateCreateInfo';
const VP = 'VkPipelineViewportStateCreateInfo';
const IA = 'VkPipelineInputAssemblyStateCreateInfo';

/**
 * Where each core dynamic state belongs, from the VkDynamicState descriptions of the spec. Tests check that this
 * covers exactly the names of dynamic-states.json, so a new core state in an SDK update fails until it is placed.
 */
export const DYN_USES: readonly DynUse[] = [
	{ state: 'VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE', stage: 'vertex-input', replaces: 'VkVertexInputBindingDescription::stride', cmd: 'vkCmdBindVertexBuffers2' },
	{ state: 'VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY', stage: 'input-assembly', replaces: `${IA}::topology`, cmd: 'vkCmdSetPrimitiveTopology' },
	{ state: 'VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE', stage: 'input-assembly', replaces: `${IA}::primitiveRestartEnable`, cmd: 'vkCmdSetPrimitiveRestartEnable' },
	{ state: 'VK_DYNAMIC_STATE_VIEWPORT', stage: 'rasterization', replaces: `${VP}::pViewports`, cmd: 'vkCmdSetViewport' },
	// Listed with the struct that holds the rectangles (pViewportState); the scissor test itself is the first fragment
	// operation, run before fragment shading (spec "Fragment Operations"), which the stage texts explain.
	{ state: 'VK_DYNAMIC_STATE_SCISSOR', stage: 'rasterization', replaces: `${VP}::pScissors`, cmd: 'vkCmdSetScissor' },
	{ state: 'VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT', stage: 'rasterization', replaces: `${VP}::viewportCount, pViewports`, cmd: 'vkCmdSetViewportWithCount' },
	{ state: 'VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT', stage: 'rasterization', replaces: `${VP}::scissorCount, pScissors`, cmd: 'vkCmdSetScissorWithCount' },
	{ state: 'VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE', stage: 'rasterization', replaces: `${RS}::rasterizerDiscardEnable`, cmd: 'vkCmdSetRasterizerDiscardEnable' },
	{ state: 'VK_DYNAMIC_STATE_CULL_MODE', stage: 'rasterization', replaces: `${RS}::cullMode`, cmd: 'vkCmdSetCullMode' },
	{ state: 'VK_DYNAMIC_STATE_FRONT_FACE', stage: 'rasterization', replaces: `${RS}::frontFace`, cmd: 'vkCmdSetFrontFace' },
	{ state: 'VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE', stage: 'rasterization', replaces: `${RS}::depthBiasEnable`, cmd: 'vkCmdSetDepthBiasEnable' },
	{ state: 'VK_DYNAMIC_STATE_DEPTH_BIAS', stage: 'rasterization', replaces: `${RS}::depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor`, cmd: 'vkCmdSetDepthBias' },
	{ state: 'VK_DYNAMIC_STATE_LINE_WIDTH', stage: 'rasterization', replaces: `${RS}::lineWidth`, cmd: 'vkCmdSetLineWidth' },
	{ state: 'VK_DYNAMIC_STATE_LINE_STIPPLE', stage: 'rasterization', replaces: 'VkPipelineRasterizationLineStateCreateInfo::lineStippleFactor, lineStipplePattern', cmd: 'vkCmdSetLineStipple' },
	{ state: 'VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE', stage: 'fragment-ops', replaces: `${DS}::depthTestEnable`, cmd: 'vkCmdSetDepthTestEnable' },
	{ state: 'VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE', stage: 'fragment-ops', replaces: `${DS}::depthWriteEnable`, cmd: 'vkCmdSetDepthWriteEnable' },
	{ state: 'VK_DYNAMIC_STATE_DEPTH_COMPARE_OP', stage: 'fragment-ops', replaces: `${DS}::depthCompareOp`, cmd: 'vkCmdSetDepthCompareOp' },
	{ state: 'VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE', stage: 'fragment-ops', replaces: `${DS}::depthBoundsTestEnable`, cmd: 'vkCmdSetDepthBoundsTestEnable' },
	{ state: 'VK_DYNAMIC_STATE_DEPTH_BOUNDS', stage: 'fragment-ops', replaces: `${DS}::minDepthBounds, maxDepthBounds`, cmd: 'vkCmdSetDepthBounds' },
	{ state: 'VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE', stage: 'fragment-ops', replaces: `${DS}::stencilTestEnable`, cmd: 'vkCmdSetStencilTestEnable' },
	{ state: 'VK_DYNAMIC_STATE_STENCIL_OP', stage: 'fragment-ops', replaces: `${DS}::front/back failOp, passOp, depthFailOp, compareOp`, cmd: 'vkCmdSetStencilOp' },
	{ state: 'VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK', stage: 'fragment-ops', replaces: `${DS}::front/back compareMask`, cmd: 'vkCmdSetStencilCompareMask' },
	{ state: 'VK_DYNAMIC_STATE_STENCIL_WRITE_MASK', stage: 'fragment-ops', replaces: `${DS}::front/back writeMask`, cmd: 'vkCmdSetStencilWriteMask' },
	{ state: 'VK_DYNAMIC_STATE_STENCIL_REFERENCE', stage: 'fragment-ops', replaces: `${DS}::front/back reference`, cmd: 'vkCmdSetStencilReference' },
	{ state: 'VK_DYNAMIC_STATE_BLEND_CONSTANTS', stage: 'color-blend', replaces: 'VkPipelineColorBlendStateCreateInfo::blendConstants', cmd: 'vkCmdSetBlendConstants' },
];

/** Core version that made `state` dynamic ("1.0", "1.3", "1.4"). */
export function versionOf(state: string): string {
	const d = CORE_DYNAMIC.find((x) => x.name === state);
	if (!d) throw new Error(`graphics-pipeline: ${state} is not a core dynamic state`);
	return d.version;
}

export const dynamicOf = (id: StageId): readonly DynUse[] => DYN_USES.filter((d) => d.stage === id);

/** Differences between DYN_USES and dynamic-states.json (empty = consistent). */
export function dynErrors(): string[] {
	const core = new Set(CORE_DYNAMIC.map((d) => d.name));
	const used = DYN_USES.map((d) => d.state);
	return [
		...used.filter((s) => !core.has(s)).map((s) => `${s}: not a core dynamic state`),
		...[...core].filter((s) => !used.includes(s)).map((s) => `${s}: no stage assigned`),
		...used.filter((s, i) => used.indexOf(s) !== i).map((s) => `${s}: listed twice`),
	];
}

// ---------------------------------------------------------------------------------------------------------------
// Pipeline count of a sample scene. Shaders are always baked into a pipeline (core Vulkan has no dynamic shaders;
// VK_EXT_shader_object is an extension), so every material is its own pipeline; the other axes multiply the count
// only while their state is baked. The scene itself is illustrative.

export const AXES = ['viewport', 'cull', 'depth'] as const;
export type Axis = (typeof AXES)[number];

/** Variants the sample scene draws per axis: 3 materials; cull back vs none; depth test on vs off. */
export const SCENE = { materials: 3, cull: 2, depth: 2 } as const;

/** Dynamic states an axis needs when it is made dynamic. */
export const AXIS_STATES: Readonly<Record<Axis, readonly string[]>> = {
	viewport: ['VK_DYNAMIC_STATE_VIEWPORT', 'VK_DYNAMIC_STATE_SCISSOR'],
	cull: ['VK_DYNAMIC_STATE_CULL_MODE'],
	depth: ['VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE'],
};

export type DynChoice = Readonly<Record<Axis, boolean>>;

export const FULLY_STATIC: DynChoice = { viewport: false, cull: false, depth: false };
export const DYNAMIC: DynChoice = { viewport: true, cull: true, depth: true };

export interface Count {
	/** Pipelines to build up front. */
	pipelines: number;
	/** Pipelines rebuilt on every window resize (baked viewport/scissor), 0 when they are dynamic. */
	perResize: number;
	/** pDynamicStates of every pipeline. */
	dynamicStates: string[];
	/** vkCmdSet* calls a draw may need before it (one per dynamic state). */
	cmds: string[];
}

export function count(c: DynChoice): Count {
	const pipelines = SCENE.materials * (c.cull ? 1 : SCENE.cull) * (c.depth ? 1 : SCENE.depth);
	const dynamicStates = AXES.filter((a) => c[a]).flatMap((a) => AXIS_STATES[a]);
	const cmds = dynamicStates.map((s) => DYN_USES.find((d) => d.state === s)!.cmd);
	return { pipelines, perResize: c.viewport ? 0 : pipelines, dynamicStates, cmds };
}

// ---------------------------------------------------------------------------------------------------------------
// One triangle through the pipeline: real rules (viewport transform, the spec's signed area for facing, coverage at
// pixel centers, linear depth, depth test LESS), tiny scale (a 16x10 framebuffer).

export const FB = { w: 16, h: 10 } as const;

export interface Vtx {
	/** Object-space position (vertex buffer contents). */
	pos: readonly [number, number, number];
	/** RGB 0..1. */
	color: readonly [number, number, number];
}

/** Vertex buffer: three vertices, one binding, position + color attributes. */
export const VERTS: readonly Vtx[] = [
	{ pos: [-0.7, 0.6, 0.2], color: [1, 0.25, 0.25] },
	{ pos: [0.7, 0.4, 0.8], color: [0.25, 0.85, 0.35] },
	{ pos: [-0.1, -0.7, 0.5], color: [0.3, 0.45, 1] },
];

/** The lab's vertex shader: a model transform that moves the triangle right by 0.15, w = 1 (no perspective). */
export const SHIFT_X = 0.15;
export function vertexShader(v: Vtx): [number, number, number, number] {
	return [v.pos[0] + SHIFT_X, v.pos[1], v.pos[2], 1];
}

/** Viewport transform with x = y = 0, width/height = the framebuffer, depth 0..1: NDC -> framebuffer pixels. */
export function toFramebuffer(clip: readonly number[]): [number, number, number] {
	const [x, y, z, w] = clip as [number, number, number, number];
	const nx = x / w;
	const ny = y / w;
	return [((nx + 1) * FB.w) / 2, ((ny + 1) * FB.h) / 2, z / w];
}

/** The spec's signed area in framebuffer coordinates (chapter "Rasterization", Basic Polygon Rasterization). */
export function signedArea(p: readonly (readonly number[])[]): number {
	let s = 0;
	for (let i = 0; i < p.length; i++) {
		const a = p[i]!;
		const b = p[(i + 1) % p.length]!;
		s += a[0]! * b[1]! - b[0]! * a[1]!;
	}
	return -s / 2;
}

export type CullMode = 'none' | 'back' | 'front';
export const CULL_MODES: readonly CullMode[] = ['none', 'back', 'front'];

/** VK_FRONT_FACE_COUNTER_CLOCKWISE: positive area = front-facing. */
export function isFront(area: number): boolean {
	return area > 0;
}

export function culled(area: number, mode: CullMode): boolean {
	return mode === 'back' ? !isFront(area) : mode === 'front' ? isFront(area) : false;
}

/** One covered pixel, the data the stages after rasterization see. */
export interface Frag {
	x: number;
	y: number;
	z: number;
	color: [number, number, number];
}

/**
 * Coverage of a triangle at pixel centers (one sample per pixel) with barycentric weights; z and color are
 * interpolated linearly (w = 1, so perspective-correct and linear agree). The lab's vertices never put a pixel
 * center exactly on an edge, so the spec's tie-breaking rule never applies (tests check it).
 */
export function rasterize(p: readonly (readonly [number, number, number])[], colors: readonly (readonly number[])[]): Frag[] {
	const a = p[0]!;
	const b = p[1]!;
	const c = p[2]!;
	const area2 = (b[0]! - a[0]!) * (c[1]! - a[1]!) - (c[0]! - a[0]!) * (b[1]! - a[1]!);
	const out: Frag[] = [];
	for (let y = 0; y < FB.h; y++)
		for (let x = 0; x < FB.w; x++) {
			const px = x + 0.5;
			const py = y + 0.5;
			const w0 = ((b[0]! - px) * (c[1]! - py) - (c[0]! - px) * (b[1]! - py)) / area2;
			const w1 = ((c[0]! - px) * (a[1]! - py) - (a[0]! - px) * (c[1]! - py)) / area2;
			const w2 = 1 - w0 - w1;
			if (w0 < 0 || w1 < 0 || w2 < 0) continue;
			const mix = (i: number) => w0 * colors[0]![i]! + w1 * colors[1]![i]! + w2 * colors[2]![i]!;
			out.push({ x, y, z: w0 * a[2]! + w1 * b[2]! + w2 * c[2]!, color: [mix(0), mix(1), mix(2)] });
		}
	return out;
}

/** Depth buffer before the draw: cleared to 1.0, with an earlier draw at depth 0.4 in columns 10.. (the "wall"). */
export const WALL = { fromX: 10, depth: 0.4 } as const;
export const depthBefore = (x: number): number => (x >= WALL.fromX ? WALL.depth : 1);

/** VK_COMPARE_OP_LESS: the fragment passes when it is nearer than what the depth buffer holds. */
export const depthPasses = (f: Frag): boolean => f.z < depthBefore(f.x);

export type WalkStep = 'vertex-input' | 'input-assembly' | 'vertex-shader' | 'skipped' | 'rasterization' | 'fragment-shader' | 'fragment-ops' | 'color-blend' | 'attachments';

export interface WalkFrame {
	step: WalkStep;
	/** The stage highlighted in the strip. */
	stage: StageId;
	/** Framebuffer positions of the vertices (after the viewport transform). */
	fb: [number, number, number][];
	area: number;
	culled: boolean;
	/** Fragments that exist at this step (after culling, and after the depth test from fragment-ops on). */
	frags: Frag[];
	/** Fragments the depth test rejected (shown from fragment-ops on). */
	rejected: Frag[];
}

/** One frame per stage the triangle passes, for a cull mode. */
export function walk(cull: CullMode): WalkFrame[] {
	const fb = VERTS.map((v) => toFramebuffer(vertexShader(v)));
	const area = signedArea(fb);
	const isCulled = culled(area, cull);
	const all = isCulled ? [] : rasterize(fb, VERTS.map((v) => v.color));
	const kept = all.filter(depthPasses);
	const rejected = all.filter((f) => !depthPasses(f));
	const f = (step: WalkStep, stage: StageId, frags: Frag[], rej: Frag[] = []): WalkFrame => ({ step, stage, fb, area, culled: isCulled, frags, rejected: rej });
	return [
		f('vertex-input', 'vertex-input', []),
		f('input-assembly', 'input-assembly', []),
		f('vertex-shader', 'vertex-shader', []),
		f('skipped', 'tessellation', []),
		f('rasterization', 'rasterization', all),
		f('fragment-shader', 'fragment-shader', all),
		f('fragment-ops', 'fragment-ops', kept, rejected),
		f('color-blend', 'color-blend', kept, rejected),
		f('attachments', 'attachments', kept, rejected),
	];
}

/** Short form of a Vulkan name for compact labels: `VK_DYNAMIC_STATE_CULL_MODE` -> `CULL_MODE`. */
export const shortState = (s: string): string => s.replace(/^VK_DYNAMIC_STATE_/, '');

/** CSS color of a fragment. */
export const rgb = (c: readonly number[]): string => `rgb(${c.map((v) => Math.round(v * 255)).join(' ')})`;
