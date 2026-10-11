/**
 * Model of the "OpenGL state vs Vulkan pipelines" lab. The reader edits a short list of draws, each with its own
 * state (shader, blending, depth test, vertex format), and the model plays one frame of them twice:
 *  - GL style: state is set by small calls right before each draw. When a draw meets a combination of the
 *    shader-affecting state that the driver has not seen yet, the driver may need to build a shader variant there
 *    and then, in the middle of the frame.
 *  - Vulkan: every combination the frame uses is a pipeline created at load time (vkCreateGraphicsPipelines);
 *    a draw only binds one. Depth test can be dynamic state (core since 1.3), then it needs no pipeline of its own.
 * Which state goes into a GL variant follows documented driver designs (see glKey); it is a model of a
 * typical driver, not of any one in particular. Pure module: the view is components/labs/GlVsVulkanView.tsx.
 */
import dynamicStates from '../../data/dynamic-states.json';

export type Shader = 'lit' | 'unlit';
/** Vertex formats: a mesh (position, normal, uv) or a UI quad (position, color). */
export type VertexFormat = 'mesh' | 'ui';

export interface DrawState {
	shader: Shader;
	blend: boolean;
	depth: boolean;
	vertex: VertexFormat;
}

/** Names of the scene's draws (translated by the view); draws the reader adds are `extra`. */
export type DrawName = 'rock' | 'tree' | 'glass' | 'sky' | 'hud' | 'rock2' | 'gizmo' | 'extra';

/** A draw of the scene: what it draws plus its state. */
export interface Draw extends DrawState {
	name: DrawName;
}

export const SHADERS: readonly Shader[] = ['lit', 'unlit'];
export const VERTEX_FORMATS: readonly VertexFormat[] = ['mesh', 'ui'];
/** The reader may add draws up to this many: a frame long enough to show the point, short enough to scrub. */
export const MAX_DRAWS = 8;

/** The starting scene: opaque meshes, glass, sky, a UI overlay and an editor gizmo drawn over everything. */
export const SCENE: readonly Draw[] = [
	{ name: 'rock', shader: 'lit', blend: false, depth: true, vertex: 'mesh' },
	{ name: 'tree', shader: 'lit', blend: false, depth: true, vertex: 'mesh' },
	{ name: 'glass', shader: 'lit', blend: true, depth: true, vertex: 'mesh' },
	{ name: 'sky', shader: 'unlit', blend: false, depth: false, vertex: 'mesh' },
	{ name: 'hud', shader: 'unlit', blend: true, depth: false, vertex: 'ui' },
	{ name: 'rock2', shader: 'lit', blend: false, depth: true, vertex: 'mesh' },
	{ name: 'gizmo', shader: 'lit', blend: false, depth: false, vertex: 'mesh' },
];

/**
 * State a GL driver may have to compile into the GPU's shader code, so that a new combination can mean a new variant:
 *  - blending: on GPUs that lack fixed-function blending in part or in full, blending is shader code; Mesa's
 *    nir_lower_blend runs it "as a dedicated 'blend shader' stage on Midgard/Bifrost [Mali], or as a
 *    fragment shader variant on typical GPUs";
 *  - vertex format: "Vertex fetch has not been a fixed function hardware unit for quite some time"; the vertex shader
 *    is patched for the current vertex declaration, or on AMD GCN calls a fetch routine the driver builds for it
 *    (Emil Persson, "Low-level Shader Optimization for Next-Gen and DX11", GDC 2014).
 * Depth test is left out: in this model it is register state of the fixed-function depth unit and costs no compile.
 * The key is the identity of a GL shader variant: shader, blending, vertex format.
 */
export function glKey(d: DrawState): string {
	return `${d.shader}|${d.blend ? 'blend' : 'opaque'}|${d.vertex}`;
}

/** Identity of a Vulkan pipeline: all baked state; depth test only when it is not dynamic. */
export function vkKey(d: DrawState, dynDepth: boolean): string {
	return dynDepth ? glKey(d) : `${glKey(d)}|${d.depth ? 'depth' : 'nodepth'}`;
}

/** What one draw costs on one side: the calls it makes and whether it waits for a compile. */
export interface Side {
	calls: string[];
	compile: boolean;
}

export interface Step {
	kind: 'load' | 'draw' | 'frame2';
	/** Draw index for `draw` steps. */
	i: number;
	gl: Side;
	vk: Side;
	/** Totals so far. */
	glFrameCompiles: number;
	/** Index of the Vulkan pipeline bound by this draw (into Result.pipelines). */
	pipeline: number;
}

export interface Result {
	/** GL programs compiled and linked at load time (one per shader the scene uses). */
	programs: Shader[];
	/** Vulkan pipelines created at load time, in order of first use. */
	pipelines: string[];
	/** Load, every draw of frame 1, then frame 2 (the same draws again). */
	steps: Step[];
	/** GL draws that had to wait for a variant compile in frame 1 (frame 2 has none: the driver keeps them). */
	glFrameCompiles: number;
}

const glCap = (on: boolean, cap: string) => `${on ? 'glEnable' : 'glDisable'}(${cap})`;
const PROGRAM: Record<Shader, string> = { lit: 'progLit', unlit: 'progUnlit' };
const VAO: Record<VertexFormat, string> = { mesh: 'vaoMesh', ui: 'vaoUi' };

/** The GL calls of a draw: only the state that differs from the previous draw is set, as a careful GL app does. */
export function glCalls(prev: DrawState | null, d: DrawState): string[] {
	const out: string[] = [];
	if (!prev) out.push('glViewport(0, 0, w, h)');
	if (prev?.shader !== d.shader) out.push(`glUseProgram(${PROGRAM[d.shader]})`);
	if (prev?.blend !== d.blend) {
		out.push(glCap(d.blend, 'GL_BLEND'));
		if (d.blend) out.push('glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)');
	}
	if (prev?.depth !== d.depth) out.push(glCap(d.depth, 'GL_DEPTH_TEST'));
	if (prev?.vertex !== d.vertex) out.push(`glBindVertexArray(${VAO[d.vertex]})`);
	out.push(d.vertex === 'ui' ? 'glDrawArrays(GL_TRIANGLES, 0, 6)' : 'glDrawElements(GL_TRIANGLES, n, GL_UNSIGNED_INT, 0)');
	return out;
}

/** The Vulkan commands of a draw: bind a pipeline when it changes, set dynamic state, draw. */
export function vkCalls(prev: DrawState | null, d: DrawState, dynDepth: boolean, prevPipe: number, pipe: number): string[] {
	const out: string[] = [];
	if (!prev) out.push('vkCmdSetViewport(cmd, 0, 1, &viewport)', 'vkCmdSetScissor(cmd, 0, 1, &scissor)');
	if (pipe !== prevPipe) out.push(`vkCmdBindPipeline(cmd, GRAPHICS, pipelines[${pipe}])`);
	// A dynamic depth test must be set in the command buffer before the first draw (VUID-vkCmdDraw-None-07843).
	if (dynDepth && prev?.depth !== d.depth) out.push(`vkCmdSetDepthTestEnable(cmd, ${d.depth ? 'VK_TRUE' : 'VK_FALSE'})`);
	if (prev?.vertex !== d.vertex) out.push(`vkCmdBindVertexBuffers(cmd, 0, 1, &${d.vertex}Buffer, &offset)`);
	out.push(d.vertex === 'ui' ? 'vkCmdDraw(cmd, 6, 1, 0, 0)' : 'vkCmdDrawIndexed(cmd, n, 1, 0, 0, 0)');
	return out;
}

/** Plays the scene: load time, frame 1 draw by draw, then frame 2 as one summary step. */
export function simulate(draws: readonly DrawState[], dynDepth: boolean): Result {
	const programs = SHADERS.filter((s) => draws.some((d) => d.shader === s));
	const pipelines: string[] = [];
	for (const d of draws) if (!pipelines.includes(vkKey(d, dynDepth))) pipelines.push(vkKey(d, dynDepth));

	const loadGl = programs.map((s) => `glCompileShader ×2, glLinkProgram(${PROGRAM[s]})`);
	const loadVk = [`vkCreateGraphicsPipelines(device, cache, ${pipelines.length}, infos, nullptr, pipelines)`];
	const steps: Step[] = [{ kind: 'load', i: -1, gl: { calls: loadGl, compile: false }, vk: { calls: loadVk, compile: false }, glFrameCompiles: 0, pipeline: -1 }];

	const seen = new Set<string>();
	let compiles = 0;
	let prevPipe = -1;
	draws.forEach((d, i) => {
		const prev = draws[i - 1] ?? null;
		const compile = !seen.has(glKey(d));
		seen.add(glKey(d));
		if (compile) compiles++;
		const pipe = pipelines.indexOf(vkKey(d, dynDepth));
		steps.push({
			kind: 'draw',
			i,
			gl: { calls: glCalls(prev, d), compile },
			vk: { calls: vkCalls(prev, d, dynDepth, prevPipe, pipe), compile: false },
			glFrameCompiles: compiles,
			pipeline: pipe,
		});
		prevPipe = pipe;
	});
	steps.push({ kind: 'frame2', i: -1, gl: { calls: [], compile: false }, vk: { calls: [], compile: false }, glFrameCompiles: compiles, pipeline: -1 });
	return { programs, pipelines, steps, glFrameCompiles: compiles };
}

/** A core VkDynamicState value and the Vulkan version that made it core (src/data/dynamic-states.json, from vk.xml). */
export interface DynamicState {
	name: string;
	version: string;
}

export const DYNAMIC_STATES: readonly DynamicState[] = dynamicStates;

/** Core dynamic states grouped by the version that added them, in version order. */
export function dynamicByVersion(): [string, string[]][] {
	const groups = new Map<string, string[]>();
	for (const s of DYNAMIC_STATES) groups.set(s.version, [...(groups.get(s.version) ?? []), s.name]);
	return [...groups].sort(([a], [b]) => a.localeCompare(b, 'en', { numeric: true }));
}

/** The dynamic state the lab uses for its depth-test toggle; must be core (tests check it is in the list). */
export const DEPTH_DYNAMIC = 'VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE';
