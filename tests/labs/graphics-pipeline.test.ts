import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import dyn from '../../src/data/dynamic-states.json';
import {
	CORE_DYNAMIC,
	DYNAMIC,
	DYN_USES,
	FB,
	FULLY_STATIC,
	STAGES,
	STAGE_IDS,
	VERTS,
	WALL,
	WHOLE,
	count,
	culled,
	depthPasses,
	dynErrors,
	dynamicOf,
	rasterize,
	signedArea,
	toFramebuffer,
	vertexShader,
	versionOf,
	walk,
} from '../../src/lib/labs/graphics-pipeline';
import { textErrors } from '../../src/lib/text';
import { TEXT } from '../../src/components/labs/graphics-pipeline.text';

const registry = process.env.VULKAN_SDK ? join(process.env.VULKAN_SDK, 'share', 'vulkan', 'registry') : '';

describe('dynamic-states.json (scripts/sync-dynamic-states.ts)', () => {
	test('core states only: 9 from 1.0, the extended dynamic state 1 and 2 promotions in 1.3, line stipple in 1.4', () => {
		const by = (v: string) => dyn.filter((d) => d.version === v).map((d) => d.name);
		expect(by('1.0')).toHaveLength(9);
		expect(by('1.3')).toHaveLength(15);
		expect(by('1.4')).toEqual(['VK_DYNAMIC_STATE_LINE_STIPPLE']);
		expect(dyn).toHaveLength(25);
		// No extension-only or alias names.
		expect(dyn.every((d) => /^VK_DYNAMIC_STATE_[A-Z0-9_]+$/.test(d.name) && !/_(EXT|KHR|NV)$/.test(d.name))).toBe(true);
		expect(dyn.map((d) => d.version)).toEqual([...dyn.map((d) => d.version)].sort());
	});
});

describe('graphics-pipeline stages', () => {
	test('spec order, every stage once; only tessellation and geometry are optional', () => {
		expect(STAGES.map((s) => s.id)).toEqual([...STAGE_IDS]);
		expect(STAGES.filter((s) => s.optional).map((s) => s.id)).toEqual(['tessellation', 'geometry']);
		expect(STAGES.filter((s) => s.kind === 'shader').map((s) => s.id)).toEqual(['vertex-shader', 'tessellation', 'geometry', 'fragment-shader']);
	});

	test('every core dynamic state is placed on exactly one stage, and shaders have none', () => {
		expect(dynErrors()).toEqual([]);
		expect(DYN_USES).toHaveLength(CORE_DYNAMIC.length);
		for (const s of STAGES.filter((x) => x.kind === 'shader')) expect(dynamicOf(s.id)).toEqual([]);
		expect(dynamicOf('rasterization').map((d) => d.state)).toContain('VK_DYNAMIC_STATE_CULL_MODE');
		expect(versionOf('VK_DYNAMIC_STATE_VIEWPORT')).toBe('1.0');
		expect(versionOf('VK_DYNAMIC_STATE_CULL_MODE')).toBe('1.3');
		expect(() => versionOf('VK_DYNAMIC_STATE_POLYGON_MODE_EXT')).toThrow();
	});

	test.skipIf(!registry || !existsSync(join(registry, 'validusage.json')))('every struct and vkCmdSet* command exists in the registry', async () => {
		const db = (await Bun.file(join(registry, 'validusage.json')).json()) as { validation: Record<string, unknown> };
		const structs = [...STAGES.flatMap((s) => s.config), ...WHOLE].map((c) => c.struct).filter((s) => s !== 'VkPipelineLayout');
		for (const s of structs) expect(s in db.validation).toBe(true);
		for (const d of DYN_USES) expect(d.cmd in db.validation).toBe(true);
		expect('VkPipelineRasterizationLineStateCreateInfo' in db.validation).toBe(true);
	});
});

describe('pipeline count', () => {
	test('fully static: materials x cull x depth, and every resize rebuilds them all', () => {
		const c = count(FULLY_STATIC);
		expect(c.pipelines).toBe(12);
		expect(c.perResize).toBe(12);
		expect(c.dynamicStates).toEqual([]);
	});
	test('dynamic: one pipeline per material, no rebuild on resize, one vkCmdSet* per dynamic state', () => {
		const c = count(DYNAMIC);
		expect(c.pipelines).toBe(3);
		expect(c.perResize).toBe(0);
		expect(c.cmds).toEqual(['vkCmdSetViewport', 'vkCmdSetScissor', 'vkCmdSetCullMode', 'vkCmdSetDepthTestEnable']);
		expect(count({ viewport: true, cull: false, depth: true }).pipelines).toBe(6);
	});
});

describe('one triangle', () => {
	const fb = VERTS.map((v) => toFramebuffer(vertexShader(v)));

	test('viewport transform maps NDC corners to the framebuffer corners', () => {
		expect(toFramebuffer([-1, -1, 0, 1])).toEqual([0, 0, 0]);
		expect(toFramebuffer([1, 1, 1, 1])).toEqual([FB.w, FB.h, 1]);
	});

	test('signed area follows the spec formula: counter-clockwise on screen (y down) is positive', () => {
		// (0,0) -> (0,1) -> (1,0) in framebuffer space (y down) turns counter-clockwise as seen on screen.
		expect(signedArea([[0, 0], [0, 1], [1, 0]])).toBe(0.5);
		expect(signedArea(fb)).toBeGreaterThan(0);
		expect(culled(signedArea(fb), 'back')).toBe(false);
		expect(culled(signedArea(fb), 'front')).toBe(true);
		expect(culled(-1, 'none')).toBe(false);
	});

	test('no pixel center lies exactly on an edge, so the tie-breaking rule never applies', () => {
		const edge = (a: readonly number[], b: readonly number[], x: number, y: number) => (b[0]! - a[0]!) * (y - a[1]!) - (b[1]! - a[1]!) * (x - a[0]!);
		for (let y = 0; y < FB.h; y++)
			for (let x = 0; x < FB.w; x++) for (let i = 0; i < 3; i++) expect(Math.abs(edge(fb[i]!, fb[(i + 1) % 3]!, x + 0.5, y + 0.5))).toBeGreaterThan(1e-9);
	});

	test('coverage, interpolation and the depth test', () => {
		const frags = rasterize(fb, VERTS.map((v) => v.color));
		expect(frags.length).toBe(34);
		for (const f of frags) {
			expect(f.z).toBeGreaterThanOrEqual(0.2);
			expect(f.z).toBeLessThanOrEqual(0.8);
			for (const c of f.color) expect(c).toBeGreaterThanOrEqual(0);
		}
		// Left of the wall the depth buffer is cleared to 1: everything passes; behind the wall (0.4) everything fails here.
		expect(frags.filter((f) => f.x < WALL.fromX).every(depthPasses)).toBe(true);
		expect(frags.filter((f) => f.x >= WALL.fromX).some(depthPasses)).toBe(false);
	});

	test('the walk: one frame per stage, culling empties everything after rasterization', () => {
		const w = walk('back');
		expect(w.map((f) => f.step)).toEqual(['vertex-input', 'input-assembly', 'vertex-shader', 'skipped', 'rasterization', 'fragment-shader', 'fragment-ops', 'color-blend', 'attachments']);
		expect(w[4]!.frags).toHaveLength(34);
		expect(w[6]!.frags.length + w[6]!.rejected.length).toBe(34);
		expect(w[6]!.rejected.length).toBeGreaterThan(0);
		const c = walk('front');
		expect(c.every((f) => f.frags.length === 0 && f.rejected.length === 0)).toBe(true);
	});
});

test('texts: every language has the same keys and placeholders', () => {
	expect(textErrors(TEXT, 'graphics-pipeline')).toEqual([]);
	for (const s of STAGE_IDS) expect(TEXT.en[`name.${s}`]).toBeTruthy();
});
