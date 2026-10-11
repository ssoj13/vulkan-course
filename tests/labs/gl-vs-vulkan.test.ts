import { describe, expect, test } from 'bun:test';
import { DEPTH_DYNAMIC, DYNAMIC_STATES, SCENE, dynamicByVersion, glKey, simulate, vkKey, type DrawState } from '../../src/lib/labs/gl-vs-vulkan';
import { textErrors } from '../../src/lib/text';
import { TEXT } from '../../src/components/labs/gl-vs-vulkan.text';

const lit: DrawState = { shader: 'lit', blend: false, depth: true, vertex: 'mesh' };

test('texts: every language has the English keys and placeholders', () => {
	expect(textErrors(TEXT, 'gl-vs-vulkan')).toEqual([]);
});

describe('dynamic states (from vk.xml)', () => {
	test('the depth-test toggle uses a core dynamic state of Vulkan 1.3', () => {
		expect(DYNAMIC_STATES.find((s) => s.name === DEPTH_DYNAMIC)?.version).toBe('1.3');
	});
	test('viewport and scissor are core since 1.0; blend enable and vertex input are not core', () => {
		const names = DYNAMIC_STATES.map((s) => s.name);
		expect(names).toContain('VK_DYNAMIC_STATE_VIEWPORT');
		expect(names).toContain('VK_DYNAMIC_STATE_SCISSOR');
		expect(names).not.toContain('VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT');
		expect(names).not.toContain('VK_DYNAMIC_STATE_VERTEX_INPUT_EXT');
	});
	test('grouped by version, in version order, nothing lost', () => {
		const g = dynamicByVersion();
		expect(g.map(([v]) => v)).toEqual(['1.0', '1.3', '1.4']);
		expect(g.flatMap(([, n]) => n).length).toBe(DYNAMIC_STATES.length);
	});
});

describe('GL variants vs Vulkan pipelines', () => {
	test('depth test is not part of a GL variant; it is part of a pipeline unless dynamic', () => {
		const off = { ...lit, depth: false };
		expect(glKey(off)).toBe(glKey(lit));
		expect(vkKey(off, false)).not.toBe(vkKey(lit, false));
		expect(vkKey(off, true)).toBe(vkKey(lit, true));
	});
	test('blending, vertex format and shader each make a new GL variant', () => {
		expect(glKey({ ...lit, blend: true })).not.toBe(glKey(lit));
		expect(glKey({ ...lit, vertex: 'ui' })).not.toBe(glKey(lit));
		expect(glKey({ ...lit, shader: 'unlit' })).not.toBe(glKey(lit));
	});
	test('the starting scene: GL may compile at 4 draws of frame 1; Vulkan builds 4 or 5 pipelines up front', () => {
		const dyn = simulate(SCENE, true);
		const baked = simulate(SCENE, false);
		expect(dyn.glFrameCompiles).toBe(4);
		expect(dyn.steps.filter((s) => s.gl.compile).map((s) => s.i)).toEqual([0, 2, 3, 4]);
		expect(dyn.pipelines.length).toBe(4);
		expect(baked.pipelines.length).toBe(5);
		expect(dyn.programs).toEqual(['lit', 'unlit']);
	});
	test('Vulkan never compiles during a frame, and frame 2 compiles nothing anywhere', () => {
		const r = simulate(SCENE, false);
		expect(r.steps.every((s) => !s.vk.compile)).toBe(true);
		expect(r.steps.at(-1)!.kind).toBe('frame2');
		expect(r.steps.at(-1)!.gl.compile).toBe(false);
		expect(r.steps.at(-1)!.glFrameCompiles).toBe(r.glFrameCompiles);
	});
	test('a draw binds a pipeline only when it changes; every pipeline is one the load step created', () => {
		const r = simulate(SCENE, true);
		const draws = r.steps.filter((s) => s.kind === 'draw');
		expect(draws[1]!.vk.calls.some((c) => c.startsWith('vkCmdBindPipeline'))).toBe(false);
		for (const s of draws) expect(s.pipeline).toBeGreaterThanOrEqual(0);
		expect(r.steps[0]!.vk.calls[0]).toContain(`, ${r.pipelines.length}, `);
	});
	test('a dynamic depth test is set before the first draw (VUID-vkCmdDraw-None-07843) and when it changes', () => {
		const r = simulate(SCENE, true);
		const sets = r.steps.filter((s) => s.vk.calls.some((c) => c.startsWith('vkCmdSetDepthTestEnable'))).map((s) => s.i);
		expect(sets[0]).toBe(0);
		expect(sets).toEqual([0, 3, 5, 6]);
		expect(simulate(SCENE, false).steps.some((s) => s.vk.calls.some((c) => c.startsWith('vkCmdSetDepthTestEnable')))).toBe(false);
	});
	test('GL sets only the state that differs from the previous draw', () => {
		const r = simulate(SCENE, true);
		expect(r.steps[2]!.gl.calls).toEqual(['glDrawElements(GL_TRIANGLES, n, GL_UNSIGNED_INT, 0)']);
	});
});
