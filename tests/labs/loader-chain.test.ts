import { describe, expect, test } from 'bun:test';
import { TEXT } from '../../src/components/labs/loader-chain.text';
import { CALLS, IMPLICIT_LAYERS, VALIDATION, activeLayers, implicitActive, layerNode, trace, type Setup } from '../../src/lib/labs/loader-chain';
import { textErrors } from '../../src/lib/text';

const base: Setup = { call: 'vkCreateInstance', env: new Set(), validation: true, second: false, gpu: 'nvidia', via: 'gdpa' };
const run = (patch: Partial<Setup>) => trace({ ...base, ...patch });
const names = (p: Partial<Setup>) => activeLayers({ ...base, ...p }).map((l) => l.name);

test('texts: every language has the same keys and placeholders; every caption key exists', () => {
	expect(textErrors(TEXT, 'loader-chain')).toEqual([]);
	for (const call of CALLS)
		for (const second of [false, true])
			for (const via of ['gdpa', 'export'] as const)
				for (const h of run({ call, second, via }).hops) expect(TEXT.en[h.key]).toBeString();
});

describe('implicit layers (manifests of the course machine)', () => {
	test('eight are registered; every one has a disable variable (required by the loader)', () => {
		expect(IMPLICIT_LAYERS).toHaveLength(8);
		for (const l of IMPLICIT_LAYERS) expect(l.disableEnv).toBeTruthy();
	});
	test('without variables the chain is the one the loader printed for step 02', () => {
		expect(names({})).toEqual(['VK_LAYER_NV_optimus', 'VK_LAYER_NV_present', 'VK_LAYER_bandicam_helper', VALIDATION]);
	});
	test('a layer with an enable variable joins only when it is set', () => {
		const rd = IMPLICIT_LAYERS.find((l) => l.name === 'VK_LAYER_RENDERDOC_Capture')!;
		expect(implicitActive(rd, new Set())).toBe(false);
		expect(implicitActive(rd, new Set([rd.enableEnv!]))).toBe(true);
		// Discovery order: RenderDoc's manifest comes between the driver's layers and Bandicam's.
		expect(names({ env: new Set([rd.enableEnv!]) })[2]).toBe(rd.name);
	});
	test('the disable variable wins, even over a set enable variable', () => {
		const rd = IMPLICIT_LAYERS.find((l) => l.enableEnv)!;
		expect(implicitActive(rd, new Set([rd.enableEnv!, rd.disableEnv]))).toBe(false);
		expect(names({ env: new Set(['DISABLE_LAYER_NV_OPTIMUS_1']) })).not.toContain('VK_LAYER_NV_optimus');
	});
	test('the explicit layer comes after the implicit ones, and only when the app enables it', () => {
		expect(names({}).at(-1)).toBe(VALIDATION);
		expect(names({ validation: false })).not.toContain(VALIDATION);
	});
});

describe('call paths (Vulkan-Loader docs)', () => {
	test('instance calls: trampoline, every layer, terminator, then every driver at once', () => {
		const tr = run({ call: 'vkEnumeratePhysicalDevices', second: true });
		expect(tr.hops.map((h) => h.at)).toEqual([
			['app'],
			['trampoline'],
			...names({}).map((n) => [layerNode(n)]),
			['terminator'],
			['icd:nvidia', 'icd:lavapipe'],
			['app'],
		]);
		expect(tr.hops.at(-1)!.vars.n).toBe(2);
	});
	test('vkCreateInstance reaches every installed driver', () => {
		expect(run({ second: true }).hops.at(-2)!.at).toEqual(['icd:nvidia', 'icd:lavapipe']);
		expect(run({}).hops.at(-2)!.at).toEqual(['icd:nvidia']);
	});
	test('physical-device calls: the terminator calls only the owning driver', () => {
		const tr = run({ call: 'vkGetPhysicalDeviceProperties', second: true, gpu: 'lavapipe' });
		expect(tr.path.has('terminator')).toBe(true);
		expect(tr.path.has('icd:lavapipe')).toBe(true);
		expect(tr.path.has('icd:nvidia')).toBe(false);
	});
	test('device calls never pass the terminator; a vkGetDeviceProcAddr pointer skips the trampoline too', () => {
		const exp = run({ call: 'vkQueueSubmit', via: 'export' });
		expect(exp.path.has('terminator')).toBe(false);
		expect(exp.path.has('trampoline')).toBe(true);
		const gdpa = run({ call: 'vkQueueSubmit', via: 'gdpa' });
		expect(gdpa.path.has('trampoline')).toBe(false);
		expect(gdpa.hops[1]!.at).toEqual([`layer:${names({})[0]}`]);
	});
	test('the GPU choice falls back to the real driver when there is no second one', () => {
		const tr = run({ call: 'vkQueueSubmit', second: false, gpu: 'lavapipe' });
		expect(tr.path.has('icd:nvidia')).toBe(true);
		expect(tr.icds).toEqual(['nvidia']);
	});
	test('with no layers at all the trampoline hands straight to the terminator', () => {
		const env = new Set(['DISABLE_LAYER_NV_OPTIMUS_1', 'DISABLE_LAYER_NV_PRESENT_1', 'VK_LAYER_bandicam_helper_DEBUG_1']);
		const tr = run({ call: 'vkEnumeratePhysicalDevices', env, validation: false });
		expect(tr.layers).toEqual([]);
		expect(tr.hops.map((h) => h.at[0])).toEqual(['app', 'trampoline', 'terminator', 'icd:nvidia', 'app']);
	});
});
