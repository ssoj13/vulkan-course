import { describe, expect, test } from 'bun:test';
import { TEXT } from '../../src/components/labs/loader-chain.text';
import { CALLS, IMPLICIT_LAYERS, VALIDATION, activeLayers, implicitActive, layerNode, trace, type Setup } from '../../src/lib/labs/loader-chain';
import { textErrors } from '../../src/lib/text';

const base: Setup = { call: 'vkCreateInstance', env: new Map(), validation: true, second: false, gpu: 'nvidia', via: 'gdpa' };
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
		const on = new Map([[rd.enableEnv!.name, '1']]);
		expect(rd.enableEnv!.value).toBe('1');
		expect(implicitActive(rd, new Map())).toBe(false);
		expect(implicitActive(rd, on)).toBe(true);
		// Discovery order: RenderDoc's manifest comes between the driver's layers and Bandicam's.
		expect(names({ env: on })[2]).toBe(rd.name);
	});
	test('the enable variable must hold the manifest value exactly (loader.c compares with strcmp)', () => {
		for (const l of IMPLICIT_LAYERS.filter((l) => l.enableEnv)) {
			expect(implicitActive(l, new Map([[l.enableEnv!.name, 'true']]))).toBe(false);
			expect(implicitActive(l, new Map([[l.enableEnv!.name, '1 ']]))).toBe(false);
			expect(implicitActive(l, new Map([[l.enableEnv!.name, l.enableEnv!.value]]))).toBe(true);
		}
	});
	test('the disable variable wins with any value, even over a matching enable variable', () => {
		const rd = IMPLICIT_LAYERS.find((l) => l.enableEnv)!;
		expect(implicitActive(rd, new Map([[rd.enableEnv!.name, '1'], [rd.disableEnv, 'yes']]))).toBe(false);
		expect(names({ env: new Map([['DISABLE_LAYER_NV_OPTIMUS_1', '0']]) })).not.toContain('VK_LAYER_NV_optimus');
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
	const bare = new Map(['DISABLE_LAYER_NV_OPTIMUS_1', 'DISABLE_LAYER_NV_PRESENT_1', 'VK_LAYER_bandicam_helper_DEBUG_1'].map((n) => [n, '1']));
	test('with no layers at all the trampoline hands an instance call straight to the terminator', () => {
		const tr = run({ call: 'vkEnumeratePhysicalDevices', env: bare, validation: false });
		expect(tr.layers).toEqual([]);
		expect(tr.hops.map((h) => h.at[0])).toEqual(['app', 'trampoline', 'terminator', 'icd:nvidia', 'app']);
	});
	test('with no layers a device call goes from the trampoline, or the program itself, straight into the driver', () => {
		const exp = run({ call: 'vkQueueSubmit', env: bare, validation: false, via: 'export' });
		expect(exp.hops.map((h) => [h.at[0], h.key])).toEqual([
			['app', 'capAppSubmitExport'],
			['trampoline', 'capTrampDeviceDirect'],
			['icd:nvidia', 'capIcdSubmitTramp'],
			['app', 'capReturn'],
		]);
		const gdpa = run({ call: 'vkQueueSubmit', env: bare, validation: false, via: 'gdpa' });
		expect(gdpa.hops.map((h) => [h.at[0], h.key])).toEqual([
			['app', 'capAppSubmitGdpaDirect'],
			['icd:nvidia', 'capIcdSubmitDirect'],
			['app', 'capReturn'],
		]);
		// No caption of a layer-less device call talks about a layer.
		for (const h of [...exp.hops, ...gdpa.hops]) expect(TEXT.en[h.key]).not.toMatch(/layer calls|first layer/);
	});
});
