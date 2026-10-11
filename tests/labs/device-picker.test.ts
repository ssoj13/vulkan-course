import { describe, expect, test } from 'bun:test';
import gpu from '../../src/data/gpu.json';
import { TEXT } from '../../src/components/labs/device-picker.text';
import {
	CANDIDATES,
	EXPLAINED,
	STEP03_RULE,
	binarySize,
	featureGroups,
	judge,
	limitRows,
	nvidiaDriver,
	pick,
	pickerErrors,
	reaches,
	usable,
	type Candidate,
} from '../../src/lib/labs/device-picker';
import { textErrors } from '../../src/lib/text';

const only = (...ids: string[]) => CANDIDATES.filter((c) => ids.includes(c.id));

describe('properties of the course GPU', () => {
	test('the data the lab needs is there', () => {
		expect(pickerErrors()).toEqual([]);
	});

	test('step 03 prints driver 0x9a100000; NVIDIA packing decodes it to the version vulkaninfo shows', () => {
		expect(gpu.device.driverVersionRaw.toString(16)).toBe('9a100000');
		expect(nvidiaDriver(gpu.device.driverVersionRaw)).toBe(gpu.device.driverVersion);
	});

	test('versions', () => {
		expect(reaches('1.4.351', 4)).toBe(true);
		expect(reaches('1.3.0', 4)).toBe(false);
		expect(reaches('1.3.0', 3)).toBe(true);
		expect(usable(4, '1.3.0')).toBe('1.3');
		expect(usable(4, gpu.device.apiVersion)).toBe('1.4');
	});
});

describe('the picking rule of step 03', () => {
	test('with every candidate the real discrete GPU wins on score', () => {
		const p = pick(CANDIDATES, STEP03_RULE);
		expect(p.picked).toBe('real');
		const real = p.verdicts.find((v) => v.id === 'real')!;
		expect(real.eligible && real.score).toBe(1000);
		// The 1.3 software rasterizer is rejected by the version check before any feature is asked.
		expect(p.verdicts.find((v) => v.id === 'cpu')!.checks).toEqual([{ kind: 'version', ok: false, have: '1.3.0', minor: 4 }]);
	});

	test('without the type score the first eligible device in enumeration order wins', () => {
		const p = pick(CANDIDATES, { ...STEP03_RULE, preferDiscrete: false });
		expect(p.picked).toBe('igpu');
		expect(p.tiedLater).toEqual(['real']);
	});

	test('no eligible device: nothing is picked (step 03 exits with an error)', () => {
		expect(pick(only('cpu'), STEP03_RULE).picked).toBeNull();
		expect(pick(only('cpu'), { ...STEP03_RULE, minMinor: 3 }).picked).toBe('cpu');
	});

	test('a feature of a struct newer than the device is not asked: it fails as n/a', () => {
		const v = judge(only('cpu')[0]!, { minMinor: 3, required: ['dynamicRendering', 'pushDescriptor'], preferDiscrete: true });
		expect(v.eligible).toBe(false);
		expect(v.checks.at(-1)).toEqual({ kind: 'feature', ok: false, name: 'pushDescriptor', value: 'na' });
	});

	test('a false feature rejects; ties keep the first', () => {
		const a: Candidate = { ...CANDIDATES[1]!, id: 'a' };
		const b: Candidate = { ...a, id: 'b' };
		const c: Candidate = { ...a, id: 'c', features: { ...a.features, synchronization2: false } };
		const p = pick([c, a, b], STEP03_RULE);
		expect(p.picked).toBe('a');
		expect(p.tiedLater).toEqual(['b']);
		expect(p.verdicts[0]!.eligible).toBe(false);
	});

	test('hypothetical 1.4/1.3 devices respect the spec\'s mandatory features', () => {
		for (const c of CANDIDATES) {
			if (reaches(c.apiVersion, 3)) expect(c.features.dynamicRendering && c.features.synchronization2).toBe(true);
			if (reaches(c.apiVersion, 4)) expect(Object.values(c.features).every(Boolean)).toBe(true);
		}
		expect(CANDIDATES.filter((c) => !c.hypothetical).map((c) => c.name)).toEqual([gpu.device.name]);
	});
});

describe('features and limits tabs', () => {
	test('feature structs, counts and highlights', () => {
		const all = featureGroups('');
		expect(all.map((g) => g.total)).toEqual(Object.values(gpu.features).map((m) => Object.keys(m).length));
		expect(all[0]!.supported).toBe(53);
		const marked = all.flatMap((g) => g.rows.filter((r) => r.step03).map((r) => r.name));
		// gpu.json keeps each struct's members sorted.
		expect(marked).toEqual(['samplerAnisotropy', 'dynamicRendering', 'synchronization2', 'maintenance5', 'pushDescriptor']);
		expect(all.flatMap((g) => g.rows.filter((r) => r.pickRule).map((r) => r.name))).toEqual(['dynamicRendering', 'synchronization2']);
		expect(featureGroups('ASTC').flatMap((g) => g.rows.map((r) => r.name))).toEqual(['textureCompressionASTC_LDR', 'textureCompressionASTC_HDR']);
		expect(featureGroups('', true).flatMap((g) => g.rows.map((r) => r.name))).toEqual([
			'textureCompressionASTC_LDR',
			'textureCompressionETC2',
			'protectedMemory',
			'storageInputOutput16',
			'textureCompressionASTC_HDR',
		]);
	});

	test('limits: explained first, values as step 03 prints them', () => {
		const rows = limitRows('');
		expect(rows.length).toBe(Object.keys(gpu.limits).length);
		expect(rows.slice(0, EXPLAINED.length).map((r) => r.name)).toEqual(EXPLAINED.map((e) => e.name));
		const by = (n: string) => rows.find((r) => r.name === n)!;
		expect(by('maxImageDimension2D').value).toBe('32768');
		expect(by('maxPushConstantsSize').value).toBe('256');
		expect(by('minUniformBufferOffsetAlignment').value).toBe('64');
		expect(by('timestampPeriod').value).toBe('1');
		expect(by('maxComputeSharedMemorySize').size).toBe('48 KiB');
		expect(by('maxComputeWorkGroupSize').value).toBe('1024, 1024, 64');
		expect(by('framebufferColorSampleCounts').value).toBe('1, 2, 4, 8');
		expect(limitRows('offsetalign', true).map((r) => r.name)).toEqual(['minUniformBufferOffsetAlignment', 'minStorageBufferOffsetAlignment']);
		expect(binarySize(4294967295)).toBeNull();
		expect(binarySize(65536)).toBe('64 KiB');
	});
});

test('texts: every language has the same keys and placeholders, every explained limit and every unnamed device has a text', () => {
	expect(textErrors(TEXT, 'device-picker')).toEqual([]);
	for (const e of EXPLAINED) expect(TEXT.en).toHaveProperty(`L_${e.name}`);
	for (const c of CANDIDATES.filter((x) => !x.name)) expect(TEXT.en).toHaveProperty(`cand_${c.id}`);
});
