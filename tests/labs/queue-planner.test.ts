import { describe, expect, test } from 'bun:test';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { TEXT } from '../../src/components/labs/queue-planner.text';
import {
	DEFAULT_OPTIONS,
	DEFAULT_REQS,
	FAMILIES,
	PRIORITY_LEVELS,
	VUIDS,
	caps,
	familyFor,
	plan,
	type Family,
	type Role,
	type RoleReq,
} from '../../src/lib/labs/queue-planner';
import { textErrors } from '../../src/lib/text';

const reqs = (over: Partial<Record<Role, Partial<RoleReq>>> = {}) =>
	Object.fromEntries(Object.entries(DEFAULT_REQS).map(([r, q]) => [r, { ...q, ...over[r as Role] }])) as Record<Role, RoleReq>;

describe('queue families of the course GPU', () => {
	test('match the lesson table and gpu.json', () => {
		expect(FAMILIES.map((f) => [f.queueCount, f.flags])).toEqual([
			[16, ['GRAPHICS', 'COMPUTE', 'TRANSFER', 'SPARSE_BINDING']],
			[2, ['TRANSFER', 'SPARSE_BINDING']],
			[8, ['COMPUTE', 'TRANSFER', 'SPARSE_BINDING']],
			[1, ['TRANSFER', 'SPARSE_BINDING', 'VIDEO_DECODE']],
			[1, ['TRANSFER', 'SPARSE_BINDING', 'VIDEO_ENCODE']],
			[1, ['TRANSFER', 'SPARSE_BINDING', 'OPTICAL_FLOW']],
		]);
		// The page text says: 32 timestamp bits on the video families, 64 elsewhere; (1,1,1) granularity everywhere.
		expect(FAMILIES.map((f) => f.timestampValidBits)).toEqual([64, 64, 64, 32, 32, 64]);
		expect(FAMILIES.every((f) => f.granularity.join() === '1,1,1')).toBe(true);
		expect(PRIORITY_LEVELS).toBe(2);
	});

	test('step 04: graphics 0, dedicated transfer 1, async compute 2; first-with-flag puts everything on 0', () => {
		expect(familyFor(FAMILIES, 'graphics', 'dedicated')).toBe(0);
		expect(familyFor(FAMILIES, 'transfer', 'dedicated')).toBe(1);
		expect(familyFor(FAMILIES, 'compute', 'dedicated')).toBe(2);
		expect(familyFor(FAMILIES, 'decode', 'dedicated')).toBe(3);
		expect(familyFor(FAMILIES, 'transfer', 'first')).toBe(0);
		expect(familyFor(FAMILIES, 'compute', 'first')).toBe(0);
		expect(familyFor(FAMILIES, 'present', 'first')).toBe(0);
		expect(familyFor(FAMILIES, 'present', 'dedicated')).toBe(0);
	});

	test('graphics/compute families copy without reporting TRANSFER (spec: the bit is optional for them)', () => {
		const fams: Family[] = [
			{ index: 0, flags: ['GRAPHICS', 'COMPUTE'], queueCount: 1, timestampValidBits: 64, granularity: [1, 1, 1] },
			{ index: 1, flags: ['VIDEO_DECODE'], queueCount: 1, timestampValidBits: 0, granularity: [0, 0, 0] },
		];
		expect(caps(fams[0]!)).toContain('TRANSFER');
		expect(familyFor(fams, 'transfer', 'dedicated')).toBe(0);
		expect(familyFor(fams, 'compute', 'dedicated')).toBe(0);
		// A family list without graphics, compute or family 0 (present): missing capabilities are reported, not invented.
		expect(familyFor([fams[1]!], 'graphics', 'first')).toBeNull();
		const p = plan([fams[1]!], reqs(), DEFAULT_OPTIONS);
		expect(p.errors.filter((e) => e.kind === 'noFamily').map((e) => e.kind === 'noFamily' && e.role)).toEqual(['graphics', 'present', 'compute', 'transfer']);
	});
});

describe('plan', () => {
	test('defaults: one create info per family; graphics and present share queue 0 of family 0', () => {
		const p = plan(FAMILIES, reqs(), DEFAULT_OPTIONS);
		expect(p.errors).toEqual([]);
		expect(p.infos).toEqual([
			{ family: 0, count: 1, priorities: [1], roles: ['graphics', 'present'] },
			{ family: 1, count: 1, priorities: [0.5], roles: ['transfer'] },
			{ family: 2, count: 1, priorities: [0.5], roles: ['compute'] },
		]);
		expect(p.roles.find((r) => r.role === 'present')).toEqual({ role: 'present', family: 0, queues: [0], withGraphics: true });
		expect(p.roles.find((r) => r.role === 'compute')!.withGraphics).toBe(false);
	});

	test('first-with-flag merges everything into family 0; without sharing the queues are consecutive', () => {
		const p = plan(FAMILIES, reqs(), { strategy: 'first', share: false, merge: true });
		expect(p.infos).toEqual([{ family: 0, count: 4, priorities: [1, 1, 0.5, 0.5], roles: ['graphics', 'present', 'compute', 'transfer'] }]);
		expect(p.roles.map((r) => r.queues)).toEqual([[0], [1], [2], [3]]);
		expect(p.roles.filter((r) => r.withGraphics).map((r) => r.role)).toEqual(['present', 'compute', 'transfer']);
	});

	test('sharing: a family gets as many queues as its hungriest role; a shared queue gets the highest priority', () => {
		const p = plan(FAMILIES, reqs({ graphics: { count: 2, priority: 0.5 }, present: { priority: 1 } }), DEFAULT_OPTIONS);
		expect(p.infos[0]).toEqual({ family: 0, count: 2, priorities: [1, 0.5], roles: ['graphics', 'present'] });
	});

	test('no merging: two create infos for family 0 break VUID 02802', () => {
		const p = plan(FAMILIES, reqs(), { ...DEFAULT_OPTIONS, merge: false });
		expect(p.infos.map((i) => i.family)).toEqual([0, 0, 2, 1]);
		expect(p.errors).toEqual([{ kind: 'duplicate', vuid: 'VUID-VkDeviceCreateInfo-queueFamilyIndex-02802', family: 0, infos: 2 }]);
	});

	test('more queues than the family has breaks VUID 00382', () => {
		const p = plan(FAMILIES, reqs({ transfer: { count: 3 }, decode: { on: true, count: 2 } }), DEFAULT_OPTIONS);
		expect(p.errors).toEqual([
			{ kind: 'tooMany', vuid: 'VUID-VkDeviceQueueCreateInfo-queueCount-00382', family: 1, count: 3, max: 2 },
			{ kind: 'tooMany', vuid: 'VUID-VkDeviceQueueCreateInfo-queueCount-00382', family: 3, count: 2, max: 1 },
		]);
		expect(plan(FAMILIES, reqs({ graphics: { count: 16 } }), DEFAULT_OPTIONS).errors).toEqual([]);
		expect(plan(FAMILIES, reqs({ graphics: { count: 16 } }), { ...DEFAULT_OPTIONS, share: false }).errors[0]).toMatchObject({ family: 0, count: 17, max: 16 });
	});

	test('a priority outside 0..1 breaks VUID 00383', () => {
		const p = plan(FAMILIES, reqs({ compute: { priority: 1.5 } }), DEFAULT_OPTIONS);
		expect(p.errors).toEqual([{ kind: 'priority', vuid: 'VUID-VkDeviceQueueCreateInfo-pQueuePriorities-00383', family: 2, value: 1.5 }]);
	});

	test('disabled roles create nothing', () => {
		const off = { on: false };
		const p = plan(FAMILIES, reqs({ present: off, compute: off, transfer: off }), DEFAULT_OPTIONS);
		expect(p.infos).toEqual([{ family: 0, count: 1, priorities: [1], roles: ['graphics'] }]);
	});
});

test('quoted VUID texts are verbatim from the SDK registry (skipped without the SDK)', () => {
	const sdk = process.env.VULKAN_SDK ?? 'C:/Programs/VulkanSDK/1.4.341.1';
	const path = join(sdk, 'share', 'vulkan', 'registry', 'validusage.json');
	if (!existsSync(path)) return;
	const plain = (t: string) => t.replace(/<[^>]+>/g, '').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&').replace(/\s+/g, ' ').trim();
	const db = JSON.parse(readFileSync(path, 'utf8')) as { validation: Record<string, Record<string, { vuid: string; text: string }[]>> };
	const index = new Map<string, string>();
	for (const byExt of Object.values(db.validation)) for (const list of Object.values(byExt)) for (const v of list) index.set(v.vuid, plain(v.text));
	for (const [vuid, text] of Object.entries(VUIDS)) expect(index.get(vuid)).toBe(text);
});

test('texts: every language has the same keys and placeholders', () => {
	expect(textErrors(TEXT, 'queue-planner')).toEqual([]);
});
