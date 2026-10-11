import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import gpu from '../../src/data/gpu.json';
import reqJson from '../../src/data/buffer-requirements.json';
import {
	COPY_FAMILIES,
	HC,
	HV,
	LIMITS,
	PLACEMENTS,
	REQUESTS,
	TYPES,
	USAGES,
	VUIDS,
	WANT,
	bitString,
	bytes,
	chooseType,
	flow,
	requirements,
	typesIn,
	type Usage,
} from '../../src/lib/labs/buffer-memory';
import { textErrors } from '../../src/lib/text';
import { TEXT } from '../../src/components/labs/buffer-memory.text';

const FLAG_BITS: Record<string, number> = {
	VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: 1,
	VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: 2,
	VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: 4,
	VK_MEMORY_PROPERTY_HOST_CACHED_BIT: 8,
};
const V: Usage = 'VK_BUFFER_USAGE_VERTEX_BUFFER_BIT';
const DST: Usage = 'VK_BUFFER_USAGE_TRANSFER_DST_BIT';

describe('buffer-requirements.json (real capture)', () => {
	test('same device and memory types as gpu.json', () => {
		expect(reqJson.meta.device).toBe(gpu.device.name);
		expect(reqJson.meta.driverVersionRaw).toBe(gpu.device.driverVersionRaw);
		expect(reqJson.meta.memoryTypePropertyFlags).toEqual(gpu.memory.types.map((t) => t.propertyFlags.reduce((m, f) => m | FLAG_BITS[f]!, 0)));
	});

	test('every non-empty combination of the six flags, every size; size = request rounded up to the alignment', () => {
		expect(reqJson.buffers).toHaveLength(2 ** USAGES.length - 1);
		for (const b of reqJson.buffers) {
			expect(b.sizes).toHaveLength(REQUESTS.length);
			b.sizes.forEach((s, i) => expect(s).toBe(Math.ceil(REQUESTS[i]! / b.alignment) * b.alignment));
		}
	});

	test('the facts the lab states: alignment 4 / 16 / 64, and memoryTypeBits never allows type 2', () => {
		const align = (u: Usage[]) => requirements(u, 1).alignment;
		expect(align(['VK_BUFFER_USAGE_INDEX_BUFFER_BIT'])).toBe(4);
		expect(align([V])).toBe(16);
		expect(align(['VK_BUFFER_USAGE_STORAGE_BUFFER_BIT'])).toBe(16);
		for (const b of reqJson.buffers) {
			expect(b.alignment).toBe(b.usage.includes('VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT') ? LIMITS.minUniformBufferOffsetAlignment : b.usage.length === 1 && b.usage[0] === 'VK_BUFFER_USAGE_INDEX_BUFFER_BIT' ? 4 : 16);
			expect(typesIn(b.memoryTypeBits)).toEqual([0, 1, 3, 4, 5]);
		}
		expect(bitString(59)).toBe('111011');
		// Usage order does not matter for the lookup.
		expect(requirements([DST, V], 1000)).toEqual(requirements([V, DST], 1000));
		expect(() => requirements([V], 12345)).toThrow();
	});
});

describe('choosing a memory type', () => {
	test('each placement lands on the plainest fitting type of the course GPU', () => {
		const picked = Object.fromEntries(PLACEMENTS.map((p) => [p, chooseType(59, p)]));
		expect(picked.device!.picked).toBe(1);
		expect(picked.upload!.picked).toBe(3);
		expect(picked.bar!.picked).toBe(5);
		expect(picked.readback!.picked).toBe(4);
		for (const p of PLACEMENTS) expect(picked[p]!.optimal).toBe(true);
	});

	test('memoryTypeBits filters before the flags; without the optimal flags the required ones decide', () => {
		// Only type 3 allowed: a BAR request falls back to plain upload memory.
		const c = chooseType(1 << 3, 'bar');
		expect(c.picked).toBe(3);
		expect(c.optimal).toBe(false);
		expect(c.rows.filter((r) => r.allowed).map((r) => r.type.index)).toEqual([3]);
		expect(chooseType(1 << 1, 'upload').picked).toBeNull();
		for (const p of PLACEMENTS) expect(WANT[p].optimal).toEqual(expect.arrayContaining([...WANT[p].required]));
	});

	test('every HOST_VISIBLE type is HOST_COHERENT on this GPU (the lab says flushes do not apply here)', () => {
		for (const t of TYPES.filter((x) => x.flags.includes(HV))) expect(t.flags).toContain(HC);
	});

	test('the BAR heap is the small device-local heap (ReBAR off): 214 MiB', () => {
		const bar = TYPES[chooseType(59, 'bar').picked!]!;
		expect(bytes(gpu.memory.heaps[bar.heapIndex]!.size)).toBe('214 MiB');
	});
});

describe('the flow', () => {
	const kinds = (u: Usage[], p: (typeof PLACEMENTS)[number]) => flow(u, 65536, p).map((s) => `${s.kind}${'which' in s ? `:${s.which}` : ''}`);

	test('device-local: own setup, staging setup, map staging, copy, sync', () => {
		expect(kinds([V, DST], 'device')).toEqual([
			'create:dst', 'reqs:dst', 'choose:dst', 'allocate:dst', 'bind:dst',
			'create:staging', 'reqs:staging', 'choose:staging', 'allocate:staging', 'bind:staging', 'map:staging', 'copy', 'sync',
		]);
		const sync = flow([V, DST], 65536, 'device').at(-1)!;
		expect(sync.kind === 'sync' && sync.consumers).toEqual([V]);
	});

	test('host-visible placements map the buffer itself: no staging', () => {
		for (const p of ['upload', 'bar', 'readback'] as const) expect(kinds([V], p)).toEqual(['create:dst', 'reqs:dst', 'choose:dst', 'allocate:dst', 'bind:dst', 'map:dst']);
	});

	test('a copy into a buffer without TRANSFER_DST is the real validation error, and the flow stops there', () => {
		const f = flow([V], 1000, 'device');
		const last = f.at(-1)!;
		expect(last.kind === 'copy' && last.ok).toBe(false);
		expect(VUIDS.copyDst.id).toBe('VUID-vkCmdCopyBuffer-dstBuffer-00120');
	});

	test('the staging buffer is TRANSFER_SRC in upload memory, and family 1 is the dedicated copy family', () => {
		const st = flow([V, DST], 100, 'device').find((s) => s.kind === 'create' && s.which === 'staging');
		expect(st?.kind === 'create' && st.usage).toEqual(['VK_BUFFER_USAGE_TRANSFER_SRC_BIT']);
		expect(COPY_FAMILIES.filter((f) => f.dedicated).map((f) => f.index)).toEqual([1]);
	});

	const registry = process.env.VULKAN_SDK ? join(process.env.VULKAN_SDK, 'share', 'vulkan', 'registry', 'validusage.json') : '';
	test.skipIf(!registry || !existsSync(registry))('quoted VUID texts match the SDK registry', async () => {
		const db = (await Bun.file(registry).json()) as { validation: Record<string, Record<string, { vuid: string; text: string }[]>> };
		const index = new Map<string, string>();
		for (const byExt of Object.values(db.validation)) for (const list of Object.values(byExt)) for (const v of list) index.set(v.vuid, v.text);
		const plain = (t: string) => t.replace(/<[^>]+>/g, '').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&').replace(/\s+/g, ' ').trim();
		for (const v of Object.values(VUIDS)) expect(plain(index.get(v.id) ?? '')).toBe(v.text);
	});
});

test('texts: every language has the same keys and placeholders', () => {
	expect(textErrors(TEXT, 'buffer-memory')).toEqual([]);
});
