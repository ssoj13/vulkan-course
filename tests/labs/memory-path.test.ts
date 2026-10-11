import { describe, expect, test } from 'bun:test';
import gpu from '../../src/data/gpu.json';
import { BW, FRAME_MS, GiB, HEAPS, MiB, NODES, PATH, PATHS, SIZES, TYPES, bytes, findType, shortFlag, transfer } from '../../src/lib/labs/memory-path';
import { textErrors } from '../../src/lib/text';
import { TEXT } from '../../src/components/labs/memory-path.text';

const has = (flags: readonly string[], f: string) => flags.includes(`VK_MEMORY_PROPERTY_${f}_BIT`);

describe('memory-path model', () => {
	test('heaps and types come from gpu.json', () => {
		expect(HEAPS.map((h) => h.size)).toEqual(gpu.memory.heaps.map((h) => h.size));
		expect(TYPES.map((t) => t.heapIndex)).toEqual(gpu.memory.types.map((t) => t.heapIndex));
	});

	test('find-memory-type: the first type, in index order, with every required flag', () => {
		for (const p of PATHS) {
			const t = findType(PATH[p].need)!;
			expect(PATH[p].need.every((f) => t.flags.includes(f))).toBe(true);
			for (const earlier of TYPES.slice(0, t.index)) expect(PATH[p].need.every((f) => earlier.flags.includes(f))).toBe(false);
		}
		expect(findType(['VK_MEMORY_PROPERTY_PROTECTED_BIT'])).toBeNull();
	});

	test('each path lands where it should on the course GPU', () => {
		const vram = transfer('vram', MiB);
		expect(has(vram.type.flags, 'DEVICE_LOCAL') && !has(vram.type.flags, 'HOST_VISIBLE')).toBe(true);
		expect(vram.heap.flags).toContain('VK_MEMORY_HEAP_DEVICE_LOCAL_BIT');

		const sys = transfer('sysram', MiB);
		expect(has(sys.type.flags, 'DEVICE_LOCAL')).toBe(false);
		expect(sys.heap.flags).not.toContain('VK_MEMORY_HEAP_DEVICE_LOCAL_BIT');

		// Without Resizable BAR the CPU-visible part of VRAM is a small separate heap (214 MiB here).
		const bar = transfer('bar', MiB);
		expect(has(bar.type.flags, 'DEVICE_LOCAL') && has(bar.type.flags, 'HOST_VISIBLE')).toBe(true);
		expect(bar.heap.size).toBeLessThan(vram.heap.size);
		expect(Math.round(bar.heap.size / MiB)).toBe(214);
	});

	test('time = bytes / peak bandwidth, measured against a 60 fps frame', () => {
		const t = transfer('vram', GiB);
		expect(t.ms).toBeCloseTo((GiB / BW.vram) * 1000);
		expect(t.budget).toBeCloseTo(t.ms / FRAME_MS);
		// The same bytes cost 912/32 times more across PCIe.
		expect(transfer('sysram', GiB).ms / t.ms).toBeCloseTo(912 / 32);
		expect(FRAME_MS).toBeCloseTo(16.67, 2);
		// 1 GiB over PCIe at 32 GB/s takes ~33.6 ms: two whole frames.
		expect(transfer('sysram', GiB).budget).toBeGreaterThan(2);
	});

	test('an allocation larger than its heap does not fit', () => {
		const heap = transfer('bar', MiB).heap.size;
		expect(transfer('bar', SIZES.findLast((s) => s <= heap)!).fits).toBe(true);
		expect(transfer('bar', SIZES.find((s) => s > heap)!).fits).toBe(false);
		expect(transfer('vram', GiB).fits).toBe(true);
	});

	test('sizes, routes, formatting', () => {
		expect(SIZES[0]).toBe(MiB);
		expect(SIZES.at(-1)).toBe(GiB);
		for (const p of PATHS) {
			// A route is a contiguous run of boxes, so the lit links between them are well defined.
			const idx = PATH[p].route.map((n) => NODES.indexOf(n)).sort((a, b) => a - b);
			idx.forEach((v, i) => expect(v).toBe(idx[0]! + i));
		}
		expect(bytes(MiB)).toBe('1 MiB');
		expect(bytes(224395264)).toBe('214 MiB');
		expect(bytes(GiB)).toBe('1 GiB');
		expect(bytes(1.5 * GiB, 'ru')).toBe('1,5 GiB');
		expect(shortFlag('VK_MEMORY_PROPERTY_HOST_COHERENT_BIT')).toBe('HOST_COHERENT');
		expect(shortFlag('VK_MEMORY_HEAP_DEVICE_LOCAL_BIT')).toBe('DEVICE_LOCAL');
	});

	test('texts', () => {
		expect(textErrors(TEXT, 'memory-path')).toEqual([]);
	});
});
