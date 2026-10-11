/**
 * Model of the memory-path lab: where a buffer lives on the course GPU (src/data/gpu.json, from vulkaninfo) and what
 * moving it costs per frame. Three paths: the GPU reads VRAM, the GPU reads host-visible system RAM over PCIe, the
 * CPU writes into the BAR window (VRAM the CPU can map). Each path picks its memory type with the spec's
 * find-memory-type rule (lib/labs/memory.ts) over the real memory types and the real memoryTypeBits of a buffer;
 * times use theoretical peak bandwidths only. Pure module: no DOM, no React, tested with `bun test`.
 */
import reqJson from '../../data/buffer-requirements.json';
import { DL, GiB, HC, HEAPS, HV, MiB, findType, type MemHeap, type MemType } from './memory';

/**
 * Theoretical peak bandwidths, bytes/s (GB = 10^9 bytes), the numbers quoted in lesson `gpu`:
 * RTX 3080 Ti GDDR6X: 19 Gbps × 384-bit bus / 8 = 912 GB/s (NVIDIA spec);
 * PCIe 4.0 x16: 16 GT/s × 16 lanes × 128/130 encoding / 8 ≈ 31.5 GB/s per direction, rounded to 32 as in the lesson.
 */
export const BW = { vram: 912e9, pcie: 32e9 } as const;

/** One frame at 60 fps, ms. */
export const FRAME_MS = 1000 / 60;

/** Data sizes on the slider: powers of two from 1 MiB to 1 GiB. */
export const SIZES: readonly number[] = Array.from({ length: 11 }, (_, i) => MiB * 2 ** i);

export const PATHS = ['vram', 'sysram', 'bar'] as const;
export type PathId = (typeof PATHS)[number];

/** Boxes of the picture, in drawing order; the links join neighbours. */
export const NODES = ['ram', 'cpu', 'pcie', 'gpu', 'vram'] as const;
export type NodeId = (typeof NODES)[number];

export interface PathDef {
	/** Memory property flags the path requires of the memory type. */
	need: readonly string[];
	/** Peak bandwidth of the slowest link on the way. */
	bw: number;
	/** Boxes the data passes through, in order. */
	route: readonly NodeId[];
}

export const PATH: Readonly<Record<PathId, PathDef>> = {
	vram: { need: [DL], bw: BW.vram, route: ['gpu', 'vram'] },
	// The GPU reads system RAM by DMA through the CPU's memory controller (the PCIe root complex is in the CPU).
	sysram: { need: [HV, HC], bw: BW.pcie, route: ['ram', 'cpu', 'pcie', 'gpu'] },
	// CPU stores go through the PCIe link into the part of VRAM mapped by the GPU's BAR.
	bar: { need: [DL, HV], bw: BW.pcie, route: ['cpu', 'pcie', 'gpu', 'vram'] },
};

/**
 * memoryTypeBits of the buffer each path allocates, from the real capture (src/data/buffer-requirements.json): every
 * usage combination reports the same mask on this GPU (59 = types 0, 1, 3, 4, 5; type 2 is never allowed for a
 * buffer), so one value serves every path. Throws if a recapture reports different masks for different usages.
 */
export const BUFFER_TYPE_BITS: number = (() => {
	const masks = new Set(reqJson.buffers.map((b) => b.memoryTypeBits));
	const [bits] = masks;
	if (masks.size !== 1 || bits === undefined) throw new Error(`memory-path: buffers report different memoryTypeBits: ${[...masks].join(', ')}`);
	return bits;
})();

/** The rule an allocation larger than its heap breaks; its text comes from `vuid()` (src/lib/vuids.ts). */
export const VUID_HEAP_SIZE = 'VUID-vkAllocateMemory-pAllocateInfo-01713';

export interface Transfer {
	type: MemType;
	heap: MemHeap;
	/** The allocation fits the heap (the whole heap: other allocations would only make it worse). */
	fits: boolean;
	/** Time to move `size` bytes once at the path's peak bandwidth, ms. */
	ms: number;
	/** ms / FRAME_MS: 1 = the whole 60 fps frame. */
	budget: number;
}

/** What moving `size` bytes along `path` costs on the course GPU. Throws if no type the buffer allows has the flags. */
export function transfer(path: PathId, size: number): Transfer {
	const p = PATH[path];
	const type = findType(BUFFER_TYPE_BITS, p.need);
	if (!type) throw new Error(`memory-path: no memory type allowed by ${BUFFER_TYPE_BITS} has ${p.need.join(' | ')}`);
	const heap = HEAPS[type.heapIndex];
	if (!heap) throw new Error(`memory-path: memory type ${type.index} points at missing heap ${type.heapIndex}`);
	const ms = (size / p.bw) * 1000;
	return { type, heap, fits: size <= heap.size, ms, budget: ms / FRAME_MS };
}

/** Bytes as the lab prints them in `lang`: MiB below 1 GiB, GiB from there (binary units, like vulkaninfo). */
export function bytes(n: number, lang = 'en'): string {
	const [v, unit, digits] = n >= GiB ? [n / GiB, 'GiB', 2] : [n / MiB, 'MiB', n < 10 * MiB ? 1 : 0];
	return `${new Intl.NumberFormat(lang, { maximumFractionDigits: digits }).format(v)} ${unit}`;
}

/** Vulkan flag without its prefix and `_BIT` (`DEVICE_LOCAL`), for compact lists. */
export function shortFlag(f: string): string {
	return f.replace(/^VK_MEMORY_(PROPERTY|HEAP)_/, '').replace(/_BIT$/, '');
}
