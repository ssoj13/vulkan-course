/**
 * Model of the memory-path lab: where a buffer lives on the course GPU (src/data/gpu.json, from vulkaninfo) and what
 * moving it costs per frame. Three paths: the GPU reads VRAM, the GPU reads host-visible system RAM over PCIe, the
 * CPU writes into the BAR window (VRAM the CPU can map). Each path picks its memory type with the spec's
 * find-memory-type rule over the real memory types; times use theoretical peak bandwidths only.
 * Pure module: no DOM, no React, tested with `bun test`.
 */
import gpu from '../../data/gpu.json';

export const MiB = 1024 ** 2;
export const GiB = 1024 ** 3;

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

const DL = 'VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT';
const HV = 'VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT';
const HC = 'VK_MEMORY_PROPERTY_HOST_COHERENT_BIT';

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

export interface MemType {
	index: number;
	heapIndex: number;
	flags: readonly string[];
}
export interface MemHeap {
	index: number;
	size: number;
	flags: readonly string[];
}

export const HEAPS: readonly MemHeap[] = gpu.memory.heaps.map((h, index) => ({ index, size: h.size, flags: h.flags }));
export const TYPES: readonly MemType[] = gpu.memory.types.map((t, index) => ({ index, heapIndex: t.heapIndex, flags: t.propertyFlags }));

/**
 * The spec's find-memory-type rule (chapter "Device Memory", sample `findProperties`): the first type, in index
 * order, that has every required flag. The spec orders the types (a type whose flags are a strict subset of another's
 * comes first) so that this simple loop finds the plainest fitting type. A real buffer would also pass its
 * `memoryTypeBits`; the lab allows every type. Returns null when no type has the flags.
 */
export function findType(need: readonly string[], types: readonly MemType[] = TYPES): MemType | null {
	return types.find((t) => need.every((f) => t.flags.includes(f))) ?? null;
}

/** VUID-vkAllocateMemory-pAllocateInfo-01713, plain text of the SDK 1.4.341 registry (validusage.json). */
export const VUID_HEAP_SIZE = {
	id: 'VUID-vkAllocateMemory-pAllocateInfo-01713',
	text:
		'pAllocateInfo->allocationSize must be less than or equal to VkPhysicalDeviceMemoryProperties::memoryHeaps[memindex].size ' +
		'where memindex = VkPhysicalDeviceMemoryProperties::memoryTypes[pAllocateInfo->memoryTypeIndex].heapIndex as returned by ' +
		'vkGetPhysicalDeviceMemoryProperties for the VkPhysicalDevice that device was created from',
} as const;

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

/** What moving `size` bytes along `path` costs on the course GPU. Throws if gpu.json has no fitting type. */
export function transfer(path: PathId, size: number): Transfer {
	const p = PATH[path];
	const type = findType(p.need);
	if (!type) throw new Error(`memory-path: gpu.json has no memory type with ${p.need.join(' | ')}`);
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
