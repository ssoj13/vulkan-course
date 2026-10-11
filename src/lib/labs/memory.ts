/**
 * Device memory of the course GPU, shared by the memory labs (memory-path, buffer-memory): the heaps and memory types
 * of src/data/gpu.json (from vulkaninfo) and the spec's find-memory-type rule, implemented once here.
 * Pure module: no DOM, no React, tested with `bun test`.
 */
import gpu from '../../data/gpu.json';

export const MiB = 1024 ** 2;
export const GiB = 1024 ** 3;

export const DL = 'VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT';
export const HV = 'VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT';
export const HC = 'VK_MEMORY_PROPERTY_HOST_COHERENT_BIT';
export const HCACHED = 'VK_MEMORY_PROPERTY_HOST_CACHED_BIT';

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

/** Bit `index` of a resource's memoryTypeBits is set: the resource may live in this type. */
export const allows = (bits: number, t: MemType): boolean => (bits & (1 << t.index)) !== 0;

/** The type has every one of `flags`. */
export const hasFlags = (t: MemType, flags: readonly string[]): boolean => flags.every((f) => t.flags.includes(f));

/**
 * The spec's find-memory-type rule (chapter "Device Memory", sample `findProperties`): the first type, in index order,
 * that the resource's memoryTypeBits allows and that has every wanted flag. The spec orders the types (a type whose
 * flags are a strict subset of another's comes first) so that this simple loop finds the plainest fitting type.
 * Returns null when no allowed type has the flags.
 */
export function findType(bits: number, flags: readonly string[], types: readonly MemType[] = TYPES): MemType | null {
	return types.find((t) => allows(bits, t) && hasFlags(t, flags)) ?? null;
}
