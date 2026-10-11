/**
 * Model of the buffer-memory lab: the real allocation flow of a VkBuffer on the course GPU.
 * vkCreateBuffer -> vkGetBufferMemoryRequirements (REAL values captured on the dev machine, src/data/
 * buffer-requirements.json) -> pick a memory type with the spec's find-memory-type rule (lib/labs/memory.ts) over the
 * REAL memory types (src/data/gpu.json) -> vkAllocateMemory -> vkBindBufferMemory -> map (host-visible) or a staging upload
 * (device-local only); a readback buffer instead gets the GPU-writes-then-CPU-reads path (GPU write, HOST barrier, fence,
 * optional invalidate, CPU read). Pure module: no DOM, no React, tested with `bun test`.
 */
import gpu from '../../data/gpu.json';
import reqJson from '../../data/buffer-requirements.json';
import { DL, GiB, HC, HCACHED, HEAPS, HV, MiB, TYPES, allows, findType, hasFlags, type MemType } from './memory';

export const USAGES = [
	'VK_BUFFER_USAGE_VERTEX_BUFFER_BIT',
	'VK_BUFFER_USAGE_INDEX_BUFFER_BIT',
	'VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT',
	'VK_BUFFER_USAGE_STORAGE_BUFFER_BIT',
	'VK_BUFFER_USAGE_TRANSFER_SRC_BIT',
	'VK_BUFFER_USAGE_TRANSFER_DST_BIT',
] as const;
export type Usage = (typeof USAGES)[number];

const SRC: Usage = 'VK_BUFFER_USAGE_TRANSFER_SRC_BIT';
const DST: Usage = 'VK_BUFFER_USAGE_TRANSFER_DST_BIT';
const STORAGE: Usage = 'VK_BUFFER_USAGE_STORAGE_BUFFER_BIT';

/** Buffer sizes the capture asked for (bytes); the lab offers exactly these, so every number shown is measured. */
export const REQUESTS: readonly number[] = reqJson.meta.requests;

export interface Req {
	size: number;
	alignment: number;
	memoryTypeBits: number;
}

/** Order-independent key of a usage set. */
const key = (u: readonly string[]) => [...u].sort().join('|');
const REQ_ROWS = new Map(reqJson.buffers.map((b) => [key(b.usage), b]));

/** Captured VkMemoryRequirements of a buffer with this usage and requested size. Throws for a combination not captured. */
export function requirements(usage: readonly Usage[], request: number): Req {
	const row = REQ_ROWS.get(key(usage));
	const i = REQUESTS.indexOf(request);
	const size = row?.sizes[i];
	if (!row || size === undefined) throw new Error(`buffer-memory: no capture for [${usage.join(', ')}] at ${request} bytes`);
	return { size, alignment: row.alignment, memoryTypeBits: row.memoryTypeBits };
}

/** Limits that decide alignment around buffers and mappings (gpu.json). */
export const LIMITS = {
	minMemoryMapAlignment: gpu.limits.minMemoryMapAlignment,
	nonCoherentAtomSize: gpu.limits.nonCoherentAtomSize,
	bufferImageGranularity: gpu.limits.bufferImageGranularity,
	minUniformBufferOffsetAlignment: gpu.limits.minUniformBufferOffsetAlignment,
	minStorageBufferOffsetAlignment: gpu.limits.minStorageBufferOffsetAlignment,
} as const;

/** Where the reader wants the buffer to live: the property flags it would like, and the ones it cannot do without. */
export const PLACEMENTS = ['device', 'upload', 'bar', 'readback'] as const;
export type Placement = (typeof PLACEMENTS)[number];

export const WANT: Readonly<Record<Placement, { optimal: readonly string[]; required: readonly string[] }>> = {
	// GPU-only data (meshes, textures' buffers): VRAM, filled through a staging copy.
	device: { optimal: [DL], required: [DL] },
	// Written by the CPU, read by the GPU over PCIe (staging buffers, per-frame data).
	upload: { optimal: [HV, HC], required: [HV] },
	// VRAM the CPU can write directly (the BAR window); falls back to plain upload memory if a GPU has none.
	bar: { optimal: [DL, HV, HC], required: [HV] },
	// Written by the GPU, read back by the CPU: cached host memory reads fast on the CPU.
	readback: { optimal: [HV, HCACHED], required: [HV] },
};

export interface TypeRow {
	type: MemType;
	/** Bit `index` of memoryTypeBits is set: the buffer may live in this type. */
	allowed: boolean;
	hasOptimal: boolean;
	hasRequired: boolean;
}

export interface Choice {
	rows: TypeRow[];
	/** Picked memory type index, or null when no allowed type has even the required flags. */
	picked: number | null;
	/** The pick has every optimal flag (otherwise it is the fallback on the required ones). */
	optimal: boolean;
}

/**
 * The spec's find-memory-type rule (`findType`), run the way the spec's sample does: first with the optimal flags,
 * then, if no allowed type has them, with the required ones.
 */
export function chooseType(bits: number, p: Placement): Choice {
	const w = WANT[p];
	const rows = TYPES.map((type) => ({ type, allowed: allows(bits, type), hasOptimal: hasFlags(type, w.optimal), hasRequired: hasFlags(type, w.required) }));
	const opt = findType(bits, w.optimal);
	const pick = opt ?? findType(bits, w.required);
	return { rows, picked: pick?.index ?? null, optimal: !!opt };
}

/** Valid-usage rules the lab names; their texts come from `vuid()` (src/lib/vuids.ts). */
export const VUIDS = {
	copyDst: 'VUID-vkCmdCopyBuffer-dstBuffer-00120',
	bindType: 'VUID-vkBindBufferMemory-memory-01035',
	mapHostVisible: 'VUID-vkMapMemory-memory-00682',
	flushAtom: 'VUID-VkMappedMemoryRange-offset-00687',
} as const;

const COPY_ONLY = new Set(['VK_QUEUE_TRANSFER_BIT', 'VK_QUEUE_SPARSE_BINDING_BIT']);

/**
 * Queue families of the course GPU that can run vkCmdCopyBuffer: TRANSFER, or GRAPHICS / COMPUTE, which imply it.
 * `dedicated` = a copy engine and nothing else (no graphics, compute or video): the DMA queues.
 */
export const COPY_FAMILIES: readonly { index: number; count: number; dedicated: boolean }[] = gpu.queueFamilies
	.filter((q) => q.queueFlags.some((f) => f === 'VK_QUEUE_TRANSFER_BIT' || f === 'VK_QUEUE_GRAPHICS_BIT' || f === 'VK_QUEUE_COMPUTE_BIT'))
	.map((q) => ({ index: q.index, count: q.queueCount, dedicated: q.queueFlags.every((f) => COPY_ONLY.has(f)) }));

/**
 * Who reads the buffer after the upload in a typical renderer: the sync2 stages and access a barrier after the copy
 * makes the copy's writes visible to.
 */
export const CONSUMER: Readonly<Partial<Record<Usage, { stages: readonly string[]; access: string }>>> = {
	VK_BUFFER_USAGE_VERTEX_BUFFER_BIT: { stages: ['VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT'], access: 'VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT' },
	VK_BUFFER_USAGE_INDEX_BUFFER_BIT: { stages: ['VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT'], access: 'VK_ACCESS_2_INDEX_READ_BIT' },
	VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT: {
		stages: ['VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT', 'VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT'],
		access: 'VK_ACCESS_2_UNIFORM_READ_BIT',
	},
	VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: { stages: ['VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT'], access: 'VK_ACCESS_2_SHADER_STORAGE_READ_BIT' },
};

/** How the GPU writes a readback buffer: a copy into it (TRANSFER_DST) or a compute shader store (STORAGE_BUFFER). */
export type Writer = 'copy' | 'dispatch';

/** The sync2 stage and access of a GPU write: the first scope of the barrier that hands the data to the host. */
export const WRITER: Readonly<Record<Writer, { stage: string; access: string }>> = {
	copy: { stage: 'VK_PIPELINE_STAGE_2_COPY_BIT', access: 'VK_ACCESS_2_TRANSFER_WRITE_BIT' },
	dispatch: { stage: 'VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT', access: 'VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT' },
};

/** Which buffer a step is about: the reader's buffer, or the staging buffer of an upload. */
export type Which = 'dst' | 'staging';

export type FlowStep =
	| { kind: 'create'; which: Which; usage: readonly Usage[]; request: number }
	| { kind: 'reqs'; which: Which; req: Req }
	| { kind: 'choose'; which: Which; placement: Placement; bits: number; choice: Choice }
	| { kind: 'allocate'; which: Which; type: MemType; size: number; heapSize: number }
	| { kind: 'bind'; which: Which; alignment: number; size: number }
	| { kind: 'map'; which: Which; type: MemType; request: number }
	| { kind: 'copy'; request: number; ok: boolean }
	| { kind: 'sync'; consumers: readonly Usage[] }
	/** Readback: the GPU writes the buffer; null = the usage allows no GPU write at all. */
	| { kind: 'gpuWrite'; writer: Writer | null; request: number }
	/** Readback: barrier from the GPU write to HOST / HOST_READ (a fence alone does not make device writes host-visible). */
	| { kind: 'hostSync'; writer: Writer }
	/** Readback: vkQueueSubmit2 with a fence, then vkWaitForFences on the host. */
	| { kind: 'fence' }
	/** Readback: the CPU reads through the mapping; `coherent` = no vkInvalidateMappedMemoryRanges needed first. */
	| { kind: 'read'; type: MemType; request: number; coherent: boolean };

/**
 * create -> requirements -> choose -> allocate -> bind for one buffer. Every placement has a type on the course GPU
 * (tests check it), so no fitting type is a data error, not a lab state.
 */
function setup(which: Which, usage: readonly Usage[], request: number, placement: Placement): { steps: FlowStep[]; type: MemType } {
	const req = requirements(usage, request);
	const choice = chooseType(req.memoryTypeBits, placement);
	const steps: FlowStep[] = [
		{ kind: 'create', which, usage, request },
		{ kind: 'reqs', which, req },
		{ kind: 'choose', which, placement, bits: req.memoryTypeBits, choice },
	];
	if (choice.picked === null) throw new Error(`buffer-memory: no memory type for ${placement} in memoryTypeBits ${req.memoryTypeBits}`);
	const type = TYPES[choice.picked]!;
	steps.push(
		{ kind: 'allocate', which, type, size: req.size, heapSize: HEAPS[type.heapIndex]!.size },
		{ kind: 'bind', which, alignment: req.alignment, size: req.size },
	);
	return { steps, type };
}

/**
 * The whole flow as timeline frames. A host-visible pick ends with a map + memcpy; a device-local pick that the CPU
 * cannot map gets a staging upload: staging buffer (TRANSFER_SRC) in upload memory, map + memcpy, vkCmdCopyBuffer,
 * then synchronization before the GPU reads it. Usage must not be empty (the view keeps at least one flag).
 */
export function flow(usage: readonly Usage[], request: number, placement: Placement): FlowStep[] {
	const main = setup('dst', usage, request, placement);
	if (placement === 'readback') return [...main.steps, ...readback(usage, request, main.type)];
	if (main.type.flags.includes(HV)) return [...main.steps, { kind: 'map', which: 'dst', type: main.type, request }];
	const st = setup('staging', [SRC], request, 'upload');
	const ok = usage.includes(DST);
	const steps: FlowStep[] = [
		...main.steps,
		...st.steps,
		{ kind: 'map', which: 'staging', type: st.type, request },
		{ kind: 'copy', request, ok },
	];
	if (ok) steps.push({ kind: 'sync', consumers: usage.filter((u) => u in CONSUMER) });
	return steps;
}

/**
 * GPU writes, CPU reads back: the GPU writes the buffer (a copy if it is TRANSFER_DST, else a compute store if it is
 * STORAGE_BUFFER), a barrier makes the write available to the host domain (dst HOST / HOST_READ), the submit carries a
 * fence the host waits on, and the CPU reads through the mapping, invalidating first unless the type is HOST_COHERENT.
 * A usage the GPU cannot write at all stops the flow at the write.
 */
function readback(usage: readonly Usage[], request: number, type: MemType): FlowStep[] {
	const writer: Writer | null = usage.includes(DST) ? 'copy' : usage.includes(STORAGE) ? 'dispatch' : null;
	if (!writer) return [{ kind: 'gpuWrite', writer, request }];
	return [
		{ kind: 'gpuWrite', writer, request },
		{ kind: 'hostSync', writer },
		{ kind: 'fence' },
		{ kind: 'read', type, request, coherent: type.flags.includes(HC) },
	];
}

/** The API name a step stands for (language-neutral labels of the step strip). */
export function stepApi(s: FlowStep): string {
	switch (s.kind) {
		case 'create':
			return 'vkCreateBuffer';
		case 'reqs':
			return 'vkGetBufferMemoryRequirements';
		case 'choose':
			return 'memoryTypeIndex';
		case 'allocate':
			return 'vkAllocateMemory';
		case 'bind':
			return 'vkBindBufferMemory';
		case 'map':
			return 'vkMapMemory';
		case 'copy':
			return 'vkCmdCopyBuffer';
		case 'sync':
		case 'hostSync':
			return 'vkCmdPipelineBarrier2';
		case 'gpuWrite':
			return s.writer === 'dispatch' ? 'vkCmdDispatch' : 'vkCmdCopyBuffer';
		case 'fence':
			return 'vkWaitForFences';
		case 'read':
			return s.coherent ? 'memcpy' : 'vkInvalidateMappedMemoryRanges';
	}
}

/** Indices of the memory types a memoryTypeBits mask allows. */
export const typesIn = (bits: number): number[] => TYPES.filter((t) => allows(bits, t)).map((t) => t.index);

/** Vulkan flag without its prefix and `_BIT`, for compact lists: `HOST_VISIBLE`, `VERTEX_BUFFER`. */
export const short = (f: string): string => f.replace(/^VK_(MEMORY_PROPERTY|MEMORY_HEAP|BUFFER_USAGE|QUEUE)_/, '').replace(/_BIT(_KHR|_NV)?$/, '');

/** Bytes as the lab prints them: exact below 1 KiB-ish sizes, binary units above (like vulkaninfo). */
export function bytes(n: number): string {
	if (n >= GiB) return `${+(n / GiB).toFixed(2)} GiB`;
	if (n >= MiB) return `${+(n / MiB).toFixed(2)} MiB`;
	if (n >= 1024 && n % 1024 === 0) return `${n / 1024} KiB`;
	return `${n.toLocaleString('en-US')} B`;
}

/** memoryTypeBits as a binary string, type 0 on the right like in the spec's bit numbering. */
export const bitString = (bits: number): string => bits.toString(2).padStart(TYPES.length, '0');
