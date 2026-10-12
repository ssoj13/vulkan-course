/**
 * Model of the "barriers" trainer: short command streams on ONE queue where two commands touch the same buffer or
 * image, and the reader writes the synchronization2 barrier between them (stage masks, access masks and, for images,
 * the two layouts). Submitting runs the rules the validation layer applies (refusals with the real VUID texts) and the
 * hazard check synchronization validation applies (accepted, but the hazard is reported and the puzzle stays open);
 * a barrier that closes every hazard is graded tight (no flag could go) or wider than needed.
 *
 * Rules and their sources (vkspec 1.4, "Synchronization and Cache Control"; VUID texts only through `vuReport`):
 * - Execution scopes: "If a synchronization command includes a source stage mask, its first synchronization scope only
 *   includes execution of the pipeline stages specified in that mask and any logically earlier stages"; the second
 *   scope: the destination stages "and any logically later stages". Logical order = the spec's lists for the graphics
 *   primitive pipeline (DRAW_INDIRECT, INDEX_INPUT, VERTEX_ATTRIBUTE_INPUT, VERTEX_SHADER, ..., EARLY_FRAGMENT_TESTS,
 *   FRAGMENT_SHADER, LATE_FRAGMENT_TESTS, COLOR_ATTACHMENT_OUTPUT) and the compute pipeline (DRAW_INDIRECT,
 *   COMPUTE_SHADER); the transfer stages COPY/BLIT/RESOLVE/CLEAR are parts of one TRANSFER stage and order nothing
 *   among themselves. TOP_OF_PIPE "is equivalent to VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT with VkAccessFlags2 set to 0
 *   when specified in the second synchronization scope, but equivalent to VK_PIPELINE_STAGE_2_NONE in the first
 *   scope"; BOTTOM_OF_PIPE the other way round. Meta stages: VERTEX_INPUT, ALL_TRANSFER, ALL_GRAPHICS, ALL_COMMANDS are
 *   the OR of their parts.
 * - Access scopes: "Its first access scope only includes memory accesses performed by pipeline stages explicitly
 *   specified in the source stage mask" (same for the second), and "access scopes do not interact with the logically
 *   earlier or later stages for either scope". MEMORY_READ/MEMORY_WRITE are "equivalent to setting all READ [WRITE]
 *   access flags that are valid where it is used"; SHADER_READ = SAMPLED_READ | STORAGE_READ (| binding table, tile
 *   reads: not modeled); SHADER_WRITE = STORAGE_WRITE.
 * - Hazards: "Write-after-read hazards can be solved with just an execution dependency, but read-after-write and
 *   write-after-write hazards need appropriate memory dependencies to be included between them." So an earlier write
 *   must be in the first execution AND access scope, a later access in the second execution scope and, after a write,
 *   in the second access scope; after a read only the execution scopes matter.
 * - Layout transitions: "Image layout transitions may perform read and write accesses on all memory bound to the image
 *   subresource range, so applications must ensure that all memory writes have been made available before a layout
 *   transition is executed. Available memory is automatically made visible to a layout transition, and writes performed
 *   by a layout transition are automatically made available." A transition is a write between the two scopes: earlier
 *   accesses face it (WAW/WAR), later accesses face its write (RAW/WAW). With synchronization2, equal old and new layouts
 *   are no transition: "the layout values are ignored and the image contents are preserved regardless of the values of
 *   oldLayout, newLayout, and the current layout of the image". Old layout UNDEFINED: "the contents of that range may be
 *   discarded" (accepted, but where the later command needs the contents the puzzle stays open).
 * - Render passes face outside work only through their load and store operations ("external synchronization with
 *   attachment access only needs to synchronize the load operations with previous commands"; "... in an earlier render
 *   pass only needs to synchronize with the store operations"). Depth/stencil load ops run in EARLY_FRAGMENT_TESTS, store
 *   ops in LATE_FRAGMENT_TESTS, color ones in COLOR_ATTACHMENT_OUTPUT; LOAD_OP_CLEAR and STORE_OP_STORE/DONT_CARE are
 *   writes (DEPTH_STENCIL_ATTACHMENT_WRITE / COLOR_ATTACHMENT_WRITE).
 * - Valid usage checked at submit (validusage.json, SDK 1.4.341): every access flag needs a stage that can perform it,
 *   VUID-Vk{Memory,Image}Barrier2-{src,dst}AccessMask-039xx/07454 (MEMORY_READ/WRITE need none); layout transitions
 *   VUID-VkImageMemoryBarrier2-oldLayout-01197 (old = UNDEFINED or the current layout), -newLayout-01198, and the usage
 *   a layout needs (-oldLayout-01208/01211/01212/01213, -srcQueueFamilyIndex-04066); the later command finds the image
 *   in the layout it names (VUID-vkCmdDraw-None-09600, VUID-VkCopyBufferToImageInfo2-dstImageLayout-00180,
 *   VUID-vkCmdBeginRendering-pRenderingInfo-09588).
 *
 * Grading of a correct barrier: tight = no flag can be removed (and no meta flag replaced by a smaller part of it)
 * without a hazard or a refusal coming back; otherwise the trainer names the flags that are not needed.
 *
 * Simplifications (stated in Barriers.astro's caption): one queue, commands in submission order, no host accesses, no
 * queue family ownership transfers, whole resources (no ranges or subresources); each command is reduced to the
 * accesses that face the barrier (a render pass to its load and store operations). Hazards are the spec's rules above;
 * the layer's own messages are quoted where a real run of a course step produced them (`Puzzle.evidence`). Pure: tested in
 * tests/labs/barriers.test.ts.
 */
import { vuReport } from './vuids';

/** Pipeline stages offered (sync2 names without VK_PIPELINE_STAGE_2_ and _BIT); meta stages included. */
export const STAGES = [
	'TOP_OF_PIPE',
	'DRAW_INDIRECT',
	'INDEX_INPUT',
	'VERTEX_ATTRIBUTE_INPUT',
	'VERTEX_INPUT',
	'VERTEX_SHADER',
	'EARLY_FRAGMENT_TESTS',
	'FRAGMENT_SHADER',
	'LATE_FRAGMENT_TESTS',
	'COLOR_ATTACHMENT_OUTPUT',
	'COMPUTE_SHADER',
	'COPY',
	'BLIT',
	'RESOLVE',
	'CLEAR',
	'ALL_TRANSFER',
	'ALL_GRAPHICS',
	'ALL_COMMANDS',
	'BOTTOM_OF_PIPE',
] as const;
export type Stage = (typeof STAGES)[number];

/** Access types offered (without VK_ACCESS_2_ and _BIT). */
export const ACCESSES = [
	'INDIRECT_COMMAND_READ',
	'INDEX_READ',
	'VERTEX_ATTRIBUTE_READ',
	'UNIFORM_READ',
	'SHADER_SAMPLED_READ',
	'SHADER_STORAGE_READ',
	'SHADER_STORAGE_WRITE',
	'SHADER_READ',
	'SHADER_WRITE',
	'COLOR_ATTACHMENT_READ',
	'COLOR_ATTACHMENT_WRITE',
	'DEPTH_STENCIL_ATTACHMENT_READ',
	'DEPTH_STENCIL_ATTACHMENT_WRITE',
	'TRANSFER_READ',
	'TRANSFER_WRITE',
	'MEMORY_READ',
	'MEMORY_WRITE',
] as const;
export type Access = (typeof ACCESSES)[number];

export const LAYOUTS = ['UNDEFINED', 'GENERAL', 'COLOR_ATTACHMENT_OPTIMAL', 'DEPTH_ATTACHMENT_OPTIMAL', 'SHADER_READ_ONLY_OPTIMAL', 'TRANSFER_SRC_OPTIMAL', 'TRANSFER_DST_OPTIMAL'] as const;
export type Layout = (typeof LAYOUTS)[number];

export type Usage = 'COLOR_ATTACHMENT' | 'DEPTH_STENCIL_ATTACHMENT' | 'SAMPLED' | 'INPUT_ATTACHMENT' | 'TRANSFER_SRC' | 'TRANSFER_DST';

export const stageName = (s: Stage) => `VK_PIPELINE_STAGE_2_${s}_BIT`;
export const accessName = (a: Access) => `VK_ACCESS_2_${a}_BIT`;
export const layoutName = (l: Layout) => `VK_IMAGE_LAYOUT_${l}`;

// --- Stage order and expansion ------------------------------------------------------------------------------------

/** Logical order of the graphics primitive pipeline (stages this model knows). */
const GRAPHICS: readonly Stage[] = ['DRAW_INDIRECT', 'INDEX_INPUT', 'VERTEX_ATTRIBUTE_INPUT', 'VERTEX_SHADER', 'EARLY_FRAGMENT_TESTS', 'FRAGMENT_SHADER', 'LATE_FRAGMENT_TESTS', 'COLOR_ATTACHMENT_OUTPUT'];
const COMPUTE: readonly Stage[] = ['DRAW_INDIRECT', 'COMPUTE_SHADER'];
const PIPELINES = [GRAPHICS, COMPUTE];
const TRANSFER: readonly Stage[] = ['COPY', 'BLIT', 'RESOLVE', 'CLEAR'];
/** Stages commands execute in (no meta, no TOP/BOTTOM). */
export const ATOMIC: readonly Stage[] = [...GRAPHICS, 'COMPUTE_SHADER', ...TRANSFER];
/** Meta stages: the OR of their parts. */
const META_STAGE: Partial<Record<Stage, readonly Stage[]>> = {
	VERTEX_INPUT: ['INDEX_INPUT', 'VERTEX_ATTRIBUTE_INPUT'],
	ALL_TRANSFER: TRANSFER,
	ALL_GRAPHICS: GRAPHICS,
	ALL_COMMANDS: ATOMIC,
};

export type Side = 'src' | 'dst';

/** Stages a mask names explicitly (metas expanded); TOP/BOTTOM name none (they carry "VkAccessFlags2 set to 0"). */
function named(mask: readonly Stage[]): Set<Stage> {
	const out = new Set<Stage>();
	for (const s of mask) {
		if (s === 'TOP_OF_PIPE' || s === 'BOTTOM_OF_PIPE') continue;
		for (const x of META_STAGE[s] ?? [s]) out.add(x);
	}
	return out;
}

/** Execution scope of a stage mask: named stages, TOP (dst) / BOTTOM (src) = everything, plus logical order. */
export function execScope(mask: readonly Stage[], side: Side): Set<Stage> {
	if (mask.includes(side === 'src' ? 'BOTTOM_OF_PIPE' : 'TOP_OF_PIPE')) return new Set(ATOMIC);
	const out = named(mask);
	for (const p of PIPELINES) {
		const at = p.map((s) => out.has(s));
		// src: a named stage pulls in every logically earlier one; dst: every logically later one.
		const reach = side === 'src' ? at.lastIndexOf(true) : at.indexOf(true);
		if (reach < 0) continue;
		for (const [i, s] of p.entries()) if (side === 'src' ? i <= reach : i >= reach) out.add(s);
	}
	return out;
}

// --- Access types --------------------------------------------------------------------------------------------------

const READS: readonly Access[] = ['INDIRECT_COMMAND_READ', 'INDEX_READ', 'VERTEX_ATTRIBUTE_READ', 'UNIFORM_READ', 'SHADER_SAMPLED_READ', 'SHADER_STORAGE_READ', 'COLOR_ATTACHMENT_READ', 'DEPTH_STENCIL_ATTACHMENT_READ', 'TRANSFER_READ'];
const WRITES: readonly Access[] = ['SHADER_STORAGE_WRITE', 'COLOR_ATTACHMENT_WRITE', 'DEPTH_STENCIL_ATTACHMENT_WRITE', 'TRANSFER_WRITE'];
const META_ACCESS: Partial<Record<Access, readonly Access[]>> = {
	SHADER_READ: ['SHADER_SAMPLED_READ', 'SHADER_STORAGE_READ'],
	SHADER_WRITE: ['SHADER_STORAGE_WRITE'],
	MEMORY_READ: READS,
	MEMORY_WRITE: WRITES,
};
/** Access types a mask covers (metas expanded). */
export function accessScope(mask: readonly Access[]): Set<Access> {
	return new Set(mask.flatMap((a) => META_ACCESS[a] ?? [a]));
}

const SHADER_STAGES: readonly Stage[] = ['VERTEX_SHADER', 'FRAGMENT_SHADER', 'COMPUTE_SHADER'];
const SHADER_ACCESS = { stages: ['ALL_GRAPHICS', 'ALL_COMMANDS', ...SHADER_STAGES] as readonly Stage[] };
/**
 * Stages that may carry each access flag, with the VUID number that says so (validusage.json, VkMemoryBarrier2 and
 * VkImageMemoryBarrier2 share the numbers). MEMORY_READ/WRITE are "always valid in any access mask".
 */
export const SUPPORT: Readonly<Record<Access, { vu: string; stages: readonly Stage[] } | null>> = {
	INDIRECT_COMMAND_READ: { vu: '03900', stages: ['DRAW_INDIRECT', 'ALL_GRAPHICS', 'ALL_COMMANDS'] },
	INDEX_READ: { vu: '03901', stages: ['INDEX_INPUT', 'VERTEX_INPUT', 'ALL_GRAPHICS', 'ALL_COMMANDS'] },
	VERTEX_ATTRIBUTE_READ: { vu: '03902', stages: ['VERTEX_ATTRIBUTE_INPUT', 'VERTEX_INPUT', 'ALL_GRAPHICS', 'ALL_COMMANDS'] },
	UNIFORM_READ: { vu: '03904', ...SHADER_ACCESS },
	SHADER_SAMPLED_READ: { vu: '03905', ...SHADER_ACCESS },
	SHADER_STORAGE_READ: { vu: '03906', ...SHADER_ACCESS },
	SHADER_STORAGE_WRITE: { vu: '03907', ...SHADER_ACCESS },
	SHADER_READ: { vu: '07454', ...SHADER_ACCESS },
	SHADER_WRITE: { vu: '03909', ...SHADER_ACCESS },
	COLOR_ATTACHMENT_READ: { vu: '03910', stages: ['COLOR_ATTACHMENT_OUTPUT', 'ALL_GRAPHICS', 'ALL_COMMANDS'] },
	COLOR_ATTACHMENT_WRITE: { vu: '03911', stages: ['COLOR_ATTACHMENT_OUTPUT', 'ALL_GRAPHICS', 'ALL_COMMANDS'] },
	DEPTH_STENCIL_ATTACHMENT_READ: { vu: '03912', stages: ['EARLY_FRAGMENT_TESTS', 'LATE_FRAGMENT_TESTS', 'ALL_GRAPHICS', 'ALL_COMMANDS'] },
	DEPTH_STENCIL_ATTACHMENT_WRITE: { vu: '03913', stages: ['EARLY_FRAGMENT_TESTS', 'LATE_FRAGMENT_TESTS', 'ALL_GRAPHICS', 'ALL_COMMANDS'] },
	TRANSFER_READ: { vu: '03914', stages: ['COPY', 'BLIT', 'RESOLVE', 'ALL_TRANSFER', 'ALL_COMMANDS'] },
	TRANSFER_WRITE: { vu: '03915', stages: ['COPY', 'BLIT', 'RESOLVE', 'CLEAR', 'ALL_TRANSFER', 'ALL_COMMANDS'] },
	MEMORY_READ: null,
	MEMORY_WRITE: null,
};

/**
 * Every access-mask VUID `accessVuids` can report, written out: scripts/vuid-refs.ts finds ids by their literal text,
 * so `bun run sync:vuids` copies exactly these (the numbers come from `SUPPORT`).
 */
export const ACCESS_VUIDS: ReadonlySet<string> = new Set([
	'VUID-VkMemoryBarrier2-srcAccessMask-03900', 'VUID-VkMemoryBarrier2-srcAccessMask-03901', 'VUID-VkMemoryBarrier2-srcAccessMask-03902', 'VUID-VkMemoryBarrier2-srcAccessMask-03904', 'VUID-VkMemoryBarrier2-srcAccessMask-03905', 'VUID-VkMemoryBarrier2-srcAccessMask-03906', 'VUID-VkMemoryBarrier2-srcAccessMask-03907', 'VUID-VkMemoryBarrier2-srcAccessMask-07454', 'VUID-VkMemoryBarrier2-srcAccessMask-03909', 'VUID-VkMemoryBarrier2-srcAccessMask-03910', 'VUID-VkMemoryBarrier2-srcAccessMask-03911', 'VUID-VkMemoryBarrier2-srcAccessMask-03912', 'VUID-VkMemoryBarrier2-srcAccessMask-03913', 'VUID-VkMemoryBarrier2-srcAccessMask-03914', 'VUID-VkMemoryBarrier2-srcAccessMask-03915',
	'VUID-VkMemoryBarrier2-dstAccessMask-03900', 'VUID-VkMemoryBarrier2-dstAccessMask-03901', 'VUID-VkMemoryBarrier2-dstAccessMask-03902', 'VUID-VkMemoryBarrier2-dstAccessMask-03904', 'VUID-VkMemoryBarrier2-dstAccessMask-03905', 'VUID-VkMemoryBarrier2-dstAccessMask-03906', 'VUID-VkMemoryBarrier2-dstAccessMask-03907', 'VUID-VkMemoryBarrier2-dstAccessMask-07454', 'VUID-VkMemoryBarrier2-dstAccessMask-03909', 'VUID-VkMemoryBarrier2-dstAccessMask-03910', 'VUID-VkMemoryBarrier2-dstAccessMask-03911', 'VUID-VkMemoryBarrier2-dstAccessMask-03912', 'VUID-VkMemoryBarrier2-dstAccessMask-03913', 'VUID-VkMemoryBarrier2-dstAccessMask-03914', 'VUID-VkMemoryBarrier2-dstAccessMask-03915',
	'VUID-VkImageMemoryBarrier2-srcAccessMask-03900', 'VUID-VkImageMemoryBarrier2-srcAccessMask-03901', 'VUID-VkImageMemoryBarrier2-srcAccessMask-03902', 'VUID-VkImageMemoryBarrier2-srcAccessMask-03904', 'VUID-VkImageMemoryBarrier2-srcAccessMask-03905', 'VUID-VkImageMemoryBarrier2-srcAccessMask-03906', 'VUID-VkImageMemoryBarrier2-srcAccessMask-03907', 'VUID-VkImageMemoryBarrier2-srcAccessMask-07454', 'VUID-VkImageMemoryBarrier2-srcAccessMask-03909', 'VUID-VkImageMemoryBarrier2-srcAccessMask-03910', 'VUID-VkImageMemoryBarrier2-srcAccessMask-03911', 'VUID-VkImageMemoryBarrier2-srcAccessMask-03912', 'VUID-VkImageMemoryBarrier2-srcAccessMask-03913', 'VUID-VkImageMemoryBarrier2-srcAccessMask-03914', 'VUID-VkImageMemoryBarrier2-srcAccessMask-03915',
	'VUID-VkImageMemoryBarrier2-dstAccessMask-03900', 'VUID-VkImageMemoryBarrier2-dstAccessMask-03901', 'VUID-VkImageMemoryBarrier2-dstAccessMask-03902', 'VUID-VkImageMemoryBarrier2-dstAccessMask-03904', 'VUID-VkImageMemoryBarrier2-dstAccessMask-03905', 'VUID-VkImageMemoryBarrier2-dstAccessMask-03906', 'VUID-VkImageMemoryBarrier2-dstAccessMask-03907', 'VUID-VkImageMemoryBarrier2-dstAccessMask-07454', 'VUID-VkImageMemoryBarrier2-dstAccessMask-03909', 'VUID-VkImageMemoryBarrier2-dstAccessMask-03910', 'VUID-VkImageMemoryBarrier2-dstAccessMask-03911', 'VUID-VkImageMemoryBarrier2-dstAccessMask-03912', 'VUID-VkImageMemoryBarrier2-dstAccessMask-03913', 'VUID-VkImageMemoryBarrier2-dstAccessMask-03914', 'VUID-VkImageMemoryBarrier2-dstAccessMask-03915',
]);

/** Usage an image needs for a layout in a transition, with the VUID (none for UNDEFINED/GENERAL). */
const LAYOUT_USAGE: Partial<Record<Layout, { vuid: string; usage: readonly Usage[] }>> = {
	COLOR_ATTACHMENT_OPTIMAL: { vuid: 'VUID-VkImageMemoryBarrier2-oldLayout-01208', usage: ['COLOR_ATTACHMENT'] },
	DEPTH_ATTACHMENT_OPTIMAL: { vuid: 'VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04066', usage: ['DEPTH_STENCIL_ATTACHMENT'] },
	SHADER_READ_ONLY_OPTIMAL: { vuid: 'VUID-VkImageMemoryBarrier2-oldLayout-01211', usage: ['SAMPLED', 'INPUT_ATTACHMENT'] },
	TRANSFER_SRC_OPTIMAL: { vuid: 'VUID-VkImageMemoryBarrier2-oldLayout-01212', usage: ['TRANSFER_SRC'] },
	TRANSFER_DST_OPTIMAL: { vuid: 'VUID-VkImageMemoryBarrier2-oldLayout-01213', usage: ['TRANSFER_DST'] },
};

// --- Puzzles -------------------------------------------------------------------------------------------------------

/** One memory access of a command, as the barrier sees it: an atomic stage and access type. */
export interface Acc {
	stage: Stage;
	access: Access;
	write: boolean;
}
export interface Cmd {
	kind: 'cmd';
	/** Stable id within the puzzle (card id suffix). */
	id: string;
	/** The call as code (language-independent). */
	call: string;
	acc: readonly Acc[];
	/** Image puzzles: the layout the command uses the image in, and the VUID that requires it. */
	layout?: Layout;
	layoutVuid?: string;
	/** Name of the command in the VUID report (`vuReport`'s call). */
	vuCall?: string;
}
export interface Slot {
	kind: 'slot';
	/** The later command needs the earlier contents (old layout UNDEFINED would throw them away). */
	keep: boolean;
}
/** A real synchronization-validation message for a hazard of this puzzle, from a run with a temporary fault. */
export interface Evidence {
	slot: number;
	hazard: Hazard;
	/** The two parties of the hazard (command ids or 'transition'), as `Found` names them. */
	before: string;
	after: string;
	/** The course step that was run with this barrier weakened or removed. */
	step: string;
	/** The layer's message (an excerpt as recorded in the course notes, handles as "..."), or null: it reported nothing. */
	text: string | null;
}
export interface Puzzle {
	id: string;
	struct: 'VkMemoryBarrier2' | 'VkImageMemoryBarrier2';
	/** Image puzzles: the image's usage flags and its layout before the first command. */
	usage?: readonly Usage[];
	initial?: Layout;
	seq: readonly (Cmd | Slot)[];
	/** Flags offered in the editor unless "show all" is on (the full lists are STAGES / ACCESSES). */
	stages: readonly Stage[];
	accesses: readonly Access[];
	layouts?: readonly Layout[];
	/** A tight solution (tests prove it tight; the trainer never shows it). */
	solution: readonly Barrier[];
	/** Course step the puzzle is based on (the situation, not always the exact barrier the step records). */
	step?: string;
	evidence?: readonly Evidence[];
}

const acc = (stage: Stage, access: Access, write: boolean): Acc => ({ stage, access, write });
const SLOT = (keep: boolean): Slot => ({ kind: 'slot', keep });
/** The puzzle's stages between the ones every editor offers (TOP/BOTTOM_OF_PIPE, ALL_GRAPHICS, ALL_COMMANDS). */
const offer = (...mid: Stage[]): Stage[] => ['TOP_OF_PIPE', ...mid, 'ALL_GRAPHICS', 'ALL_COMMANDS', 'BOTTOM_OF_PIPE'];

/** Empty masks, no transition: what a fresh barrier struct holds. */
export const barrier = (b: Partial<Barrier> = {}): Barrier => ({ srcStage: [], srcAccess: [], dstStage: [], dstAccess: [], oldLayout: 'UNDEFINED', newLayout: 'UNDEFINED', ...b });

/**
 * The puzzles, in the trainer's order. Ids are stable (stored in readers' runs). `evidence` = the first message
 * synchronization validation (VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true + SYNCVAL_SHADER_ACCESSES_HEURISTIC=true) printed
 * in real runs on the course machine (2026-10-11, RTX 3080 Ti, SDK 1.4.341.1): copies of steps 21, 22 and 26 built
 * OUTSIDE the repo with one barrier weakened by an env switch (21: barrier 1 dstAccessMask NONE; barrier 2
 * srcAccessMask NONE; 22: depth barrier src access NONE, or src stage EARLY only, or dst access NONE; 26: the WAR
 * barrier, the compute->compute barrier removed), handles replaced by 0x..., lines joined. The same probe confirmed the
 * depth puzzle's tight answer: 22 with src LATE_FRAGMENT_TESTS / DEPTH_STENCIL_ATTACHMENT_WRITE -> dst
 * EARLY_FRAGMENT_TESTS / DEPTH_STENCIL_ATTACHMENT_WRITE ran without a message.
 */
export const PUZZLES: readonly Puzzle[] = [
	{
		id: 'upload',
		struct: 'VkMemoryBarrier2',
		step: '15_staging',
		seq: [
			{ kind: 'cmd', id: 'copy', call: 'vkCmdCopyBuffer(staging -> vertex buffer)', acc: [acc('COPY', 'TRANSFER_WRITE', true)] },
			SLOT(true),
			{ kind: 'cmd', id: 'draw', call: 'vkCmdDraw', acc: [acc('VERTEX_ATTRIBUTE_INPUT', 'VERTEX_ATTRIBUTE_READ', false)] },
		],
		stages: offer('COPY', 'ALL_TRANSFER', 'VERTEX_ATTRIBUTE_INPUT', 'VERTEX_INPUT', 'VERTEX_SHADER', 'FRAGMENT_SHADER'),
		accesses: ['TRANSFER_READ', 'TRANSFER_WRITE', 'INDEX_READ', 'VERTEX_ATTRIBUTE_READ', 'UNIFORM_READ', 'MEMORY_READ', 'MEMORY_WRITE'],
		solution: [barrier({ srcStage: ['COPY'], srcAccess: ['TRANSFER_WRITE'], dstStage: ['VERTEX_ATTRIBUTE_INPUT'], dstAccess: ['VERTEX_ATTRIBUTE_READ'] })],
	},
	{
		id: 'war',
		struct: 'VkMemoryBarrier2',
		step: '26_particles',
		evidence: [
			{ slot: 0, hazard: 'WRITE_AFTER_READ', before: 'draw', after: 'dispatch', step: '26_particles', text: 'vkQueueSubmit2(): WRITE_AFTER_READ hazard detected. vkCmdDispatch (from VkCommandBuffer 0x... submitted on the current VkQueue 0x...) writes to VkBuffer 0x..., which was previously read by vkCmdDraw (from VkCommandBuffer 0x... submitted on VkQueue 0x...). The current synchronization defines the destination stage mask as VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, but to prevent this hazard, it must include VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT. Vulkan insight: an execution dependency is sufficient to prevent this hazard.' },
		],
		seq: [
			{ kind: 'cmd', id: 'draw', call: 'vkCmdDraw (frame N-1)', acc: [acc('VERTEX_ATTRIBUTE_INPUT', 'VERTEX_ATTRIBUTE_READ', false)] },
			SLOT(false),
			{ kind: 'cmd', id: 'dispatch', call: 'vkCmdDispatch (frame N)', acc: [acc('COMPUTE_SHADER', 'SHADER_STORAGE_READ', false), acc('COMPUTE_SHADER', 'SHADER_STORAGE_WRITE', true)] },
		],
		stages: offer('VERTEX_ATTRIBUTE_INPUT', 'VERTEX_SHADER', 'FRAGMENT_SHADER', 'COLOR_ATTACHMENT_OUTPUT', 'COMPUTE_SHADER'),
		accesses: ['VERTEX_ATTRIBUTE_READ', 'SHADER_STORAGE_READ', 'SHADER_STORAGE_WRITE', 'MEMORY_READ', 'MEMORY_WRITE'],
		solution: [barrier({ srcStage: ['VERTEX_ATTRIBUTE_INPUT'], dstStage: ['COMPUTE_SHADER'] })],
	},
	{
		id: 'waw',
		struct: 'VkMemoryBarrier2',
		seq: [
			{ kind: 'cmd', id: 'copy1', call: 'vkCmdCopyBuffer(staging A -> buffer)', acc: [acc('COPY', 'TRANSFER_WRITE', true)] },
			SLOT(false),
			{ kind: 'cmd', id: 'copy2', call: 'vkCmdCopyBuffer(staging B -> same bytes)', acc: [acc('COPY', 'TRANSFER_WRITE', true)] },
		],
		stages: offer('COPY', 'BLIT', 'CLEAR', 'ALL_TRANSFER', 'VERTEX_ATTRIBUTE_INPUT'),
		accesses: ['TRANSFER_READ', 'TRANSFER_WRITE', 'VERTEX_ATTRIBUTE_READ', 'MEMORY_READ', 'MEMORY_WRITE'],
		solution: [barrier({ srcStage: ['COPY'], srcAccess: ['TRANSFER_WRITE'], dstStage: ['COPY'], dstAccess: ['TRANSFER_WRITE'] })],
	},
	{
		id: 'compute',
		struct: 'VkMemoryBarrier2',
		step: '26_particles',
		evidence: [
			// Step 26's dispatch reads what the previous frame's dispatch wrote; without that barrier the layer was silent.
			{ slot: 0, hazard: 'READ_AFTER_WRITE', before: 'pass1', after: 'pass2', step: '26_particles', text: null },
		],
		seq: [
			{ kind: 'cmd', id: 'pass1', call: 'vkCmdDispatch (pass 1)', acc: [acc('COMPUTE_SHADER', 'SHADER_STORAGE_WRITE', true)] },
			SLOT(true),
			{ kind: 'cmd', id: 'pass2', call: 'vkCmdDispatch (pass 2)', acc: [acc('COMPUTE_SHADER', 'SHADER_STORAGE_READ', false)] },
		],
		stages: offer('DRAW_INDIRECT', 'VERTEX_SHADER', 'FRAGMENT_SHADER', 'COMPUTE_SHADER'),
		accesses: ['UNIFORM_READ', 'SHADER_SAMPLED_READ', 'SHADER_STORAGE_READ', 'SHADER_STORAGE_WRITE', 'SHADER_READ', 'SHADER_WRITE', 'MEMORY_READ', 'MEMORY_WRITE'],
		solution: [barrier({ srcStage: ['COMPUTE_SHADER'], srcAccess: ['SHADER_STORAGE_WRITE'], dstStage: ['COMPUTE_SHADER'], dstAccess: ['SHADER_STORAGE_READ'] })],
	},
	{
		id: 'sample',
		struct: 'VkImageMemoryBarrier2',
		step: '12_timeline',
		usage: ['COLOR_ATTACHMENT', 'SAMPLED'],
		initial: 'COLOR_ATTACHMENT_OPTIMAL',
		seq: [
			{ kind: 'cmd', id: 'render', call: 'vkCmdBeginRendering ... vkCmdEndRendering (pass 1, storeOp STORE)', acc: [acc('COLOR_ATTACHMENT_OUTPUT', 'COLOR_ATTACHMENT_WRITE', true)], layout: 'COLOR_ATTACHMENT_OPTIMAL' },
			SLOT(true),
			{
				kind: 'cmd',
				id: 'draw',
				call: 'vkCmdDraw (pass 2, samples the image)',
				acc: [acc('FRAGMENT_SHADER', 'SHADER_SAMPLED_READ', false)],
				layout: 'SHADER_READ_ONLY_OPTIMAL',
				layoutVuid: 'VUID-vkCmdDraw-None-09600',
				vuCall: 'vkCmdDraw',
			},
		],
		stages: offer('VERTEX_SHADER', 'EARLY_FRAGMENT_TESTS', 'FRAGMENT_SHADER', 'LATE_FRAGMENT_TESTS', 'COLOR_ATTACHMENT_OUTPUT'),
		accesses: ['COLOR_ATTACHMENT_READ', 'COLOR_ATTACHMENT_WRITE', 'UNIFORM_READ', 'SHADER_SAMPLED_READ', 'SHADER_STORAGE_READ', 'MEMORY_READ', 'MEMORY_WRITE'],
		layouts: ['UNDEFINED', 'GENERAL', 'COLOR_ATTACHMENT_OPTIMAL', 'SHADER_READ_ONLY_OPTIMAL', 'TRANSFER_SRC_OPTIMAL'],
		solution: [
			barrier({
				srcStage: ['COLOR_ATTACHMENT_OUTPUT'],
				srcAccess: ['COLOR_ATTACHMENT_WRITE'],
				dstStage: ['FRAGMENT_SHADER'],
				dstAccess: ['SHADER_SAMPLED_READ'],
				oldLayout: 'COLOR_ATTACHMENT_OPTIMAL',
				newLayout: 'SHADER_READ_ONLY_OPTIMAL',
			}),
		],
	},
	{
		id: 'texture',
		struct: 'VkImageMemoryBarrier2',
		step: '21_texture',
		usage: ['TRANSFER_DST', 'SAMPLED'],
		initial: 'UNDEFINED',
		evidence: [
			{ slot: 0, hazard: 'WRITE_AFTER_WRITE', before: 'transition', after: 'copy', step: '21_texture', text: 'vkCmdCopyBufferToImage2(): pCopyBufferToImageInfo WRITE_AFTER_WRITE hazard detected. vkCmdCopyBufferToImage2 writes to VkImage 0x..., which was previously written during an image layout transition initiated by vkCmdPipelineBarrier2. No sufficient synchronization is present to ensure that a write (VK_ACCESS_2_TRANSFER_WRITE_BIT) at VK_PIPELINE_STAGE_2_COPY_BIT does not conflict with a prior layout transition.' },
			{ slot: 1, hazard: 'WRITE_AFTER_WRITE', before: 'copy', after: 'transition', step: '21_texture', text: 'vkCmdPipelineBarrier2(): WRITE_AFTER_WRITE hazard detected. vkCmdPipelineBarrier2 performs image layout transition on the VkImage 0x..., which was previously written by vkCmdCopyBufferToImage2. No sufficient synchronization is present to ensure that a layout transition does not conflict with a prior write (VK_ACCESS_2_TRANSFER_WRITE_BIT) at VK_PIPELINE_STAGE_2_COPY_BIT.' },
		],
		seq: [
			SLOT(false),
			{
				kind: 'cmd',
				id: 'copy',
				call: 'vkCmdCopyBufferToImage2(staging -> texture)',
				acc: [acc('COPY', 'TRANSFER_WRITE', true)],
				layout: 'TRANSFER_DST_OPTIMAL',
				layoutVuid: 'VUID-VkCopyBufferToImageInfo2-dstImageLayout-00180',
				vuCall: 'vkCmdCopyBufferToImage2',
			},
			SLOT(true),
			{
				kind: 'cmd',
				id: 'draw',
				call: 'vkCmdDraw (samples the texture)',
				acc: [acc('FRAGMENT_SHADER', 'SHADER_SAMPLED_READ', false)],
				layout: 'SHADER_READ_ONLY_OPTIMAL',
				layoutVuid: 'VUID-vkCmdDraw-None-09600',
				vuCall: 'vkCmdDraw',
			},
		],
		stages: offer('COPY', 'BLIT', 'ALL_TRANSFER', 'VERTEX_SHADER', 'FRAGMENT_SHADER'),
		accesses: ['TRANSFER_READ', 'TRANSFER_WRITE', 'SHADER_SAMPLED_READ', 'SHADER_STORAGE_READ', 'MEMORY_READ', 'MEMORY_WRITE'],
		layouts: ['UNDEFINED', 'GENERAL', 'SHADER_READ_ONLY_OPTIMAL', 'TRANSFER_SRC_OPTIMAL', 'TRANSFER_DST_OPTIMAL'],
		solution: [
			barrier({ dstStage: ['COPY'], dstAccess: ['TRANSFER_WRITE'], oldLayout: 'UNDEFINED', newLayout: 'TRANSFER_DST_OPTIMAL' }),
			barrier({
				srcStage: ['COPY'],
				srcAccess: ['TRANSFER_WRITE'],
				dstStage: ['FRAGMENT_SHADER'],
				dstAccess: ['SHADER_SAMPLED_READ'],
				oldLayout: 'TRANSFER_DST_OPTIMAL',
				newLayout: 'SHADER_READ_ONLY_OPTIMAL',
			}),
		],
	},
	{
		id: 'depth',
		struct: 'VkImageMemoryBarrier2',
		step: '22_depth',
		usage: ['DEPTH_STENCIL_ATTACHMENT'],
		initial: 'DEPTH_ATTACHMENT_OPTIMAL',
		evidence: [
			{ slot: 0, hazard: 'WRITE_AFTER_WRITE', before: 'prev', after: 'transition', step: '22_depth', text: 'vkQueueSubmit2(): WRITE_AFTER_WRITE hazard detected. vkCmdPipelineBarrier2 (from VkCommandBuffer 0x... submitted on the current VkQueue 0x...) writes to VkImage 0x..., which was previously written by vkCmdEndRendering (from VkCommandBuffer 0x... submitted on VkQueue 0x...). No sufficient synchronization is present to ensure that a layout transition does not conflict with a prior write (VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) at VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT.' },
			{ slot: 0, hazard: 'WRITE_AFTER_WRITE', before: 'transition', after: 'next', step: '22_depth', text: 'vkCmdBeginRendering(): WRITE_AFTER_WRITE hazard detected. vkCmdBeginRendering clears pRenderingInfo.pDepthAttachment (VkImageView 0x..., loadOp VK_ATTACHMENT_LOAD_OP_CLEAR), which was previously written during an image layout transition initiated by vkCmdPipelineBarrier2. No sufficient synchronization is present to ensure that a write (VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) at VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT does not conflict with a prior layout transition.' },
		],
		seq: [
			{
				kind: 'cmd',
				id: 'prev',
				call: 'vkCmdEndRendering (frame N-1, depth storeOp DONT_CARE)',
				acc: [acc('LATE_FRAGMENT_TESTS', 'DEPTH_STENCIL_ATTACHMENT_WRITE', true)],
				layout: 'DEPTH_ATTACHMENT_OPTIMAL',
			},
			SLOT(false),
			{
				kind: 'cmd',
				id: 'next',
				call: 'vkCmdBeginRendering (frame N, depth loadOp CLEAR)',
				acc: [acc('EARLY_FRAGMENT_TESTS', 'DEPTH_STENCIL_ATTACHMENT_WRITE', true)],
				layout: 'DEPTH_ATTACHMENT_OPTIMAL',
				layoutVuid: 'VUID-vkCmdBeginRendering-pRenderingInfo-09588',
				vuCall: 'vkCmdBeginRendering',
			},
		],
		stages: offer('VERTEX_SHADER', 'EARLY_FRAGMENT_TESTS', 'FRAGMENT_SHADER', 'LATE_FRAGMENT_TESTS', 'COLOR_ATTACHMENT_OUTPUT'),
		accesses: ['COLOR_ATTACHMENT_WRITE', 'DEPTH_STENCIL_ATTACHMENT_READ', 'DEPTH_STENCIL_ATTACHMENT_WRITE', 'SHADER_SAMPLED_READ', 'MEMORY_READ', 'MEMORY_WRITE'],
		layouts: ['UNDEFINED', 'GENERAL', 'COLOR_ATTACHMENT_OPTIMAL', 'DEPTH_ATTACHMENT_OPTIMAL', 'SHADER_READ_ONLY_OPTIMAL'],
		solution: [
			barrier({
				srcStage: ['LATE_FRAGMENT_TESTS'],
				srcAccess: ['DEPTH_STENCIL_ATTACHMENT_WRITE'],
				dstStage: ['EARLY_FRAGMENT_TESTS'],
				dstAccess: ['DEPTH_STENCIL_ATTACHMENT_WRITE'],
				oldLayout: 'UNDEFINED',
				newLayout: 'DEPTH_ATTACHMENT_OPTIMAL',
			}),
		],
	},
];

export const puzzleOf = (id: string): Puzzle => {
	const p = PUZZLES.find((x) => x.id === id);
	if (!p) throw new Error(`barriers: no puzzle "${id}"`);
	return p;
};
export const slotsOf = (p: Puzzle): number => p.seq.filter((x) => x.kind === 'slot').length;
export const cmdsOf = (p: Puzzle): Cmd[] => p.seq.filter((x): x is Cmd => x.kind === 'cmd');

// --- State ---------------------------------------------------------------------------------------------------------

export interface Barrier {
	srcStage: readonly Stage[];
	srcAccess: readonly Access[];
	dstStage: readonly Stage[];
	dstAccess: readonly Access[];
	oldLayout: Layout;
	newLayout: Layout;
}
/** Result of the last accepted submit of a puzzle. */
export type Grade = 'hazard' | 'tight' | 'wide';
export interface PuzzleState {
	slots: readonly Barrier[];
	/** Verdict of the last submit, and per slot whether it left a hazard (or threw contents away). */
	grade: Grade | null;
	broken: readonly boolean[];
	/** Edited since the last submit: the verdict is stale. */
	dirty: boolean;
}
export interface BarrierState {
	/** Editors offer every stage and access, not only the puzzle's selection. */
	all: boolean;
	puzzles: Readonly<Record<string, PuzzleState>>;
}

/** The layout a slot finds the image in (the layout of the command before it, or the initial one). */
function layoutBefore(p: Puzzle, slot: number): Layout {
	let cur: Layout = p.initial ?? 'UNDEFINED';
	let k = 0;
	for (const x of p.seq) {
		if (x.kind === 'slot') {
			if (k++ === slot) return cur;
		} else if (x.layout) cur = x.layout;
	}
	return cur;
}

export const INIT: BarrierState = {
	all: false,
	puzzles: Object.fromEntries(
		PUZZLES.map((p) => {
			// A fresh struct: no stages, no accesses; images keep their layout (old = new = current: no transition).
			const slots = Array.from({ length: slotsOf(p) }, (_, k) => {
				const l = p.struct === 'VkImageMemoryBarrier2' ? layoutBefore(p, k) : 'UNDEFINED';
				return barrier({ oldLayout: l, newLayout: l });
			});
			return [p.id, { slots, grade: null, broken: slots.map(() => false), dirty: false }];
		}),
	),
};

// --- Console lines -------------------------------------------------------------------------------------------------

export type Hazard = 'READ_AFTER_WRITE' | 'WRITE_AFTER_READ' | 'WRITE_AFTER_WRITE';
/** Which part of the barrier misses an access: first/second execution scope, first/second access scope. */
export type Miss = 'exec1' | 'acc1' | 'exec2' | 'acc2';
/** Simulator remarks; the trainer words them (barriers.text.ts `m_*`). */
export type Msg = `hazard_${Miss}` | 'evidence' | 'silent' | 'discard' | 'tight' | 'wide' | `extra_${Extra['why']}` | 'noTransition' | 'transition';
export type LogKind = 'call' | 'error' | 'warn' | 'info';
/**
 * One console line: calls and validation reports verbatim English (`text`), remarks as `msg` + values (`slots` = the
 * puzzle's barrier count: the trainer names the barrier only when there are several). `vuid` = the quoted rule, `sync`
 * = the synchronization-validation hazard id.
 */
export type Line = ({ text: string } | { msg: Msg; vars: Record<string, string | number> }) & { kind: LogKind; vuid?: string; sync?: string };
export interface Step {
	state: BarrierState;
	log: Line[];
}

const msg = (kind: LogKind, m: Msg, vars: Record<string, string | number> = {}): Line => ({ kind, msg: m, vars });
const syncId = (h: Hazard) => `SYNC-HAZARD-${h.replaceAll('_', '-')}`;

/** A mask in C notation: `VK_PIPELINE_STAGE_2_COPY_BIT | ...`, or the NONE constant. */
export function maskText(mask: readonly string[], kind: 'stage' | 'access'): string {
	if (!mask.length) return kind === 'stage' ? 'VK_PIPELINE_STAGE_2_NONE' : 'VK_ACCESS_2_NONE';
	return mask.map((m) => (kind === 'stage' ? stageName(m as Stage) : accessName(m as Access))).join(' | ');
}
/** Short form for cards: `COPY / TRANSFER_WRITE -> VERTEX_ATTRIBUTE_INPUT / VERTEX_ATTRIBUTE_READ`. */
export function shortText(b: Barrier, image: boolean): string {
	const m = (x: readonly string[]) => (x.length ? x.join('|') : 'NONE');
	const core = `${m(b.srcStage)} / ${m(b.srcAccess)} -> ${m(b.dstStage)} / ${m(b.dstAccess)}`;
	return image ? `${core} · ${b.oldLayout} -> ${b.newLayout}` : core;
}
/** The struct as C code, one line per field (the console's call line). */
export function structText(p: Puzzle, b: Barrier): string {
	const f = [
		`srcStageMask = ${maskText(b.srcStage, 'stage')}`,
		`srcAccessMask = ${maskText(b.srcAccess, 'access')}`,
		`dstStageMask = ${maskText(b.dstStage, 'stage')}`,
		`dstAccessMask = ${maskText(b.dstAccess, 'access')}`,
	];
	if (p.struct === 'VkImageMemoryBarrier2') f.push(`oldLayout = ${layoutName(b.oldLayout)}`, `newLayout = ${layoutName(b.newLayout)}`);
	return `vkCmdPipelineBarrier2: ${p.struct} { ${f.join(', ')} }`;
}

// --- Valid usage ---------------------------------------------------------------------------------------------------

/** Access flags whose stage is missing from the mask: one report per flag (VUID-<struct>-<side>AccessMask-<n>). */
function accessVuids(p: Puzzle, b: Barrier, side: Side): Line[] {
	const stages = side === 'src' ? b.srcStage : b.dstStage;
	const out: Line[] = [];
	for (const a of side === 'src' ? b.srcAccess : b.dstAccess) {
		const s = SUPPORT[a];
		if (!s || s.stages.some((x) => stages.includes(x))) continue;
		const vuid = `VUID-${p.struct}-${side}AccessMask-${s.vu}`;
		if (!ACCESS_VUIDS.has(vuid)) throw new Error(`barriers: ${vuid} is not in ACCESS_VUIDS`);
		out.push({ kind: 'error', text: vuReport(vuid, 'vkCmdPipelineBarrier2'), vuid });
	}
	return out;
}

/**
 * Everything the validation layer refuses when the puzzle's command buffer is recorded and submitted with these
 * barriers: access flags without a stage that performs them, layout rules of transitions, and later commands that
 * find the image in another layout than the one they name. Empty = valid.
 */
export function blockers(p: Puzzle, slots: readonly Barrier[]): Line[] {
	const out: Line[] = [];
	const image = p.struct === 'VkImageMemoryBarrier2';
	let cur: Layout = p.initial ?? 'UNDEFINED';
	let k = 0;
	for (const x of p.seq) {
		if (x.kind === 'cmd') {
			if (x.layoutVuid && x.layout && cur !== x.layout) out.push({ kind: 'error', text: vuReport(x.layoutVuid, x.vuCall ?? x.call), vuid: x.layoutVuid });
			if (x.layout) cur = x.layout;
			continue;
		}
		const b = slots[k++]!;
		out.push(...accessVuids(p, b, 'src'), ...accessVuids(p, b, 'dst'));
		// Equal layouts are ignored (sync2): no transition, no layout rule.
		if (!image || b.oldLayout === b.newLayout) continue;
		const vu = (vuid: string) => out.push({ kind: 'error', text: vuReport(vuid, 'vkCmdPipelineBarrier2'), vuid });
		if (b.oldLayout !== 'UNDEFINED' && b.oldLayout !== cur) vu('VUID-VkImageMemoryBarrier2-oldLayout-01197');
		if (b.newLayout === 'UNDEFINED') vu('VUID-VkImageMemoryBarrier2-newLayout-01198');
		const needs = new Set([LAYOUT_USAGE[b.oldLayout], LAYOUT_USAGE[b.newLayout]].filter((n) => n !== undefined));
		for (const n of needs) if (!n.usage.some((u) => p.usage?.includes(u))) vu(n.vuid);
		cur = b.newLayout;
	}
	return out;
}

// --- Hazards -------------------------------------------------------------------------------------------------------

/** One unprotected pair: the hazard, the two parties (a command id or 'transition'), the access and what misses it. */
export interface Found {
	slot: number;
	hazard: Hazard;
	before: string;
	after: string;
	acc: Acc | null;
	miss: Miss;
}
const TRANSITION = 'transition';

/** Unprotected pairs around every slot, plus the slots whose transition from UNDEFINED drops needed contents. */
export function hazards(p: Puzzle, slots: readonly Barrier[]): { found: Found[]; discard: number[] } {
	const found: Found[] = [];
	const discard: number[] = [];
	const seq = p.seq;
	let k = 0;
	for (const [i, x] of seq.entries()) {
		if (x.kind !== 'slot') continue;
		const slot = k++;
		const b = slots[slot]!;
		const prev = seq[i - 1];
		const next = seq[i + 1];
		const before = prev?.kind === 'cmd' ? prev : null;
		const after = next?.kind === 'cmd' ? next : null;
		const x1 = execScope(b.srcStage, 'src');
		const x2 = execScope(b.dstStage, 'dst');
		const s1 = named(b.srcStage);
		const s2 = named(b.dstStage);
		const a1 = accessScope(b.srcAccess);
		const a2 = accessScope(b.dstAccess);
		/** First reason the earlier access `e` is not covered; `asWrite` = it needs the memory half too. */
		const first = (e: Acc, asWrite: boolean): Miss | null => (!x1.has(e.stage) ? 'exec1' : asWrite && !(s1.has(e.stage) && a1.has(e.access)) ? 'acc1' : null);
		const second = (l: Acc, mem: boolean): Miss | null => (!x2.has(l.stage) ? 'exec2' : mem && !(s2.has(l.stage) && a2.has(l.access)) ? 'acc2' : null);
		const transition = p.struct === 'VkImageMemoryBarrier2' && b.oldLayout !== b.newLayout;
		if (transition && b.oldLayout === 'UNDEFINED' && x.keep) discard.push(slot);
		if (transition) {
			// The transition is a write between the scopes: earlier accesses before it, later accesses after its write.
			for (const e of before?.acc ?? []) {
				const m = first(e, e.write);
				if (m) found.push({ slot, hazard: e.write ? 'WRITE_AFTER_WRITE' : 'WRITE_AFTER_READ', before: before!.id, after: TRANSITION, acc: e, miss: m });
			}
			for (const l of after?.acc ?? []) {
				const m = second(l, true);
				if (m) found.push({ slot, hazard: l.write ? 'WRITE_AFTER_WRITE' : 'READ_AFTER_WRITE', before: TRANSITION, after: after!.id, acc: l, miss: m });
			}
			continue;
		}
		for (const e of before?.acc ?? []) {
			for (const l of after?.acc ?? []) {
				if (!e.write && !l.write) continue;
				const m = first(e, e.write) ?? second(l, e.write);
				if (!m) continue;
				const hazard: Hazard = !e.write ? 'WRITE_AFTER_READ' : l.write ? 'WRITE_AFTER_WRITE' : 'READ_AFTER_WRITE';
				found.push({ slot, hazard, before: before!.id, after: after!.id, acc: m === 'exec1' || m === 'acc1' ? e : l, miss: m });
			}
		}
	}
	return { found, discard };
}

/** Correct = valid and no hazard, no dropped contents. */
export function correct(p: Puzzle, slots: readonly Barrier[]): boolean {
	if (blockers(p, slots).length) return false;
	const h = hazards(p, slots);
	return h.found.length === 0 && h.discard.length === 0;
}

export type MaskKey = 'srcStage' | 'srcAccess' | 'dstStage' | 'dstAccess';
export const MASKS: readonly MaskKey[] = ['srcStage', 'srcAccess', 'dstStage', 'dstAccess'];
/** A flag of a correct barrier that is not needed. */
export interface Extra {
	slot: number;
	mask: MaskKey;
	flag: string;
	/** 'drop' = the barrier stays correct without it; 'part' = a meta flag of which a smaller part would do. */
	why: 'drop' | 'part' | 'narrow' | 'all';
	/** 'narrow': the stage that would do instead. */
	alt?: string;
}

const withMask = (slots: readonly Barrier[], k: number, mask: MaskKey, v: readonly string[]) => slots.map((b, i) => (i === k ? { ...b, [mask]: v } : b));

/**
 * Flags a correct barrier does not need. Greedy: each flag is dropped in turn and stays dropped when the barrier stays
 * correct (so of two flags that cover each other only one is reported). A meta flag that must stay is split into the
 * parts that are valid in its mask (MEMORY_READ on FRAGMENT_SHADER = UNIFORM | SAMPLED | STORAGE reads: "all READ
 * access flags that are valid where it is used") and each part dropped in turn (still correct = a smaller flag would
 * do). A stage that must stay is compared with the stages whose execution scope is a strict part of its own (the
 * narrowest that keeps the barrier correct is named). TOP_OF_PIPE in the destination and BOTTOM_OF_PIPE in the source
 * mean ALL_COMMANDS (with no access) and are always reported.
 */
export function extras(p: Puzzle, slots: readonly Barrier[]): Extra[] {
	const out: Extra[] = [];
	let cur = slots;
	for (const k of slots.keys()) {
		for (const mask of MASKS) {
			for (const f of slots[k]![mask] as readonly string[]) {
				const flags = cur[k]![mask] as readonly string[];
				const rest = flags.filter((x) => x !== f);
				if ((mask === 'dstStage' && f === 'TOP_OF_PIPE') || (mask === 'srcStage' && f === 'BOTTOM_OF_PIPE')) {
					out.push({ slot: k, mask, flag: f, why: 'all' });
					continue;
				}
				const without = withMask(cur, k, mask, rest);
				if (correct(p, without)) {
					out.push({ slot: k, mask, flag: f, why: 'drop' });
					cur = without;
					continue;
				}
				const stageMask = cur[k]![mask === 'srcAccess' ? 'srcStage' : 'dstStage'];
				const parts = (mask.endsWith('Stage') ? META_STAGE[f as Stage] : META_ACCESS[f as Access])?.filter((a) => mask.endsWith('Stage') || !SUPPORT[a as Access] || SUPPORT[a as Access]!.stages.some((s) => stageMask.includes(s)));
				if (parts && parts.length > 1 && parts.some((x) => correct(p, withMask(cur, k, mask, [...rest, ...parts.filter((y) => y !== x)])))) {
					out.push({ slot: k, mask, flag: f, why: 'part' });
					continue;
				}
				// A stage whose execution scope is a strict part of this one's (e.g. VERTEX_ATTRIBUTE_INPUT instead of
				// VERTEX_SHADER as a source, which pulls in every logically earlier stage).
				if (mask === 'srcStage' || mask === 'dstStage') {
					const side: Side = mask === 'srcStage' ? 'src' : 'dst';
					const own = execScope([f as Stage], side);
					const alts = ATOMIC.filter((g) => {
						const sub = execScope([g], side);
						return g !== f && sub.size < own.size && [...sub].every((x) => own.has(x)) && correct(p, withMask(cur, k, mask, [...rest, g]));
					}).sort((x, y) => execScope([x], side).size - execScope([y], side).size);
					if (alts[0]) out.push({ slot: k, mask, flag: f, why: 'narrow', alt: alts[0] });
				}
			}
		}
	}
	return out;
}

// --- Transitions of the trainer ------------------------------------------------------------------------------------

const setPuzzle = (s: BarrierState, id: string, v: Partial<PuzzleState>): BarrierState => ({ ...s, puzzles: { ...s.puzzles, [id]: { ...s.puzzles[id]!, ...v } } });

/** Offered flags of a puzzle in a state ("show all" = every flag). */
export const offeredStages = (s: BarrierState, p: Puzzle): readonly Stage[] => (s.all ? STAGES : p.stages);
export const offeredAccesses = (s: BarrierState, p: Puzzle): readonly Access[] => (s.all ? ACCESSES : p.accesses);

/**
 * Sets one mask of a slot's barrier from the editor: the flags shown are replaced by `flags`, flags the editor does
 * not show right now (set while "show all" was on) are kept. Never refused: editing the struct is not a Vulkan call;
 * the rules run at the submit.
 */
export function setMask(s: BarrierState, id: string, slot: number, mask: MaskKey, flags: readonly string[]): Step {
	const p = puzzleOf(id);
	const ps = s.puzzles[id]!;
	const b = ps.slots[slot]!;
	const stage = mask.endsWith('Stage');
	const shown: readonly string[] = stage ? offeredStages(s, p) : offeredAccesses(s, p);
	const keep = (b[mask] as readonly string[]).filter((x) => !shown.includes(x));
	// Canonical order (the list order) so equal masks compare equal and print the same.
	const all: readonly string[] = stage ? STAGES : ACCESSES;
	const v = all.filter((x) => keep.includes(x) || flags.includes(x));
	const same = v.length === (b[mask] as readonly string[]).length && v.every((x, i) => x === (b[mask] as readonly string[])[i]);
	const state = same ? s : setPuzzle(s, id, { slots: ps.slots.map((x, i) => (i === slot ? { ...b, [mask]: v } : x)), dirty: true });
	return { state, log: [{ kind: 'info', text: `${mask}Mask = ${maskText(v, stage ? 'stage' : 'access')}` }] };
}

/** Sets the two layouts of a slot's image barrier. */
export function setLayouts(s: BarrierState, id: string, slot: number, oldLayout: Layout, newLayout: Layout): Step {
	const ps = s.puzzles[id]!;
	const b = ps.slots[slot]!;
	const same = b.oldLayout === oldLayout && b.newLayout === newLayout;
	const state = same ? s : setPuzzle(s, id, { slots: ps.slots.map((x, i) => (i === slot ? { ...x, oldLayout, newLayout } : x)), dirty: true });
	return { state, log: [{ kind: 'info', text: `oldLayout = ${layoutName(oldLayout)}; newLayout = ${layoutName(newLayout)}` }, msg('info', oldLayout === newLayout ? 'noTransition' : 'transition', { slot: slot + 1, slots: ps.slots.length })] };
}

/** Why a submit is refused (the valid-usage reports). */
export const submitBlockers = (s: BarrierState, id: string): Line[] => blockers(puzzleOf(id), s.puzzles[id]!.slots);

const partyOf = (p: Puzzle, id: string): string => (id === TRANSITION ? TRANSITION : (p.seq.find((x): x is Cmd => x.kind === 'cmd' && x.id === id)?.call ?? id));

/**
 * Records the command buffer with the barriers and submits it under synchronization validation (call only after
 * `submitBlockers` is empty). Hazards and dropped contents are logged as errors and leave the puzzle open; a correct
 * set of barriers is graded tight or wide (with the flags it does not need).
 */
export function submit(s: BarrierState, id: string): Step {
	const p = puzzleOf(id);
	const ps = s.puzzles[id]!;
	const log: Line[] = [...ps.slots.map((b) => ({ kind: 'call' as const, text: structText(p, b) })), { kind: 'call', text: 'vkQueueSubmit2 -> VK_SUCCESS' }];
	const { found, discard } = hazards(p, ps.slots);
	const broken = ps.slots.map((_, k) => found.some((f) => f.slot === k) || discard.includes(k));
	const slots = ps.slots.length;
	for (const f of found) {
		const vars = { hazard: f.hazard, before: partyOf(p, f.before), after: partyOf(p, f.after), stage: f.acc ? stageName(f.acc.stage) : '', access: f.acc ? accessName(f.acc.access) : '', slot: f.slot + 1, slots };
		log.push({ ...msg('error', `hazard_${f.miss}`, vars), sync: syncId(f.hazard) });
		const ev = p.evidence?.find((e) => e.slot === f.slot && e.hazard === f.hazard && e.before === f.before && e.after === f.after);
		if (ev?.text) log.push(msg('info', 'evidence', { step: ev.step, slot: f.slot + 1, slots }), { kind: 'error', text: ev.text, sync: syncId(f.hazard) });
		else if (ev) log.push(msg('info', 'silent', { step: ev.step, slot: f.slot + 1, slots }));
	}
	for (const k of discard) log.push(msg('warn', 'discard', { slot: k + 1, slots }));
	let grade: Grade = 'hazard';
	if (!found.length && !discard.length) {
		const ex = extras(p, ps.slots);
		grade = ex.length ? 'wide' : 'tight';
		if (!ex.length) log.push(msg('info', 'tight', { slots }));
		else {
			log.push(msg('warn', 'wide', { n: ex.length, slots }));
			for (const e of ex) log.push(msg('info', `extra_${e.why}`, { mask: `${e.mask}Mask`, flag: e.flag, alt: e.alt ?? '', slot: e.slot + 1, slots }));
		}
	}
	return { state: setPuzzle(s, id, { grade, broken, dirty: false }), log };
}

/** Puzzles solved (correct, submitted, not edited since) and tight ones. */
export function tally(s: BarrierState): { solved: number; tight: number; total: number } {
	const ps = PUZZLES.map((p) => s.puzzles[p.id]!);
	const ok = (x: PuzzleState) => !x.dirty && (x.grade === 'tight' || x.grade === 'wide');
	return { solved: ps.filter(ok).length, tight: ps.filter((x) => ok(x) && x.grade === 'tight').length, total: PUZZLES.length };
}
