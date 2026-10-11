/**
 * Model of the queue-sync lab: a scenario of host operations (vkQueueSubmit2 calls with one batch each, and
 * vkWaitForFences + vkResetFences pairs) on the graphics, async compute and transfer queues of the course GPU, a
 * tick-based scheduler that draws ONE schedule the spec allows, and a checker that finds data races and valid-usage
 * violations from the synchronization rules themselves, not from the drawn schedule.
 *
 * Rules (Vulkan 1.4 spec, "Synchronization and Cache Control" and vkQueueSubmit2):
 * - Submission order on one queue = order of the vkQueueSubmit2 calls on the host. It defines no dependency by itself:
 *   later batches may overlap earlier ones.
 * - A semaphore signal's first scope: every command of its batch and all commands earlier in submission order on
 *   that queue, limited by its stageMask. The lab always signals with ALL_COMMANDS, so a signal fires when its batch
 *   and every earlier batch of the queue have finished.
 * - A semaphore wait's second scope: every command of its batch and all commands later in submission order on that
 *   queue, limited by its stageMask and the logically later stages ("its second synchronization scope only includes
 *   execution of the pipeline stages specified in that mask and any logically later stages").
 * - Binary semaphores pair 1:1 (a wait unsignals); the signal must already be submitted when the wait is
 *   (VUID-vkQueueSubmit2-semaphore-03873). A timeline wait for v is satisfied by any signal of a value >= v, and may be
 *   submitted before that signal.
 * - A fence signal's first scope: the batches of its vkQueueSubmit2 call and all commands earlier in submission order.
 * - Host write ordering: a submission happens after everything the host did before the call.
 * Valid-usage ids per issue kind are in ISSUE_VUID; their texts come from `vuid()` (src/lib/vuids.ts).
 * Pure module: no DOM, no React, tested with `bun test`.
 */
import gpu from '../../data/gpu.json';

// ---------------------------------------------------------------------------------------------------------------
// Queues and pipeline stages

export const QUEUE_IDS = ['graphics', 'compute', 'transfer'] as const;
export type QueueId = (typeof QUEUE_IDS)[number];

/** Capability a queue role needs, and the flags it must not have to count as dedicated (async compute, copy-only). */
const ROLE: Record<QueueId, { need: string; without: string[] }> = {
	graphics: { need: 'VK_QUEUE_GRAPHICS_BIT', without: [] },
	compute: { need: 'VK_QUEUE_COMPUTE_BIT', without: ['VK_QUEUE_GRAPHICS_BIT'] },
	transfer: { need: 'VK_QUEUE_TRANSFER_BIT', without: ['VK_QUEUE_GRAPHICS_BIT', 'VK_QUEUE_COMPUTE_BIT', 'VK_QUEUE_VIDEO_DECODE_BIT_KHR', 'VK_QUEUE_VIDEO_ENCODE_BIT_KHR', 'VK_QUEUE_OPTICAL_FLOW_BIT_NV'] },
};

export interface QueueInfo {
	id: QueueId;
	/** Queue family index on the course GPU (gpu.json). */
	family: number;
	/** Capability flags without `VK_QUEUE_` / `_BIT` (graphics and compute families imply TRANSFER). */
	caps: string[];
}

const short = (f: string) => f.replace(/^VK_QUEUE_/, '').replace(/_BIT(_\w+)?$/, '');

/** The three queues of the lab, each from the first family of the course GPU that fits its role. */
export const QUEUES: readonly QueueInfo[] = QUEUE_IDS.map((id) => {
	const r = ROLE[id];
	const fam = gpu.queueFamilies.find((f) => f.queueFlags.includes(r.need) && !r.without.some((w) => f.queueFlags.includes(w)));
	if (!fam) throw new Error(`gpu.json: no queue family for the ${id} queue`);
	const caps = fam.queueFlags.map(short).filter((c) => c !== 'SPARSE_BINDING');
	// The spec makes TRANSFER implicit on graphics and compute families.
	if (!caps.includes('TRANSFER')) caps.push('TRANSFER');
	return { id, family: fam.index, caps };
});
export const queue = (id: QueueId): QueueInfo => QUEUES.find((q) => q.id === id)!;

/** Pipeline stages the lab's work executes (VK_PIPELINE_STAGE_2_*_BIT without prefix/suffix). */
export const STAGES = ['COPY', 'COMPUTE_SHADER', 'VERTEX_SHADER', 'FRAGMENT_SHADER', 'COLOR_ATTACHMENT_OUTPUT'] as const;
export type Stage = (typeof STAGES)[number];
/** Stage masks a wait can use: one stage or ALL_COMMANDS. */
export const MASKS = [...STAGES, 'ALL_COMMANDS'] as const;
export type Mask = (typeof MASKS)[number];

/** Full Vulkan name of a stage mask. */
export const stageName = (m: Mask) => `VK_PIPELINE_STAGE_2_${m}_BIT`;

/** Spec table "Supported pipeline stage flags": the queue capabilities of which ONE is required (none = any queue). */
const STAGE_NEEDS: Record<Mask, string[]> = {
	COPY: ['GRAPHICS', 'COMPUTE', 'TRANSFER'],
	COMPUTE_SHADER: ['COMPUTE'],
	VERTEX_SHADER: ['GRAPHICS'],
	FRAGMENT_SHADER: ['GRAPHICS'],
	COLOR_ATTACHMENT_OUTPUT: ['GRAPHICS'],
	ALL_COMMANDS: [],
};
export const stageSupported = (m: Mask, q: QueueId) => STAGE_NEEDS[m].length === 0 || STAGE_NEEDS[m].some((c) => queue(q).caps.includes(c));

/**
 * Stages in a mask's second scope: the stage itself and the logically later stages of the same pipeline (graphics
 * primitive pipeline: VERTEX_SHADER < FRAGMENT_SHADER < COLOR_ATTACHMENT_OUTPUT; copy and compute stand alone).
 */
export function expand(m: Mask): Stage[] {
	if (m === 'ALL_COMMANDS') return [...STAGES];
	const gfx: Stage[] = ['VERTEX_SHADER', 'FRAGMENT_SHADER', 'COLOR_ATTACHMENT_OUTPUT'];
	const i = gfx.indexOf(m);
	return i < 0 ? [m] : gfx.slice(i);
}

/** Kinds of work a batch records: one copy, one dispatch, or one draw, and the queue capability its command needs. */
export const WORKS = ['copy', 'dispatch', 'draw'] as const;
export type Work = (typeof WORKS)[number];
const WORK_NEEDS: Record<Work, string[]> = { copy: ['GRAPHICS', 'COMPUTE', 'TRANSFER'], dispatch: ['COMPUTE'], draw: ['GRAPHICS'] };
export const workSupported = (w: Work, q: QueueId) => WORK_NEEDS[w].some((c) => queue(q).caps.includes(c));
export const queuesFor = (w: Work): QueueId[] => QUEUE_IDS.filter((q) => workSupported(w, q));

/** Shortest duration of a work kind: a draw shows its three stages one tick at least each. */
export const MIN_DUR: Record<Work, number> = { copy: 1, dispatch: 1, draw: 3 };
export const MAX_DUR = 12;

/**
 * Phases of a batch in time. A draw's vertex and color-output stages take one tick each and fragment shading the rest.
 * Durations are illustrative units, not measurements.
 */
export function phases(w: Work, dur: number): { stage: Stage; len: number }[] {
	if (w === 'copy') return [{ stage: 'COPY', len: dur }];
	if (w === 'dispatch') return [{ stage: 'COMPUTE_SHADER', len: dur }];
	return [
		{ stage: 'VERTEX_SHADER', len: 1 },
		{ stage: 'FRAGMENT_SHADER', len: dur - 2 },
		{ stage: 'COLOR_ATTACHMENT_OUTPUT', len: 1 },
	];
}

// ---------------------------------------------------------------------------------------------------------------
// Scenario

export type SemType = 'binary' | 'timeline';
export interface Sem {
	id: string;
	type: SemType;
}
export interface Fence {
	id: string;
	/** Created with VK_FENCE_CREATE_SIGNALED_BIT (the usual choice for frames in flight: the first wait returns). */
	signaled: boolean;
}
export interface Wait {
	sem: string;
	/** Timeline wait value (ignored for binary). */
	value: number;
	/** VkSemaphoreSubmitInfo::stageMask of the wait. */
	stage: Mask;
}
export interface Signal {
	sem: string;
	/** Timeline signal value (ignored for binary). Signals use stageMask ALL_COMMANDS. */
	value: number;
}
export interface Access {
	res: string;
	write: boolean;
	stage: Stage;
}

/** Names of the work items; their texts live in the lab's text table (`w_<name>`). */
export type WorkName = 'upload' | 'render' | 'simulate' | 'gbuffer' | 'particles' | 'frame';

/** vkQueueSubmit2 with one VkSubmitInfo2. The host records its command buffer first (vkBeginCommandBuffer). */
export interface SubmitOp {
	kind: 'submit';
	name: WorkName;
	/** Number shown in the name (frame 2, simulation step 1). */
	n?: number;
	queue: QueueId;
	work: Work;
	dur: number;
	/** Command buffer recorded and submitted. */
	cb: string;
	waits: Wait[];
	signals: Signal[];
	fence?: string;
	/** Resources the batch reads or writes, with the stage that does it. */
	uses: Access[];
	/** Resources the host writes while recording (a mapped uniform buffer). */
	hostWrites: string[];
}
/** vkWaitForFences(UINT64_MAX) followed by vkResetFences on one fence. */
export interface FenceWaitOp {
	kind: 'wait';
	fence: string;
}
export type HostOp = SubmitOp | FenceWaitOp;

export interface Scenario {
	sems: Sem[];
	fences: Fence[];
	/** Host operations in program order. */
	ops: HostOp[];
}

/** Host time to record a command buffer before submitting it, illustrative units. */
export const RECORD = 1;

const submit = (s: Omit<SubmitOp, 'kind' | 'waits' | 'signals' | 'uses' | 'hostWrites'> & Partial<SubmitOp>): SubmitOp => ({
	kind: 'submit',
	waits: [],
	signals: [],
	uses: [],
	hostWrites: [],
	...s,
});

export const PRESET_IDS = ['upload', 'race', 'async', 'frames'] as const;
export type PresetId = (typeof PRESET_IDS)[number];

const upload = (withSem: boolean): Scenario => ({
	sems: [{ id: 'uploaded', type: 'binary' }],
	fences: [],
	ops: [
		submit({ name: 'upload', queue: 'transfer', work: 'copy', dur: 4, cb: 'cbUpload', signals: withSem ? [{ sem: 'uploaded', value: 1 }] : [], uses: [{ res: 'texture', write: true, stage: 'COPY' }] }),
		submit({
			name: 'render',
			queue: 'graphics',
			work: 'draw',
			dur: 5,
			cb: 'cbRender',
			waits: withSem ? [{ sem: 'uploaded', value: 1, stage: 'FRAGMENT_SHADER' }] : [],
			uses: [{ res: 'texture', write: false, stage: 'FRAGMENT_SHADER' }],
		}),
	],
});

const frame = (n: number): SubmitOp =>
	submit({
		name: 'frame',
		n,
		queue: 'graphics',
		work: 'draw',
		dur: 6,
		cb: `cb${n % 2}`,
		fence: `F${n % 2}`,
		hostWrites: [`uniforms${n % 2}`],
		uses: [{ res: `uniforms${n % 2}`, write: false, stage: 'VERTEX_SHADER' }],
	});

/** The starting scenarios: (a) upload then render, (b) the same without the semaphore, (c) async compute, (d) frames in flight. */
export const PRESETS: Readonly<Record<PresetId, Scenario>> = {
	upload: upload(true),
	race: upload(false),
	async: {
		sems: [{ id: 'timeline', type: 'timeline' }],
		fences: [],
		ops: [
			submit({ name: 'simulate', n: 1, queue: 'compute', work: 'dispatch', dur: 5, cb: 'cbSim1', signals: [{ sem: 'timeline', value: 1 }], uses: [{ res: 'particles', write: true, stage: 'COMPUTE_SHADER' }] }),
			submit({ name: 'gbuffer', queue: 'graphics', work: 'draw', dur: 6, cb: 'cbGbuffer', uses: [{ res: 'gbuffer', write: true, stage: 'COLOR_ATTACHMENT_OUTPUT' }] }),
			submit({
				name: 'particles',
				queue: 'graphics',
				work: 'draw',
				dur: 4,
				cb: 'cbParticles',
				waits: [{ sem: 'timeline', value: 1, stage: 'VERTEX_SHADER' }],
				signals: [{ sem: 'timeline', value: 2 }],
				uses: [{ res: 'particles', write: false, stage: 'VERTEX_SHADER' }],
			}),
			submit({
				name: 'simulate',
				n: 2,
				queue: 'compute',
				work: 'dispatch',
				dur: 5,
				cb: 'cbSim2',
				waits: [{ sem: 'timeline', value: 2, stage: 'COMPUTE_SHADER' }],
				signals: [{ sem: 'timeline', value: 3 }],
				uses: [{ res: 'particles', write: true, stage: 'COMPUTE_SHADER' }],
			}),
		],
	},
	frames: {
		sems: [],
		fences: [
			{ id: 'F0', signaled: true },
			{ id: 'F1', signaled: true },
		],
		ops: [0, 1, 2, 3].flatMap((n): HostOp[] => [{ kind: 'wait', fence: `F${n % 2}` }, frame(n)]),
	},
};

/** Structural problems of a scenario (unknown names, bad durations, work on a queue that cannot run it); empty = valid. */
export function scenarioErrors(sc: Scenario): string[] {
	const errs: string[] = [];
	const sems = new Set(sc.sems.map((s) => s.id));
	const fences = new Set(sc.fences.map((f) => f.id));
	sc.ops.forEach((op, i) => {
		const at = `op ${i}`;
		if (op.kind === 'wait') {
			if (!fences.has(op.fence)) errs.push(`${at}: unknown fence ${op.fence}`);
			return;
		}
		if (!workSupported(op.work, op.queue)) errs.push(`${at}: ${op.work} cannot run on the ${op.queue} queue`);
		if (!Number.isInteger(op.dur) || op.dur < MIN_DUR[op.work] || op.dur > MAX_DUR) errs.push(`${at}: duration ${op.dur} out of range`);
		for (const x of [...op.waits, ...op.signals]) if (!sems.has(x.sem)) errs.push(`${at}: unknown semaphore ${x.sem}`);
		for (const x of [...op.waits, ...op.signals]) if (!Number.isInteger(x.value) || x.value < 1) errs.push(`${at}: semaphore value ${x.value} must be a positive integer`);
		if (op.fence && !fences.has(op.fence)) errs.push(`${at}: unknown fence ${op.fence}`);
		const stages = phases(op.work, op.dur).map((p) => p.stage);
		for (const u of op.uses) if (!stages.includes(u.stage)) errs.push(`${at}: ${u.res} is used at ${u.stage}, which a ${op.work} does not execute`);
	});
	return errs;
}

// ---------------------------------------------------------------------------------------------------------------
// Issues

export type IssueKind =
	| 'race'
	| 'binaryNoSignal'
	| 'binarySignaled'
	| 'timelineNotIncreasing'
	| 'timelineSameBatch'
	| 'stageUnsupported'
	| 'cbPending'
	| 'fenceSignaled'
	| 'fencePending'
	| 'fenceNeverSubmitted'
	| 'cpuBlocked'
	| 'neverRuns';

/** The valid-usage rule each issue kind breaks; races and deadlocks break none (every call is valid on its own). */
export const ISSUE_VUID: Readonly<Partial<Record<IssueKind, string>>> = {
	binaryNoSignal: 'VUID-vkQueueSubmit2-semaphore-03873',
	binarySignaled: 'VUID-vkQueueSubmit2-semaphore-03868',
	timelineNotIncreasing: 'VUID-VkSubmitInfo2-semaphore-03882',
	timelineSameBatch: 'VUID-VkSubmitInfo2-semaphore-03881',
	stageUnsupported: 'VUID-vkQueueSubmit2-stageMask-03870',
	cbPending: 'VUID-vkBeginCommandBuffer-commandBuffer-00049',
	fenceSignaled: 'VUID-vkQueueSubmit2-fence-04894',
	fencePending: 'VUID-vkQueueSubmit2-fence-04895',
};

export interface Issue {
	kind: IssueKind;
	/** Host op the issue is reported at. */
	op: number;
	/** Other ops involved (the other side of a race, the earlier use of a command buffer). */
	others: number[];
	/** Values for the issue's text: names, stages, times. */
	vars: Record<string, string | number>;
	/** Valid-usage id (its text: `vuid()` from src/lib/vuids.ts); races and deadlocks have none. */
	vuid?: string;
}

function issue(kind: IssueKind, op: number, vars: Record<string, string | number> = {}, others: number[] = []): Issue {
	const vuid = ISSUE_VUID[kind];
	return vuid ? { kind, op, others, vars, vuid } : { kind, op, others, vars };
}

// ---------------------------------------------------------------------------------------------------------------
// Binary semaphore pairing (host order)

/** `op:index` key of a wait or signal of a submit. */
const key = (op: number, k: number) => `${op}:${k}`;

/**
 * Pairs every binary wait with a signal of the same semaphore submitted before it and not yet taken by another wait
 * (the oldest one, as they execute in that order). null = no such signal: VUID-vkQueueSubmit2-semaphore-03873.
 */
export function pairBinary(sc: Scenario): Map<string, string | null> {
	const type = new Map(sc.sems.map((s) => [s.id, s.type]));
	const open = new Map<string, string[]>();
	const pairs = new Map<string, string | null>();
	sc.ops.forEach((op, i) => {
		if (op.kind !== 'submit') return;
		// Waits of a batch are paired before its own signals: a batch cannot wait for itself.
		op.waits.forEach((w, k) => {
			if (type.get(w.sem) !== 'binary') return;
			pairs.set(key(i, k), open.get(w.sem)?.shift() ?? null);
		});
		op.signals.forEach((s, k) => {
			if (type.get(s.sem) !== 'binary') return;
			open.set(s.sem, [...(open.get(s.sem) ?? []), key(i, k)]);
		});
	});
	return pairs;
}

// ---------------------------------------------------------------------------------------------------------------
// Happens-before graph (the rules, independent of any schedule)

/**
 * Happens-before among the operations of a scenario. Nodes: stage executions of each batch `A:op:stage`, semaphore
 * signals `S:op:k` and waits `W:op:k`, fence signals `F:op`, and host operations `H:i`. An edge a -> b means "a
 * happens-before b" by one rule of the spec; reachability gives execution dependency chains (a chain through a batch
 * needs an actual stage execution in both scopes, which the A nodes enforce).
 */
export class Order {
	private readonly next = new Map<string, Set<string>>();
	private readonly memo = new Map<string, Set<string>>();

	constructor(readonly sc: Scenario) {
		const ops = sc.ops;
		const type = new Map(sc.sems.map((s) => [s.id, s.type]));
		const pairs = pairBinary(sc);
		const stagesOf = (i: number) => {
			const op = ops[i] as SubmitOp;
			return phases(op.work, op.dur).map((p) => p.stage);
		};
		const submits = ops.flatMap((op, i) => (op.kind === 'submit' ? [i] : []));
		const sameQueue = (a: number, b: number) => (ops[a] as SubmitOp).queue === (ops[b] as SubmitOp).queue;

		ops.forEach((op, i) => {
			if (i + 1 < ops.length) this.edge(`H:${i}`, `H:${i + 1}`);
			if (op.kind !== 'submit') return;
			// Host write ordering: everything the host did before the call happens-before the batch.
			for (const s of stagesOf(i)) this.edge(`H:${i}`, `A:${i}:${s}`);
			// Signals (stageMask ALL_COMMANDS) and the fence cover this batch and every earlier batch of the queue.
			for (const j of submits.filter((j) => j <= i && sameQueue(i, j))) {
				for (const s of stagesOf(j)) {
					op.signals.forEach((_, k) => this.edge(`A:${j}:${s}`, `S:${i}:${k}`));
					if (op.fence) this.edge(`A:${j}:${s}`, `F:${i}`);
				}
			}
			// Waits cover this batch and every later batch of the queue, at the mask's stages and logically later ones.
			op.waits.forEach((w, k) => {
				const scope = expand(w.stage);
				for (const j of submits.filter((j) => j >= i && sameQueue(i, j))) for (const s of stagesOf(j)) if (scope.includes(s)) this.edge(`W:${i}:${k}`, `A:${j}:${s}`);
				if (type.get(w.sem) === 'binary') {
					const p = pairs.get(key(i, k));
					if (p) this.edge(`S:${p}`, `W:${i}:${k}`);
					return;
				}
				// Timeline: the wait is satisfied by the first value >= w.value to arrive. Signal values increase on
				// execution (VUID-VkSubmitInfo2-semaphore-03882), so every signal up to the smallest value >= w.value
				// has happened by then.
				const vals = ops.flatMap((o, j) => (o.kind === 'submit' ? o.signals.flatMap((s, m) => (s.sem === w.sem ? [{ j, m, v: s.value }] : [])) : []));
				const reached = vals.filter((x) => x.v >= w.value).map((x) => x.v);
				if (!reached.length) return;
				const first = Math.min(...reached);
				for (const x of vals) if (x.v <= first) this.edge(`S:${x.j}:${x.m}`, `W:${i}:${k}`);
			});
		});
		// Fences: the host wait happens after the latest submission that used the fence since its last reset.
		const lastUse = new Map<string, number>();
		ops.forEach((op, i) => {
			if (op.kind === 'submit' && op.fence) lastUse.set(op.fence, i);
			if (op.kind === 'wait') {
				const j = lastUse.get(op.fence);
				if (j !== undefined) this.edge(`F:${j}`, `H:${i}`);
				lastUse.delete(op.fence);
			}
		});
	}

	private edge(a: string, b: string) {
		let s = this.next.get(a);
		if (!s) this.next.set(a, (s = new Set()));
		s.add(b);
	}

	/** True when a happens-before b (a chain of rules leads from a to b). */
	before(a: string, b: string): boolean {
		let seen = this.memo.get(a);
		if (!seen) {
			seen = new Set();
			const stack = [a];
			while (stack.length) for (const n of this.next.get(stack.pop()!) ?? []) if (!seen.has(n)) (seen.add(n), stack.push(n));
			this.memo.set(a, seen);
		}
		return seen.has(b);
	}
}

/** One access in the happens-before graph: a batch's stage or a host operation. */
interface Touch {
	op: number;
	node: string;
	res: string;
	write: boolean;
	stage: Stage | 'HOST';
}

function touches(sc: Scenario): Touch[] {
	return sc.ops.flatMap((op, i): Touch[] =>
		op.kind !== 'submit'
			? []
			: [
					...op.hostWrites.map((res) => ({ op: i, node: `H:${i}`, res, write: true, stage: 'HOST' as const })),
					...op.uses.map((u) => ({ op: i, node: `A:${i}:${u.stage}`, res: u.res, write: u.write, stage: u.stage })),
				],
	);
}

/** Issues that follow from the rules alone: races, binary pairing, stage support, timeline values in one batch, command buffer reuse. */
export function staticIssues(sc: Scenario): Issue[] {
	const out: Issue[] = [];
	const ord = new Order(sc);
	const type = new Map(sc.sems.map((s) => [s.id, s.type]));
	const pairs = pairBinary(sc);

	sc.ops.forEach((op, i) => {
		if (op.kind !== 'submit') return;
		op.waits.forEach((w, k) => {
			if (type.get(w.sem) === 'binary' && pairs.get(key(i, k)) === null) out.push(issue('binaryNoSignal', i, { sem: w.sem }));
			if (!stageSupported(w.stage, op.queue)) out.push(issue('stageUnsupported', i, { sem: w.sem, stage: stageName(w.stage), queue: op.queue, family: queue(op.queue).family }));
			if (type.get(w.sem) === 'timeline') {
				const s = op.signals.find((s) => s.sem === w.sem && s.value <= w.value);
				if (s) out.push(issue('timelineSameBatch', i, { sem: w.sem, wait: w.value, signal: s.value }));
			}
		});
		// Recording into a command buffer is safe only after every earlier submission of it is known to be complete.
		sc.ops.forEach((prev, j) => {
			if (j >= i || prev.kind !== 'submit' || prev.cb !== op.cb) return;
			const done = phases(prev.work, prev.dur).every((p) => ord.before(`A:${j}:${p.stage}`, `H:${i}`));
			if (!done) out.push(issue('cbPending', i, { cb: op.cb }, [j]));
		});
	});

	// Data races: two accesses to one resource, at least one a write, with no happens-before either way. Accesses
	// inside one batch are left to the barriers in its command buffer, which the lab does not model.
	const ts = touches(sc);
	ts.forEach((a, x) => {
		for (const b of ts.slice(x + 1)) {
			if (a.res !== b.res || (!a.write && !b.write) || a.op === b.op) continue;
			if (ord.before(a.node, b.node) || ord.before(b.node, a.node)) continue;
			out.push(issue('race', b.op, { res: a.res, a: a.write ? 'write' : 'read', b: b.write ? 'write' : 'read', aStage: a.stage, bStage: b.stage }, [a.op]));
		}
	});
	return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Scheduler

/** Lanes of the timeline: the host CPU and the three queues. */
export type Lane = 'cpu' | QueueId;

/** A stretch of one lane: a stage running, a batch stalled on a wait (started or not yet), the CPU recording or waiting. */
export interface Seg {
	lane: Lane;
	op: number;
	from: number;
	to: number;
	what: Stage | 'pending' | 'stall' | 'record' | 'wait';
}

export type SimEvent =
	| { t: number; kind: 'submit' | 'start' | 'done'; op: number }
	| { t: number; kind: 'signal' | 'waitDone'; op: number; sem: string; value?: number }
	| { t: number; kind: 'blocked'; op: number; sem: string; stage: Stage }
	| { t: number; kind: 'fenceSignal'; op: number; fence: string }
	| { t: number; kind: 'cpuWait' | 'cpuWaitDone'; op: number; fence: string };

/** Signal -> wait (semaphore) or fence signal -> CPU wait, drawn as arrows. */
export interface Link {
	kind: 'sem' | 'fence';
	name: string;
	from: { op: number; t: number };
	/** Host op that waits (a submit for a semaphore, a fence wait for a fence). */
	to: { op: number; t: number };
}

export interface BatchRun {
	submitted: number | null;
	started: number | null;
	done: number | null;
}

export interface Run {
	/** Last time step; frames go from 0 to end. */
	end: number;
	segs: Seg[];
	events: SimEvent[];
	links: Link[];
	/** Per host op (null for fence waits). */
	batches: (BatchRun | null)[];
	/** Issues found while running: double signals, non-increasing timeline values, fence reuse, deadlocks. */
	issues: Issue[];
}

/** Longest run the scheduler draws before it stops (only reached by scenarios that never finish). */
export const MAX_T = 120;

/**
 * Runs a scenario tick by tick and returns one schedule the spec allows: the host performs its operations in order
 * (recording takes RECORD ticks, a fence wait blocks); a batch runs as soon as it is submitted, the batches before it on
 * its queue have started, and no unsatisfied wait covers its current stage. Batches of a queue may overlap: nothing
 * but synchronization orders them.
 */
export function simulate(sc: Scenario): Run {
	const ops = sc.ops;
	const type = new Map(sc.sems.map((s) => [s.id, s.type]));
	const pairs = pairBinary(sc);
	const segs: Seg[] = [];
	const events: SimEvent[] = [];
	const links: Link[] = [];
	const issues: Issue[] = [];

	interface B {
		run: BatchRun;
		phases: { stage: Stage; len: number }[];
		phase: number;
		left: number;
		waitDone: (number | null)[];
		signalDone: (number | null)[];
		fenceDone: number | null;
		blockedOn: Set<string>;
	}
	const bs: (B | null)[] = ops.map((op) => {
		if (op.kind !== 'submit') return null;
		const ph = phases(op.work, op.dur);
		return {
			run: { submitted: null, started: null, done: null },
			phases: ph,
			phase: 0,
			left: ph[0]!.len,
			waitDone: op.waits.map(() => null),
			signalDone: op.signals.map(() => null),
			fenceDone: null,
			blockedOn: new Set(),
		};
	});
	const binSignaled = new Map<string, boolean>();
	const value = new Map<string, number>();
	/** Which signal last set a timeline value: the source of the arrow to the waits it satisfies. */
	const lastSetter = new Map<string, { op: number; t: number }>();
	const fenceOn = new Map(sc.fences.map((f) => [f.id, f.signaled]));
	/** The submission a fence is currently associated with (until a host wait resets it). */
	const fenceOwner = new Map<string, number>();
	const sigTime = new Map<string, number>();

	let cpu = 0;
	let recordFrom: number | null = null;
	let waitFrom: number | null = null;

	const queueOf = (i: number) => (ops[i] as SubmitOp).queue;
	const earlier = (i: number) => ops.flatMap((o, j) => (j < i && o.kind === 'submit' && o.queue === queueOf(i) ? [j] : []));
	const scopeDone = (i: number, t: number) => [...earlier(i), i].every((j) => bs[j]!.run.done !== null && bs[j]!.run.done! <= t);

	/** Instantaneous events at time t: signals, satisfied waits, host progress. Returns true if anything changed. */
	const settle = (t: number): boolean => {
		let changed = false;
		ops.forEach((op, i) => {
			const b = bs[i];
			if (op.kind !== 'submit' || !b || b.run.submitted === null || !scopeDone(i, t)) return;
			op.signals.forEach((s, k) => {
				if (b.signalDone[k] !== null) return;
				b.signalDone[k] = t;
				sigTime.set(key(i, k), t);
				changed = true;
				if (type.get(s.sem) === 'binary') {
					if (binSignaled.get(s.sem)) issues.push(issue('binarySignaled', i, { sem: s.sem, t }));
					binSignaled.set(s.sem, true);
					events.push({ t, kind: 'signal', op: i, sem: s.sem });
				} else {
					const cur = value.get(s.sem) ?? 0;
					if (s.value <= cur) issues.push(issue('timelineNotIncreasing', i, { sem: s.sem, value: s.value, current: cur }));
					else (value.set(s.sem, s.value), lastSetter.set(s.sem, { op: i, t }));
					events.push({ t, kind: 'signal', op: i, sem: s.sem, value: s.value });
				}
			});
			if (op.fence && b.fenceDone === null) {
				b.fenceDone = t;
				changed = true;
				fenceOn.set(op.fence, true);
				events.push({ t, kind: 'fenceSignal', op: i, fence: op.fence });
			}
		});
		ops.forEach((op, i) => {
			const b = bs[i];
			if (op.kind !== 'submit' || !b || b.run.submitted === null) return;
			op.waits.forEach((w, k) => {
				if (b.waitDone[k] !== null) return;
				let from: { op: number; t: number } | undefined;
				if (type.get(w.sem) === 'binary') {
					const p = pairs.get(key(i, k));
					const st = p ? sigTime.get(p) : undefined;
					if (!p || st === undefined || !binSignaled.get(w.sem)) return;
					binSignaled.set(w.sem, false);
					from = { op: Number(p.split(':')[0]), t: st };
				} else {
					if ((value.get(w.sem) ?? 0) < w.value) return;
					from = lastSetter.get(w.sem);
				}
				b.waitDone[k] = t;
				changed = true;
				events.push({ t, kind: 'waitDone', op: i, sem: w.sem, value: type.get(w.sem) === 'timeline' ? w.value : undefined });
				if (from) links.push({ kind: 'sem', name: w.sem, from, to: { op: i, t } });
			});
		});
		// The host: fence waits return when the fence is signaled; a submit happens RECORD ticks after recording began.
		while (cpu < ops.length) {
			const op = ops[cpu]!;
			if (op.kind === 'wait') {
				if (waitFrom === null) {
					waitFrom = t;
					events.push({ t, kind: 'cpuWait', op: cpu, fence: op.fence });
				}
				if (!fenceOn.get(op.fence)) break;
				if (t > waitFrom) segs.push({ lane: 'cpu', op: cpu, from: waitFrom, to: t, what: 'wait' });
				const owner = fenceOwner.get(op.fence);
				const fd = owner !== undefined ? bs[owner]!.fenceDone : null;
				if (owner !== undefined && fd !== null) links.push({ kind: 'fence', name: op.fence, from: { op: owner, t: fd }, to: { op: cpu, t } });
				events.push({ t, kind: 'cpuWaitDone', op: cpu, fence: op.fence });
				fenceOn.set(op.fence, false);
				fenceOwner.delete(op.fence);
				waitFrom = null;
			} else {
				if (recordFrom === null) recordFrom = t;
				if (t - recordFrom < RECORD) break;
				segs.push({ lane: 'cpu', op: cpu, from: recordFrom, to: t, what: 'record' });
				recordFrom = null;
				if (op.fence) {
					const owner = fenceOwner.get(op.fence);
					if (fenceOn.get(op.fence)) issues.push(issue('fenceSignaled', cpu, { fence: op.fence }));
					else if (owner !== undefined) issues.push(issue('fencePending', cpu, { fence: op.fence }, [owner]));
					fenceOn.set(op.fence, false);
					fenceOwner.set(op.fence, cpu);
				}
				bs[cpu]!.run.submitted = t;
				events.push({ t, kind: 'submit', op: cpu });
			}
			cpu++;
			changed = true;
		}
		return changed;
	};

	/** Advances the GPU by one tick [t, t+1); returns true if any batch made progress. */
	const tick = (t: number): boolean => {
		let moved = false;
		for (const q of QUEUE_IDS) {
			const mine = ops.flatMap((o, i) => (o.kind === 'submit' && o.queue === q ? [i] : []));
			mine.forEach((i, n) => {
				const b = bs[i]!;
				if (b.run.submitted === null || b.run.done !== null) return;
				const stage = b.phases[b.phase]!.stage;
				// Started in submission order: the drawing choice of this lab (the spec allows even more freedom).
				const inOrder = mine.slice(0, n).every((j) => bs[j]!.run.started !== null);
				const blocker = mine
					.slice(0, n + 1)
					.flatMap((j) => (ops[j] as SubmitOp).waits.map((w, k) => ({ j, k, w })))
					.find(({ j, k, w }) => bs[j]!.waitDone[k] === null && expand(w.stage).includes(stage));
				if (!inOrder || blocker) {
					segs.push({ lane: q, op: i, from: t, to: t + 1, what: b.run.started === null ? 'pending' : 'stall' });
					if (blocker) {
						const id = `${blocker.j}:${blocker.k}:${stage}`;
						if (!b.blockedOn.has(id)) (b.blockedOn.add(id), events.push({ t, kind: 'blocked', op: i, sem: blocker.w.sem, stage }));
					}
					return;
				}
				if (b.run.started === null) (b.run.started = t), events.push({ t, kind: 'start', op: i });
				segs.push({ lane: q, op: i, from: t, to: t + 1, what: stage });
				moved = true;
				if (--b.left > 0) return;
				if (++b.phase < b.phases.length) b.left = b.phases[b.phase]!.len;
				else {
					b.run.done = t + 1;
					events.push({ t: t + 1, kind: 'done', op: i });
				}
			});
		}
		return moved;
	};

	let end = MAX_T;
	for (let t = 0; t < MAX_T; t++) {
		let changed = false;
		while (settle(t)) changed = true;
		if (cpu >= ops.length && bs.every((b) => !b || b.run.done !== null)) {
			end = t;
			break;
		}
		const moved = tick(t);
		const recording = cpu < ops.length && ops[cpu]!.kind === 'submit';
		// Nothing can change any more: a deadlock. The last tick drew the stalls, so the picture ends after it.
		if (!changed && !moved && !recording) {
			end = t + 1;
			break;
		}
	}

	if (cpu < ops.length) {
		const op = ops[cpu]!;
		if (op.kind === 'wait') {
			if (waitFrom !== null && end > waitFrom) segs.push({ lane: 'cpu', op: cpu, from: waitFrom, to: end, what: 'wait' });
			const never = !ops.slice(0, cpu).some((o) => o.kind === 'submit' && o.fence === op.fence);
			issues.push(issue(never ? 'fenceNeverSubmitted' : 'cpuBlocked', cpu, { fence: op.fence }));
		}
	}
	const noSignal = new Set(staticBinaryGaps(sc, pairs));
	ops.forEach((op, i) => {
		const b = bs[i];
		if (op.kind !== 'submit' || !b || b.run.submitted === null || b.run.done !== null || noSignal.has(i)) return;
		const open = op.waits.find((_, k) => b.waitDone[k] === null);
		issues.push(issue('neverRuns', i, { sem: open?.sem ?? '' }));
	});
	// Merge consecutive ticks of the same kind into one segment.
	const merged: Seg[] = [];
	for (const s of [...segs].sort((a, b) => a.lane.localeCompare(b.lane) || a.op - b.op || a.from - b.from)) {
		const last = merged.findLast((m) => m.lane === s.lane && m.op === s.op);
		if (last && last.what === s.what && last.to === s.from) last.to = s.to;
		else merged.push({ ...s });
	}
	return { end, segs: merged, events, links, batches: bs.map((b) => b?.run ?? null), issues };
}

/** Submits with a binary wait that has no signal to pair with (reported once as binaryNoSignal, not again as never running). */
function staticBinaryGaps(sc: Scenario, pairs: Map<string, string | null>): number[] {
	return sc.ops.flatMap((op, i) => (op.kind === 'submit' && op.waits.some((_, k) => pairs.get(key(i, k)) === null) ? [i] : []));
}

// ---------------------------------------------------------------------------------------------------------------
// Everything the view shows

export interface Analysis {
	run: Run;
	issues: Issue[];
	/** Sub-row of each submit inside its queue lane (overlapping batches stack). */
	rows: (number | null)[];
	/** Number of sub-rows per lane. */
	laneRows: Record<Lane, number>;
}

/** Runs and checks a scenario; issues sorted by the host op they are reported at. */
export function analyze(sc: Scenario): Analysis {
	const run = simulate(sc);
	const issues = [...staticIssues(sc), ...run.issues].sort((a, b) => a.op - b.op);
	const rows: (number | null)[] = sc.ops.map(() => null);
	const laneRows: Record<Lane, number> = { cpu: 1, graphics: 1, compute: 1, transfer: 1 };
	for (const q of QUEUE_IDS) {
		const busyUntil: number[] = [];
		sc.ops.forEach((op, i) => {
			const r = run.batches[i];
			if (op.kind !== 'submit' || op.queue !== q || !r || r.submitted === null) return;
			const to = r.done ?? run.end;
			let row = busyUntil.findIndex((u) => u <= r.submitted!);
			if (row < 0) row = busyUntil.push(0) - 1;
			busyUntil[row] = Math.max(to, r.submitted + 1);
			rows[i] = row;
		});
		laneRows[q] = Math.max(1, busyUntil.length);
	}
	return { run, issues, rows, laneRows };
}

/** Timeline frames for the scrubber: one per time step, with what happened at that step. */
export interface Frame {
	t: number;
	events: SimEvent[];
}
export function frames(run: Run): Frame[] {
	return Array.from({ length: run.end + 1 }, (_, t) => ({ t, events: run.events.filter((e) => e.t === t) }));
}

/**
 * Accesses of a race in the drawn schedule: the time spans of the two sides, to show whether they happened to
 * overlap this time (the race exists either way: nothing orders them).
 */
export function raceSpans(run: Run, i: Issue): { a: [number, number] | null; b: [number, number] | null } {
	const span = (op: number, stage: string): [number, number] | null => {
		if (stage === 'HOST') {
			const s = run.segs.find((s) => s.lane === 'cpu' && s.op === op && s.what === 'record');
			return s ? [s.from, s.to] : null;
		}
		const ss = run.segs.filter((s) => s.op === op && s.what === stage);
		return ss.length ? [Math.min(...ss.map((s) => s.from)), Math.max(...ss.map((s) => s.to))] : null;
	};
	return { a: span(i.others[0]!, String(i.vars.aStage)), b: span(i.op, String(i.vars.bStage)) };
}
