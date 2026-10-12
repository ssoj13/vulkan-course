/**
 * Model of the "frame assembly" trainer: the frame loop of step 10 (2 frames in flight, one graphics queue, a 3-image
 * swapchain) as host calls the reader makes one by one, a GPU queue and a presentation engine that advance when the
 * reader (or a blocking host wait) lets them, and the rules that refuse a wrong call.
 *
 * Objects (as step 10 creates them): per frame slot k a command buffer (ONE_TIME_SUBMIT, pool with
 * RESET_COMMAND_BUFFER), a fence and an `image_available` semaphore; per swapchain image i a `render_finished`
 * semaphore. Fences start "not created": the reader picks VK_FENCE_CREATE_SIGNALED_BIT or not.
 *
 * Rules and their sources (registry texts through `vuReport`, never pasted). Marked [run]: also reported by the real
 * validation layer on the course machine (2026-10-11, step 10 with a temporary fault, see AGENTS.md "Frame assembly
 * trainer"); the others are from validusage.json and the spec only:
 * - vkResetCommandBuffer / vkBeginCommandBuffer on a pending buffer: VUID-vkResetCommandBuffer-commandBuffer-00045 [run],
 *   VUID-vkBeginCommandBuffer-commandBuffer-00049 [run].
 * - vkQueueSubmit2: buffer not executable (VUID-vkQueueSubmit2-commandBuffer-03874) or pending without
 *   SIMULTANEOUS_USE (VUID-vkQueueSubmit2-commandBuffer-03875 [run]); fence signaled (VUID-vkQueueSubmit2-fence-04894) or
 *   still in use (VUID-vkQueueSubmit2-fence-04895); binary wait without an unclaimed submitted signal
 *   (VUID-vkQueueSubmit2-semaphore-03873); binary signal that cannot be proven unsignaled when it executes
 *   (VUID-vkQueueSubmit2-semaphore-03868 [run]; for a render_finished still waited on by a present the layer adds "may still
 *   be in use by VkSwapchainKHR ... was presented but was not re-acquired").
 * - vkAcquireNextImageKHR: semaphore signaled (VUID-vkAcquireNextImageKHR-semaphore-01286) or with a pending signal or
 *   wait (VUID-vkAcquireNextImageKHR-semaphore-01779 [run]); more than S - minImageCount images held with an infinite
 *   timeout (VUID-vkAcquireNextImageKHR-surface-07783, spec: "Acquiring too many images at once may block
 *   indefinitely, which is covered by valid usage when attempting to use UINT64_MAX").
 * - vkQueuePresentKHR: image not acquired or not transitioned by a submitted batch
 *   (VUID-VkPresentInfoKHR-pImageIndices-01430 [run]); wait without a submitted signal
 *   (VUID-vkQueuePresentKHR-pWaitSemaphores-03268).
 * - vkResetFences on a fence of a pending submission: VUID-vkResetFences-pFences-01123 [run].
 * - Not VUIDs, refused with a labeled explanation: vkWaitForFences(UINT64_MAX) on an unsignaled fence that no
 *   submission will signal (spec: it "will block and wait until the condition is satisfied or the timeout has
 *   expired" = forever); recording for an image the app does not hold (spec: "Use of a presentable image must occur
 *   only after the image is returned by vkAcquireNextImageKHR, and before it is released by vkQueuePresentKHR").
 * - Hazards (valid usage holds, synchronization validation reports them; accepted, logged as errors, the frame counts
 *   as broken): the batch does not wait on the image's acquire semaphore, or waits at BOTTOM_OF_PIPE ("equivalent to
 *   VK_PIPELINE_STAGE_2_NONE in the second scope"), so the UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL transition (src
 *   stage COLOR_ATTACHMENT_OUTPUT) is not chained after the acquire: WRITE_AFTER_READ; the present does not wait on
 *   the semaphore the rendering batch signals: PRESENT_AFTER_WRITE.
 *
 * Simplifications (stated on the lab page): the queue completes batches in submission order; the presentation engine
 * is FIFO (step 10 used MAILBOX on the course machine) and returns images in the order they were presented; the host
 * calls never fail (no OUT_OF_DATE). Pure module: no DOM, tested in tests/labs/frame-assembly.test.ts.
 */
import { vuReport } from './vuids';

/** Frames in flight (step 10's default). */
export const F = 2;
/** Swapchain images and minImageCount of the course machine (step 07: minImageCount 2, 3 images created). */
export const S = 3;
export const MIN_IMAGES = 2;

export const SLOTS = [0, 1] as const;
export const IMAGES = [0, 1, 2] as const;

/** Semaphore ids: `ia<k>` = image_available of slot k, `rf<i>` = render_finished of image i. */
export const SEMS = ['ia0', 'ia1', 'rf0', 'rf1', 'rf2'] as const;
export type SemId = (typeof SEMS)[number];
export const semName = (s: string) => (s.startsWith('ia') ? `image_available[${s.slice(2)}]` : `render_finished[${s.slice(2)}]`);

/** Wait stages offered at the submit (all measured in step 10 with sync validation, except ALL_COMMANDS: spec). */
export const STAGES = ['TOP_OF_PIPE', 'COLOR_ATTACHMENT_OUTPUT', 'ALL_COMMANDS', 'BOTTOM_OF_PIPE'] as const;
export type Stage = (typeof STAGES)[number];
export const stageName = (s: Stage) => `VK_PIPELINE_STAGE_2_${s}_BIT`;

/**
 * A binary semaphore as the app can know it. `sig`: no signal / a submitted signal not executed yet / signaled.
 * `claim`: the wait that takes the current signal (a batch not yet executed, or a present whose wait the app cannot
 * see complete until the same image is acquired again).
 */
export interface Sem {
	sig: 'none' | 'pending' | 'signaled';
	claim: null | { by: 'batch'; id: number } | { by: 'present'; img: number };
}
export interface Fence {
	created: boolean;
	signaled: boolean;
	/** Id of the pending batch that will signal it. */
	batch: number | null;
}
export type CbState = 'initial' | 'executable' | 'pending' | 'invalid';
export interface Cb {
	state: CbState;
	/** Image the recorded commands render to. */
	image: number | null;
}
export interface Img {
	/** free = not held by anyone, app = acquired, queued = presented and not shown yet, shown = on screen. */
	owner: 'free' | 'app' | 'queued' | 'shown';
	/** Semaphore the acquire signals for this image (while the app holds it). */
	acq: string | null;
	/** True once a batch rendering it waited on `acq`. */
	acqWaited: boolean;
	/** Last batch that rendered it since the acquire. */
	rendered: number | null;
	/** Semaphore whose pending acquire signal fires when the presentation engine releases the image. */
	release: string | null;
}
export interface Batch {
	id: number;
	slot: number;
	image: number;
	wait: string;
	stage: Stage;
	signal: string;
	fence: string;
	hazard: boolean;
	done: boolean;
}
export interface FrameState {
	sems: Readonly<Record<string, Sem>>;
	fences: readonly Fence[];
	cbs: readonly Cb[];
	imgs: readonly Img[];
	/** Images not held by the app, in the order the presentation engine gives them back (oldest present first). */
	order: readonly number[];
	/** Image on screen and images presented after it, in order. */
	screen: number | null;
	queue: readonly number[];
	/** Every batch submitted (done ones included: the console and goals read them). */
	batches: readonly Batch[];
	/** Presented frames: slot of the rendering batch, image, and whether a hazard broke it. */
	frames: readonly { slot: number; image: number; hazard: boolean }[];
	/** Batches of both slots were pending at the same time at least once. */
	overlap: boolean;
}

export const INIT: FrameState = {
	sems: Object.fromEntries(SEMS.map((s) => [s, { sig: 'none', claim: null }])),
	fences: SLOTS.map(() => ({ created: false, signaled: false, batch: null })),
	cbs: SLOTS.map(() => ({ state: 'initial', image: null })),
	imgs: IMAGES.map(() => ({ owner: 'free', acq: null, acqWaited: false, rendered: null, release: null })),
	order: [...IMAGES],
	screen: null,
	queue: [],
	batches: [],
	frames: [],
	overlap: false,
};

export type LogKind = 'call' | 'error' | 'warn' | 'info';
/** Simulator remarks; the trainer words them (frame-assembly.text.ts `m_*`). */
export type Msg =
	| 'deadlock'
	| 'timeout'
	| 'blocked'
	| 'gpuRan'
	| 'peShown'
	| 'peReleased'
	| 'gpuIdle'
	| 'gpuWaits'
	| 'peNothing'
	| 'peWaits'
	| 'noImage'
	| 'notHeld'
	| 'slowStage'
	| 'semInUseByPresent'
	| 'tooManyImages'
	| 'hazardAcquire'
	| 'hazardStage'
	| 'hazardPresent'
	| 'pendingAcquire'
	| 'stuck';
/**
 * One console line: calls and validation reports verbatim English (`text`), remarks as `msg` + values. `vuid` = the
 * quoted rule, `sync` = the synchronization-validation hazard name.
 */
export type Line = { kind: LogKind; text: string; vuid?: string; sync?: string } | { kind: LogKind; msg: Msg; vars: Record<string, string | number> };
export interface Step {
	state: FrameState;
	log: Line[];
}

const vu = (id: string, call: string): Line => ({ kind: 'error', vuid: id, text: vuReport(id, call) });
const call = (text: string): Line => ({ kind: 'call', text });
const msg = (kind: LogKind, m: Msg, vars: Record<string, string | number> = {}): Line => ({ kind, msg: m, vars });

/**
 * Hazard reports of synchronization validation (VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true), copied from step 10 runs
 * with a temporary fault on the course machine (2026-10-11), handles cut out.
 */
export const SYNC = {
	acquire: {
		id: 'SYNC-HAZARD-WRITE-AFTER-READ',
		text: 'vkQueueSubmit2(): WRITE_AFTER_READ hazard detected. vkCmdPipelineBarrier2 (from VkCommandBuffer ... submitted on the current VkQueue ...) writes to VkImage ..., which was previously read by vkAcquireNextImageKHR. No sufficient synchronization is present to ensure that a layout transition does not conflict with a prior swapchain image acquire operation.',
	},
	present: {
		id: 'SYNC-HAZARD-PRESENT-AFTER-WRITE',
		text: 'vkQueuePresentKHR(): PRESENT_AFTER_WRITE hazard detected. vkQueuePresentKHR presents swapchain image {i} (VkImage ... from VkSwapchainKHR ...), which was previously written during an image layout transition initiated by vkCmdPipelineBarrier2. No sufficient synchronization is present to ensure that a swapchain present operation does not conflict with a prior layout transition.',
	},
} as const;

const slotOf = (id: string) => Number(id.slice(-1));
const pendingBatches = (s: FrameState) => s.batches.filter((b) => !b.done);
/** Images the app holds and has not presented: what a command buffer can be recorded for. */
export const heldImages = (s: FrameState): number[] => IMAGES.filter((i) => s.imgs[i]!.owner === 'app');
export const createdFences = (s: FrameState): string[] => SLOTS.filter((k) => s.fences[k]!.created).map((k) => `f${k}`);

// Small immutable updates (the state is tiny; clarity over cleverness).
const setSem = (s: FrameState, id: string, v: Partial<Sem>): FrameState => ({ ...s, sems: { ...s.sems, [id]: { ...s.sems[id]!, ...v } } });
const setFence = (s: FrameState, k: number, v: Partial<Fence>): FrameState => ({ ...s, fences: s.fences.map((f, i) => (i === k ? { ...f, ...v } : f)) });
const setCb = (s: FrameState, k: number, v: Partial<Cb>): FrameState => ({ ...s, cbs: s.cbs.map((c, i) => (i === k ? { ...c, ...v } : c)) });
const setImg = (s: FrameState, n: number, v: Partial<Img>): FrameState => ({ ...s, imgs: s.imgs.map((m, i) => (i === n ? { ...m, ...v } : m)) });
const setBatch = (s: FrameState, id: number, v: Partial<Batch>): FrameState => ({ ...s, batches: s.batches.map((b) => (b.id === id ? { ...b, ...v } : b)) });

/** The semaphore may receive a new signal: no signal, no wait the app cannot see completed. */
const semIdle = (x: Sem) => x.sig === 'none' && x.claim === null;
/** The semaphore holds a submitted signal no other wait has taken. */
const semWaitable = (x: Sem) => x.sig !== 'none' && x.claim === null;

// ---------------------------------------------------------------------------------------------------------------
// Fences

export function createFence(s: FrameState, k: number, signaled: boolean): Step {
	return {
		state: setFence(s, k, { created: true, signaled }),
		log: [call(`vkCreateFence(flags = ${signaled ? 'VK_FENCE_CREATE_SIGNALED_BIT' : '0'}) -> in_flight[${k}]`)],
	};
}

/**
 * Why vkWaitForFences on fence k with an infinite timeout would never return: no pending submission signals the fence,
 * or the GPU and the presentation engine can never get that far. Empty when the wait returns.
 */
export function waitBlockers(s: FrameState, k: number, infinite: boolean): Line[] {
	const f = s.fences[k]!;
	if (!infinite || f.signaled) return [];
	if (f.batch === null || !runUntil(s, (x) => x.fences[k]!.signaled)) return [msg('error', 'deadlock', { fence: `in_flight[${k}]` })];
	return [];
}

export function waitFence(s: FrameState, k: number, infinite: boolean): Step {
	const f = s.fences[k]!;
	const head = call(`vkWaitForFences(in_flight[${k}], timeout = ${infinite ? 'UINT64_MAX' : '0'})`);
	if (f.signaled) return { state: s, log: [head, call('-> VK_SUCCESS')] };
	if (!infinite || f.batch === null) return { state: s, log: [head, call('-> VK_TIMEOUT'), msg('info', 'timeout', { fence: `in_flight[${k}]` })] };
	const r = runUntil(s, (x) => x.fences[k]!.signaled)!;
	return { state: r.state, log: [head, msg('info', 'blocked'), ...r.log, call('-> VK_SUCCESS')] };
}

export function resetBlockers(s: FrameState, k: number): Line[] {
	return s.fences[k]!.batch !== null ? [vu('VUID-vkResetFences-pFences-01123', 'vkResetFences')] : [];
}

export function resetFence(s: FrameState, k: number): Step {
	return { state: setFence(s, k, { signaled: false }), log: [call(`vkResetFences(in_flight[${k}])`)] };
}

// ---------------------------------------------------------------------------------------------------------------
// Acquire

export function acquireBlockers(s: FrameState, sem: string): Line[] {
	const out: Line[] = [];
	const x = s.sems[sem]!;
	// A submitted wait takes the signal: the layer reports the pending wait (01779), not the payload (01286); seen in
	// step 10 without its fence wait.
	if (x.claim?.by === 'batch' || (x.claim === null && x.sig === 'pending')) out.push(vu('VUID-vkAcquireNextImageKHR-semaphore-01779', 'vkAcquireNextImageKHR'));
	else if (x.claim === null && x.sig === 'signaled') out.push(vu('VUID-vkAcquireNextImageKHR-semaphore-01286', 'vkAcquireNextImageKHR'));
	else if (x.claim?.by === 'present') out.push(vu('VUID-vkAcquireNextImageKHR-semaphore-01779', 'vkAcquireNextImageKHR'), msg('info', 'semInUseByPresent', { sem: semName(sem), i: x.claim.img }));
	if (s.imgs[s.order[0]!]!.owner === 'queued' && !runUntil(s, (x) => x.imgs[x.order[0]!]!.owner !== 'queued')) out.push(msg('error', 'stuck'));
	if (heldImages(s).length > S - MIN_IMAGES) out.push(vu('VUID-vkAcquireNextImageKHR-surface-07783', 'vkAcquireNextImageKHR'), msg('info', 'tooManyImages', { held: heldImages(s).length, s: S, m: MIN_IMAGES }));
	return out;
}

/**
 * vkAcquireNextImageKHR(UINT64_MAX, sem): the image the presentation engine gives back first. A free image signals the
 * semaphore at once; the image on screen signals it when the next present replaces it. An image presented but not
 * shown yet cannot come back: the call blocks while the engine (and the GPU feeding it) moves on.
 */
export function acquire(s: FrameState, sem: string): Step {
	const log: Line[] = [];
	let st = s;
	const head = () => st.order[0]!;
	if (st.imgs[head()]!.owner === 'queued') {
		const r = runUntil(st, (x) => x.imgs[x.order[0]!]!.owner !== 'queued')!;
		log.push(msg('info', 'blocked'), ...r.log);
		st = r.state;
	}
	const i = head();
	const img = st.imgs[i]!;
	const now = img.owner === 'free';
	st = { ...st, order: st.order.slice(1) };
	st = setImg(st, i, { owner: 'app', acq: sem, acqWaited: false, rendered: null, release: now ? null : sem });
	st = setSem(st, sem, { sig: now ? 'signaled' : 'pending' });
	// The image is back, so the present that showed it has finished waiting on its semaphore.
	for (const id of SEMS) {
		const c = st.sems[id]!.claim;
		if (c?.by === 'present' && c.img === i) st = setSem(st, id, { claim: null });
	}
	log.push(call(`vkAcquireNextImageKHR(UINT64_MAX, ${semName(sem)}) -> image ${i}`));
	if (!now) log.push(msg('info', 'pendingAcquire', { i, sem: semName(sem) }));
	return { state: st, log };
}

// ---------------------------------------------------------------------------------------------------------------
// Command buffers

export function recordBlockers(s: FrameState, k: number, image: number | null): Line[] {
	const out: Line[] = [];
	if (s.cbs[k]!.state === 'pending') out.push(vu('VUID-vkResetCommandBuffer-commandBuffer-00045', 'vkResetCommandBuffer'), vu('VUID-vkBeginCommandBuffer-commandBuffer-00049', 'vkBeginCommandBuffer'));
	if (image === null) out.push(msg('error', 'noImage'));
	return out;
}

export function record(s: FrameState, k: number, image: number): Step {
	return {
		state: setCb(s, k, { state: 'executable', image }),
		log: [
			call(`vkResetCommandBuffer(cmd[${k}]); vkBeginCommandBuffer(cmd[${k}], ONE_TIME_SUBMIT)`),
			call(`vkCmdPipelineBarrier2(image ${image}: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL, src COLOR_ATTACHMENT_OUTPUT)`),
			call('vkCmdBeginRendering / vkCmdDraw / vkCmdEndRendering'),
			call(`vkCmdPipelineBarrier2(image ${image}: COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR); vkEndCommandBuffer(cmd[${k}])`),
		],
	};
}

export interface SubmitOpts {
	wait: string;
	stage: Stage;
	signal: string;
	/** `f<k>`, or '' = VK_NULL_HANDLE. */
	fence: string;
}

export function submitBlockers(s: FrameState, k: number, o: SubmitOpts): Line[] {
	const out: Line[] = [];
	const cb = s.cbs[k]!;
	if (cb.state === 'pending') out.push(vu('VUID-vkQueueSubmit2-commandBuffer-03875', 'vkQueueSubmit2'));
	else if (cb.state !== 'executable') out.push(vu('VUID-vkQueueSubmit2-commandBuffer-03874', 'vkQueueSubmit2'));
	else if (s.imgs[cb.image!]!.owner !== 'app') out.push(msg('error', 'notHeld', { i: cb.image! }));
	if (o.fence) {
		const f = s.fences[slotOf(o.fence)]!;
		if (f.signaled) out.push(vu('VUID-vkQueueSubmit2-fence-04894', 'vkQueueSubmit2'));
		else if (f.batch !== null) out.push(vu('VUID-vkQueueSubmit2-fence-04895', 'vkQueueSubmit2'));
	}
	const w = s.sems[o.wait]!;
	if (!semWaitable(w)) out.push(vu('VUID-vkQueueSubmit2-semaphore-03873', 'vkQueueSubmit2'));
	// The batch's own wait unsignals its semaphore before the batch's signal executes.
	const sg: Sem = o.signal === o.wait && semWaitable(w) ? { sig: 'none', claim: null } : s.sems[o.signal]!;
	if (!semIdle(sg)) {
		out.push(vu('VUID-vkQueueSubmit2-semaphore-03868', 'vkQueueSubmit2'));
		if (sg.claim?.by === 'present') out.push(msg('info', 'semInUseByPresent', { sem: semName(o.signal), i: sg.claim.img }));
	}
	return out;
}

/** Whether a batch rendering `image` and waiting on `wait` at `stage` is chained after the image's acquire. */
function acquireChained(s: FrameState, image: number, wait: string, stage: Stage): boolean {
	const img = s.imgs[image]!;
	if (img.acqWaited) return true; // an earlier batch of this acquire already waited on it
	return wait === img.acq && stage !== 'BOTTOM_OF_PIPE';
}

export function submit(s: FrameState, k: number, o: SubmitOpts): Step {
	const image = s.cbs[k]!.image!;
	const hazard = !acquireChained(s, image, o.wait, o.stage);
	const id = s.batches.length;
	const others = pendingBatches(s).some((b) => b.slot !== k);
	let st: FrameState = { ...s, batches: [...s.batches, { id, slot: k, image, wait: o.wait, stage: o.stage, signal: o.signal, fence: o.fence, hazard, done: false }], overlap: s.overlap || others };
	st = setCb(st, k, { state: 'pending' });
	st = setSem(st, o.wait, { claim: { by: 'batch', id } });
	if (o.signal !== o.wait) st = setSem(st, o.signal, { sig: 'pending' });
	if (o.fence) st = setFence(st, slotOf(o.fence), { batch: id });
	st = setImg(st, image, { rendered: id, acqWaited: st.imgs[image]!.acqWaited || o.wait === st.imgs[image]!.acq });
	const log: Line[] = [call(`vkQueueSubmit2(cmd[${k}], wait ${semName(o.wait)} at ${stageName(o.stage)}, signal ${semName(o.signal)}, fence ${o.fence ? `in_flight[${slotOf(o.fence)}]` : 'VK_NULL_HANDLE'}) -> batch #${id}`)];
	if (hazard) log.push({ kind: 'error', text: SYNC.acquire.text, sync: SYNC.acquire.id }, o.wait === s.imgs[image]!.acq ? msg('info', 'hazardStage', { i: image, stage: stageName(o.stage) }) : msg('info', 'hazardAcquire', { i: image, sem: semName(s.imgs[image]!.acq!) }));
	else if (o.stage === 'TOP_OF_PIPE' || o.stage === 'ALL_COMMANDS') log.push(msg('info', 'slowStage', { stage: stageName(o.stage) }));
	return { state: st, log };
}

// ---------------------------------------------------------------------------------------------------------------
// Present

export function presentBlockers(s: FrameState, i: number, wait: string): Line[] {
	const out: Line[] = [];
	const img = s.imgs[i]!;
	if (img.owner !== 'app' || img.rendered === null) out.push(vu('VUID-VkPresentInfoKHR-pImageIndices-01430', 'vkQueuePresentKHR'));
	if (!semWaitable(s.sems[wait]!)) out.push(vu('VUID-vkQueuePresentKHR-pWaitSemaphores-03268', 'vkQueuePresentKHR'));
	return out;
}

export function present(s: FrameState, i: number, wait: string): Step {
	const b = s.batches[s.imgs[i]!.rendered!]!;
	const badWait = wait !== b.signal;
	let st: FrameState = { ...s, queue: [...s.queue, i], order: [...s.order, i], frames: [...s.frames, { slot: b.slot, image: i, hazard: b.hazard || badWait }] };
	st = setImg(st, i, { owner: 'queued', acq: null });
	st = setSem(st, wait, { claim: { by: 'present', img: i } });
	const log: Line[] = [call(`vkQueuePresentKHR(image ${i}, wait ${semName(wait)}) -> VK_SUCCESS`)];
	if (badWait) log.push({ kind: 'error', text: SYNC.present.text.replace('{i}', String(i)), sync: SYNC.present.id }, msg('info', 'hazardPresent', { i, sem: semName(b.signal) }));
	return { state: st, log };
}

// ---------------------------------------------------------------------------------------------------------------
// Time: the GPU queue and the presentation engine

/** The oldest pending batch, and the semaphore it still waits for (null = it can run). */
function nextBatch(s: FrameState): { b: Batch; blockedBy: string | null } | null {
	const b = pendingBatches(s)[0];
	if (!b) return null;
	return { b, blockedBy: s.sems[b.wait]!.sig === 'signaled' ? null : b.wait };
}

export function gpuBlockers(s: FrameState): Line[] {
	const n = nextBatch(s);
	if (!n) return [msg('info', 'gpuIdle')];
	if (n.blockedBy) {
		const img = s.imgs.findIndex((m) => m.release === n.blockedBy);
		return [msg('info', 'gpuWaits', { n: n.b.id, sem: semName(n.blockedBy), i: img < 0 ? '?' : img })];
	}
	return [];
}

/** The GPU executes the oldest pending batch: its wait completes, its signal and fence fire. */
export function gpuStep(s: FrameState): Step {
	const { b } = nextBatch(s)!;
	let st = setBatch(s, b.id, { done: true });
	st = setSem(st, b.wait, { sig: 'none', claim: null });
	st = setSem(st, b.signal, { sig: 'signaled' });
	st = setCb(st, b.slot, { state: 'invalid' });
	if (b.fence) st = setFence(st, slotOf(b.fence), { signaled: true, batch: null });
	return { state: st, log: [msg('info', 'gpuRan', { n: b.id, slot: b.slot, i: b.image, sem: semName(b.signal), fence: b.fence ? `in_flight[${slotOf(b.fence)}]` : '-' })] };
}

/** The image the engine shows at the next vblank, and the semaphore its present still waits for. */
function nextShown(s: FrameState): { i: number; waits: string | null } | null {
	const i = s.queue[0];
	if (i === undefined) return null;
	const sem = SEMS.find((id) => { const c = s.sems[id]!.claim; return c?.by === 'present' && c.img === i; });
	// The present's wait completes when its semaphore is signaled.
	return { i, waits: sem && s.sems[sem]!.sig !== 'signaled' ? sem : null };
}

export function peBlockers(s: FrameState): Line[] {
	const n = nextShown(s);
	if (!n) return [msg('info', 'peNothing', { i: s.screen ?? '-' })];
	if (n.waits) return [msg('info', 'peWaits', { i: n.i, sem: semName(n.waits) })];
	return [];
}

/** Vblank: the next presented image goes on screen; the one it replaces returns to the pool. */
export function peStep(s: FrameState): Step {
	const { i } = nextShown(s)!;
	let st: FrameState = { ...s, queue: s.queue.slice(1), screen: i };
	const sem = SEMS.find((id) => { const c = st.sems[id]!.claim; return c?.by === 'present' && c.img === i; });
	// The present's wait has executed; the app still cannot see it (the claim stays until image i is acquired again).
	if (sem) st = setSem(st, sem, { sig: 'none' });
	st = setImg(st, i, { owner: 'shown' });
	const log: Line[] = [msg('info', 'peShown', { i })];
	const old = s.screen;
	if (old !== null) {
		const o = st.imgs[old]!;
		if (o.owner === 'shown') st = setImg(st, old, { owner: 'free' });
		if (o.release) {
			st = setSem(st, o.release, { sig: 'signaled' });
			st = setImg(st, old, { release: null });
		}
		log.push(msg('info', 'peReleased', { i: old, sem: o.release ? semName(o.release) : '-' }));
	}
	return { state: st, log };
}

/**
 * What happens while the host blocks (vkWaitForFences, vkAcquireNextImageKHR): the GPU and the presentation engine
 * move on, GPU first, until `done` holds. Null when neither can move before that: the wait would never return (the
 * rules refuse such waits; every step either completes a batch or shows an image, so the loop is finite).
 */
export function runUntil(s: FrameState, done: (s: FrameState) => boolean): Step | null {
	let st = s;
	const log: Line[] = [];
	while (!done(st)) {
		const r = gpuBlockers(st).length === 0 ? gpuStep(st) : peBlockers(st).length === 0 ? peStep(st) : null;
		if (!r) return null;
		st = r.state;
		log.push(...r.log);
	}
	return { state: st, log };
}

// ---------------------------------------------------------------------------------------------------------------
// Words on the cards

export function semState(s: FrameState, id: string): 'unsignaled' | 'pending' | 'signaled' | 'waited' | 'present' {
	const x = s.sems[id]!;
	if (x.claim?.by === 'present') return 'present';
	if (x.claim?.by === 'batch') return 'waited';
	return x.sig === 'none' ? 'unsignaled' : x.sig;
}
export function fenceState(s: FrameState, k: number): 'none' | 'signaled' | 'unsignaled' | 'pending' {
	const f = s.fences[k]!;
	if (!f.created) return 'none';
	if (f.signaled) return 'signaled';
	return f.batch !== null ? 'pending' : 'unsignaled';
}

/** Presented frames per slot, and whether any of them was broken by a hazard. */
export function frameCounts(s: FrameState): { perSlot: number[]; hazards: number } {
	return { perSlot: SLOTS.map((k) => s.frames.filter((f) => f.slot === k).length), hazards: s.frames.filter((f) => f.hazard).length };
}
