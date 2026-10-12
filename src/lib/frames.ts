/**
 * Model of the "frame assembly" trainer: the frame loop of step 10 (2 frames in flight, one graphics queue, a 3-image
 * swapchain) as host calls the reader makes one by one, a GPU queue and a presentation engine that advance when the
 * reader (or a blocking host wait) lets them, and the rules that refuse a wrong call.
 *
 * Objects (as step 10 creates them): per frame slot k a command buffer (ONE_TIME_SUBMIT, pool with
 * RESET_COMMAND_BUFFER), a fence and an `image_available` semaphore; per swapchain image i a `render_finished`
 * semaphore. Fences start "not created": the reader picks VK_FENCE_CREATE_SIGNALED_BIT or not.
 *
 * Two kinds of truth. The GPU and the presentation engine know when a batch, a semaphore wait or a signal has executed
 * (`Batch.done`, `Sig.fired`, `Sig.used`, `Fence.signaled`). The host knows a batch has completed only once a fence
 * wait returned VK_SUCCESS for that batch or a later one on the queue (`Batch.seen`; the spec: "Before a fence or
 * semaphore is signaled, it is guaranteed that any previously submitted queue operations have completed execution").
 * Until then the command buffer stays pending, the fence stays in use and the semaphore keeps the batch's wait, as far
 * as the host (and the validation layer, which tracks completion through the same waits) can tell. A present's wait is
 * known to be over only when the same image is acquired again. The rules about host calls read the host's view.
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
 *   be in use by VkSwapchainKHR ... was presented but was not re-acquired"). A signal behind a wait of an earlier batch
 *   on the same queue is fine: that wait executes first ("The first synchronization scope of each semaphore signal
 *   operation ... includes ... all commands that occur earlier in submission order"), so the semaphore is unsignaled
 *   when the new signal executes.
 * - vkAcquireNextImageKHR: semaphore signaled (VUID-vkAcquireNextImageKHR-semaphore-01286) or with a pending signal or
 *   wait (VUID-vkAcquireNextImageKHR-semaphore-01779 [run]); with UINT64_MAX while the app holds more than S - M images
 *   (VUID-vkAcquireNextImageKHR-surface-07783; spec: vkAcquireNextImageKHR "should not be called if the number of images
 *   that the application has currently acquired is greater than S-M", and "Acquiring too many images at once may block
 *   indefinitely, which is covered by valid usage when attempting to use UINT64_MAX").
 * - vkQueuePresentKHR: image not acquired or not transitioned by a submitted batch
 *   (VUID-VkPresentInfoKHR-pImageIndices-01430 [run]); wait without a submitted signal
 *   (VUID-vkQueuePresentKHR-pWaitSemaphores-03268).
 * - vkResetFences on a fence of a submission the host has not seen complete: VUID-vkResetFences-pFences-01123 [run].
 * - Not VUIDs, refused with a labeled explanation: vkWaitForFences(UINT64_MAX) on an unsignaled fence that no
 *   submission will signal (spec: it "will block and wait until the condition is satisfied or the timeout has
 *   expired" = forever); submitting commands that render to an image the app does not hold (spec: "Use of a
 *   presentable image must occur only after the image is returned by vkAcquireNextImageKHR, and before it is released
 *   by vkQueuePresentKHR"; recording does not touch the image, executing does, so the trainer checks at the submit).
 * - Hazards (valid usage holds, synchronization validation reports them; accepted, logged as errors, the frame counts
 *   as broken): no batch rendering the image is chained after its acquire (the batch neither waits on the acquire
 *   semaphore nor follows in submission order a batch that did, or it waits at BOTTOM_OF_PIPE, "equivalent to
 *   VK_PIPELINE_STAGE_2_NONE in the second scope"), so the UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL transition (src stage
 *   COLOR_ATTACHMENT_OUTPUT) is not ordered after the acquire: WRITE_AFTER_READ, and the frame stays broken whatever
 *   later batches do; the present waits on no signal of the image's last rendering batch or of a batch after it in
 *   submission order: PRESENT_AFTER_WRITE.
 *
 * Simplifications (stated in FrameAssembly.astro's caption, so on every page that embeds the trainer): the queue
 * completes batches in submission order; the presentation engine is FIFO (step 10 used MAILBOX on the course machine)
 * and returns images in the order they were presented; the host learns of completions only through vkWaitForFences;
 * the host calls never fail (no OUT_OF_DATE). Pure module: no DOM, tested in tests/labs/frame-assembly.test.ts.
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
 * One signal operation on a binary semaphore. It leaves the semaphore once the host knows the wait that took it is
 * over: the waiting batch was seen complete, or the image whose present waited was acquired again.
 */
export interface Sig {
	/** The signaling operation: a batch, or the acquire of an image. */
	from: { by: 'batch'; id: number } | { by: 'acquire'; img: number };
	/** The signal has executed. */
	fired: boolean;
	/** The wait that takes this signal: a submitted batch, or a present of image `img`. */
	claim: null | { by: 'batch'; id: number } | { by: 'present'; img: number };
	/** That wait has executed (the semaphore is unsignaled again, whether or not the host knows it). */
	used: boolean;
}
/** A binary semaphore: its signal operations, oldest first; none = unsignaled as far as the host knows. */
export interface Sem {
	sigs: readonly Sig[];
}
export interface Fence {
	created: boolean;
	signaled: boolean;
	/** The batch last submitted with it (null when none, or after vkResetFences). */
	batch: number | null;
}
/** Command buffer states as the host knows them: `pending` until the host saw its batch complete. */
export type CbState = 'initial' | 'executable' | 'pending' | 'invalid';
export interface Cb {
	state: CbState;
	/** Image the recorded commands render to. */
	image: number | null;
	/** Batch that executes it while pending. */
	batch: number | null;
}
export interface Img {
	/** free = not held by anyone, app = acquired, queued = presented and not shown yet, shown = on screen. */
	owner: 'free' | 'app' | 'queued' | 'shown';
	/** Semaphore the acquire signals for this image (while the app holds it). */
	acq: string | null;
	/** A batch rendering it since the acquire is chained after the acquire: later batches are too (submission order). */
	acqWaited: boolean;
	/** A batch rendering it since the acquire was not chained after the acquire: the frame is broken. */
	broken: boolean;
	/** Last batch that rendered it since the acquire. */
	rendered: number | null;
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
	/** Executed by the GPU. */
	done: boolean;
	/** The host saw it complete (a fence wait returned VK_SUCCESS for it or a later batch). */
	seen: boolean;
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
	sems: Object.fromEntries(SEMS.map((s) => [s, { sigs: [] }])),
	fences: SLOTS.map(() => ({ created: false, signaled: false, batch: null })),
	cbs: SLOTS.map(() => ({ state: 'initial', image: null, batch: null })),
	imgs: IMAGES.map(() => ({ owner: 'free', acq: null, acqWaited: false, broken: false, rendered: null })),
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
	| 'noPick'
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
/** A stored run can carry an option value no longer offered as an object ('' = none picked): refused, never thrown. */
const noPick = (): Line[] => [msg('error', 'noPick')];

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
/** Images the app holds and has not presented. */
export const heldImages = (s: FrameState): number[] => IMAGES.filter((i) => s.imgs[i]!.owner === 'app');
/** Images a command buffer can be recorded for: any (recording does not touch the image), held ones first. */
export const recordImages = (s: FrameState): number[] => [...heldImages(s), ...IMAGES.filter((i) => s.imgs[i]!.owner !== 'app')];
export const createdFences = (s: FrameState): string[] => SLOTS.filter((k) => s.fences[k]!.created).map((k) => `f${k}`);

// Small immutable updates (the state is tiny; clarity over cleverness).
const setFence = (s: FrameState, k: number, v: Partial<Fence>): FrameState => ({ ...s, fences: s.fences.map((f, i) => (i === k ? { ...f, ...v } : f)) });
const setCb = (s: FrameState, k: number, v: Partial<Cb>): FrameState => ({ ...s, cbs: s.cbs.map((c, i) => (i === k ? { ...c, ...v } : c)) });
const setImg = (s: FrameState, n: number, v: Partial<Img>): FrameState => ({ ...s, imgs: s.imgs.map((m, i) => (i === n ? { ...m, ...v } : m)) });
const setBatch = (s: FrameState, id: number, v: Partial<Batch>): FrameState => ({ ...s, batches: s.batches.map((b) => (b.id === id ? { ...b, ...v } : b)) });
/** Rewrites the signal operations of every semaphore (`f` returns the new list for one semaphore). */
const mapSigs = (s: FrameState, f: (sigs: readonly Sig[], id: string) => readonly Sig[]): FrameState => ({
	...s,
	sems: Object.fromEntries(Object.entries(s.sems).map(([id, x]) => [id, { sigs: f(x.sigs, id) }])),
});
const sameClaim = (a: Sig['claim'], b: NonNullable<Sig['claim']>) => a !== null && a.by === b.by && (a.by === 'batch' ? a.id === (b as { id: number }).id : a.img === (b as { img: number }).img);

/** The signal a new wait takes: the oldest one no wait has taken. */
const unclaimed = (x: Sem): Sig | undefined => x.sigs.find((g) => g.claim === null);
/** Gives the oldest unclaimed signal of `sem` to the wait `claim`. */
function claimSig(s: FrameState, sem: string, claim: NonNullable<Sig['claim']>): FrameState {
	const k = s.sems[sem]!.sigs.findIndex((g) => g.claim === null);
	return mapSigs(s, (sigs, id) => (id === sem ? sigs.map((g, i) => (i === k ? { ...g, claim } : g)) : sigs));
}
/** The signal a wait took, with its semaphore. */
function claimedBy(s: FrameState, claim: NonNullable<Sig['claim']>): { sem: string; sig: Sig } | null {
	for (const sem of SEMS) {
		const sig = s.sems[sem]!.sigs.find((g) => sameClaim(g.claim, claim));
		if (sig) return { sem, sig };
	}
	return null;
}
/**
 * A new signal from a batch is proven to execute on an unsignaled semaphore: every signal it has is taken by a wait
 * of an earlier batch on this queue, which executes first (submission order). A present's wait gives no such order.
 */
const signalable = (sigs: readonly Sig[]) => sigs.every((g) => g.claim?.by === 'batch');

/** The host saw batch `id` complete: so are all batches before it on this queue (they completed first). */
function see(s: FrameState, id: number): FrameState {
	let st: FrameState = { ...s, batches: s.batches.map((b) => (b.id <= id ? { ...b, seen: true } : b)) };
	const seen = (n: number | null) => n !== null && st.batches[n]!.seen;
	st = { ...st, cbs: st.cbs.map((c) => (c.state === 'pending' && seen(c.batch) ? { ...c, state: 'invalid', batch: null } : c)) };
	return mapSigs(st, (sigs) => sigs.filter((g) => !(g.claim?.by === 'batch' && seen(g.claim.id))));
}

// ---------------------------------------------------------------------------------------------------------------
// Fences

export function createFence(s: FrameState, k: number, signaled: boolean): Step {
	return {
		state: setFence(s, k, { created: true, signaled }),
		log: [call(`vkCreateFence(flags = ${signaled ? 'VK_FENCE_CREATE_SIGNALED_BIT' : '0'}) -> in_flight[${k}]`)],
	};
}

/**
 * Why vkWaitForFences on fence k with an infinite timeout would never return: no submission signals the fence, or the
 * GPU and the presentation engine can never get that far. Empty when the wait returns.
 */
export function waitBlockers(s: FrameState, k: number, infinite: boolean): Line[] {
	const f = s.fences[k]!;
	if (!infinite || f.signaled) return [];
	if (f.batch === null || !runUntil(s, (x) => x.fences[k]!.signaled)) return [msg('error', 'deadlock', { fence: `in_flight[${k}]` })];
	return [];
}

/** vkWaitForFences: VK_SUCCESS tells the host that the fence's batch, and every batch before it, has completed. */
export function waitFence(s: FrameState, k: number, infinite: boolean): Step {
	const head = call(`vkWaitForFences(in_flight[${k}], timeout = ${infinite ? 'UINT64_MAX' : '0'})`);
	const ok = (st: FrameState, log: Line[]): Step => {
		const b = st.fences[k]!.batch;
		return { state: b === null ? st : see(st, b), log: [...log, call('-> VK_SUCCESS')] };
	};
	const f = s.fences[k]!;
	if (f.signaled) return ok(s, [head]);
	if (!infinite || f.batch === null) return { state: s, log: [head, call('-> VK_TIMEOUT'), msg('info', 'timeout', { fence: `in_flight[${k}]` })] };
	const r = runUntil(s, (x) => x.fences[k]!.signaled)!;
	return ok(r.state, [head, msg('info', 'blocked'), ...r.log]);
}

/** The fence belongs to a submission the host has not seen complete. */
const fenceInUse = (s: FrameState, k: number) => {
	const b = s.fences[k]!.batch;
	return b !== null && !s.batches[b]!.seen;
};

export function resetBlockers(s: FrameState, k: number): Line[] {
	return fenceInUse(s, k) ? [vu('VUID-vkResetFences-pFences-01123', 'vkResetFences')] : [];
}

export function resetFence(s: FrameState, k: number): Step {
	return { state: setFence(s, k, { signaled: false, batch: null }), log: [call(`vkResetFences(in_flight[${k}])`)] };
}

// ---------------------------------------------------------------------------------------------------------------
// Acquire

export function acquireBlockers(s: FrameState, sem: string): Line[] {
	const x = s.sems[sem];
	if (!x) return noPick();
	const out: Line[] = [];
	// A submitted wait takes the signal: the layer reports the pending wait (01779), not the payload (01286); seen in
	// step 10 without its fence wait.
	const pending = x.sigs.some((g) => g.claim?.by === 'batch' || (g.claim === null && !g.fired));
	const present = x.sigs.find((g) => g.claim?.by === 'present')?.claim;
	if (pending || present) out.push(vu('VUID-vkAcquireNextImageKHR-semaphore-01779', 'vkAcquireNextImageKHR'));
	else if (x.sigs.length) out.push(vu('VUID-vkAcquireNextImageKHR-semaphore-01286', 'vkAcquireNextImageKHR'));
	if (!pending && present?.by === 'present') out.push(msg('info', 'semInUseByPresent', { sem: semName(sem), i: present.img }));
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
	const now = st.imgs[i]!.owner === 'free';
	st = { ...st, order: st.order.slice(1) };
	st = setImg(st, i, { owner: 'app', acq: sem, acqWaited: false, broken: false, rendered: null });
	// The image is back, so the present that showed it has finished waiting on its semaphore: the host knows now.
	st = mapSigs(st, (sigs) => sigs.filter((g) => !(g.claim?.by === 'present' && g.claim.img === i)));
	const sig: Sig = { from: { by: 'acquire', img: i }, fired: now, claim: null, used: false };
	st = mapSigs(st, (sigs, id) => (id === sem ? [...sigs, sig] : sigs));
	log.push(call(`vkAcquireNextImageKHR(UINT64_MAX, ${semName(sem)}) -> image ${i}`));
	if (!now) log.push(msg('info', 'pendingAcquire', { i, sem: semName(sem) }));
	return { state: st, log };
}

// ---------------------------------------------------------------------------------------------------------------
// Command buffers

/** `image` null = none picked. Any image may be recorded for: the submit checks that the app holds it. */
export function recordBlockers(s: FrameState, k: number, image: number | null): Line[] {
	if (image === null || !s.imgs[image]) return noPick();
	return s.cbs[k]!.state === 'pending' ? [vu('VUID-vkResetCommandBuffer-commandBuffer-00045', 'vkResetCommandBuffer'), vu('VUID-vkBeginCommandBuffer-commandBuffer-00049', 'vkBeginCommandBuffer')] : [];
}

export function record(s: FrameState, k: number, image: number): Step {
	return {
		state: setCb(s, k, { state: 'executable', image, batch: null }),
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
	if (!s.sems[o.wait] || !s.sems[o.signal] || (o.fence && !s.fences[slotOf(o.fence)])) return noPick();
	const out: Line[] = [];
	const cb = s.cbs[k]!;
	if (cb.state === 'pending') out.push(vu('VUID-vkQueueSubmit2-commandBuffer-03875', 'vkQueueSubmit2'));
	else if (cb.state !== 'executable') out.push(vu('VUID-vkQueueSubmit2-commandBuffer-03874', 'vkQueueSubmit2'));
	else if (s.imgs[cb.image!]!.owner !== 'app') out.push(msg('error', 'notHeld', { i: cb.image! }));
	if (o.fence) {
		const k2 = slotOf(o.fence);
		if (s.fences[k2]!.signaled) out.push(vu('VUID-vkQueueSubmit2-fence-04894', 'vkQueueSubmit2'));
		else if (fenceInUse(s, k2)) out.push(vu('VUID-vkQueueSubmit2-fence-04895', 'vkQueueSubmit2'));
	}
	if (!unclaimed(s.sems[o.wait]!)) out.push(vu('VUID-vkQueueSubmit2-semaphore-03873', 'vkQueueSubmit2'));
	// The batch's own wait takes its signal before the batch's signal executes.
	const own = o.signal === o.wait && unclaimed(s.sems[o.wait]!) ? claimSig(s, o.wait, { by: 'batch', id: s.batches.length }) : s;
	const sigs = own.sems[o.signal]!.sigs;
	if (!signalable(sigs)) {
		out.push(vu('VUID-vkQueueSubmit2-semaphore-03868', 'vkQueueSubmit2'));
		const p = sigs.find((g) => g.claim?.by === 'present')?.claim;
		if (p?.by === 'present') out.push(msg('info', 'semInUseByPresent', { sem: semName(o.signal), i: p.img }));
	}
	return out;
}

/** Whether a batch rendering `image` and waiting on `wait` at `stage` is chained after the image's acquire. */
function acquireChained(s: FrameState, image: number, wait: string, stage: Stage): boolean {
	const img = s.imgs[image]!;
	if (img.acqWaited) return true; // an earlier batch of this acquire was chained: this one follows it in submission order
	return wait === img.acq && stage !== 'BOTTOM_OF_PIPE';
}

export function submit(s: FrameState, k: number, o: SubmitOpts): Step {
	const image = s.cbs[k]!.image!;
	const hazard = !acquireChained(s, image, o.wait, o.stage);
	const id = s.batches.length;
	const others = pendingBatches(s).some((b) => b.slot !== k);
	let st: FrameState = { ...s, batches: [...s.batches, { id, slot: k, image, wait: o.wait, stage: o.stage, signal: o.signal, fence: o.fence, hazard, done: false, seen: false }], overlap: s.overlap || others };
	st = setCb(st, k, { state: 'pending', batch: id });
	st = claimSig(st, o.wait, { by: 'batch', id });
	const sig: Sig = { from: { by: 'batch', id }, fired: false, claim: null, used: false };
	st = mapSigs(st, (sigs, sem) => (sem === o.signal ? [...sigs, sig] : sigs));
	if (o.fence) st = setFence(st, slotOf(o.fence), { batch: id });
	const img = st.imgs[image]!;
	st = setImg(st, image, { rendered: id, acqWaited: img.acqWaited || !hazard, broken: img.broken || hazard });
	const log: Line[] = [call(`vkQueueSubmit2(cmd[${k}], wait ${semName(o.wait)} at ${stageName(o.stage)}, signal ${semName(o.signal)}, fence ${o.fence ? `in_flight[${slotOf(o.fence)}]` : 'VK_NULL_HANDLE'}) -> batch #${id}`)];
	if (hazard) log.push({ kind: 'error', text: SYNC.acquire.text, sync: SYNC.acquire.id }, o.wait === img.acq ? msg('info', 'hazardStage', { i: image, stage: stageName(o.stage) }) : msg('info', 'hazardAcquire', { i: image, sem: semName(img.acq!) }));
	else if (o.stage === 'TOP_OF_PIPE' || o.stage === 'ALL_COMMANDS') log.push(msg('info', 'slowStage', { stage: stageName(o.stage) }));
	return { state: st, log };
}

// ---------------------------------------------------------------------------------------------------------------
// Present

export function presentBlockers(s: FrameState, i: number, wait: string): Line[] {
	if (!s.sems[wait]) return noPick();
	const out: Line[] = [];
	const img = s.imgs[i]!;
	if (img.owner !== 'app' || img.rendered === null) out.push(vu('VUID-VkPresentInfoKHR-pImageIndices-01430', 'vkQueuePresentKHR'));
	if (!unclaimed(s.sems[wait]!)) out.push(vu('VUID-vkQueuePresentKHR-pWaitSemaphores-03268', 'vkQueuePresentKHR'));
	return out;
}

export function present(s: FrameState, i: number, wait: string): Step {
	const img = s.imgs[i]!;
	const b = s.batches[img.rendered!]!;
	// A signal of the last batch rendering the image, or of any later batch on the queue, follows the rendering and
	// the transition to PRESENT_SRC_KHR in submission order (its first synchronization scope covers them).
	const from = unclaimed(s.sems[wait]!)!.from;
	const badWait = !(from.by === 'batch' && from.id >= b.id);
	let st: FrameState = { ...s, queue: [...s.queue, i], order: [...s.order, i], frames: [...s.frames, { slot: b.slot, image: i, hazard: img.broken || badWait }] };
	st = setImg(st, i, { owner: 'queued', acq: null });
	st = claimSig(st, wait, { by: 'present', img: i });
	const log: Line[] = [call(`vkQueuePresentKHR(image ${i}, wait ${semName(wait)}) -> VK_SUCCESS`)];
	if (badWait) log.push({ kind: 'error', text: SYNC.present.text.replace('{i}', String(i)), sync: SYNC.present.id }, msg('info', 'hazardPresent', { i, sem: semName(b.signal) }));
	return { state: st, log };
}

// ---------------------------------------------------------------------------------------------------------------
// Time: the GPU queue and the presentation engine

/** The oldest pending batch, and the signal it still waits for (null = it can run). */
function nextBatch(s: FrameState): { b: Batch; blockedBy: Sig | null } | null {
	const b = pendingBatches(s)[0];
	if (!b) return null;
	const w = claimedBy(s, { by: 'batch', id: b.id })!.sig;
	return { b, blockedBy: w.fired ? null : w };
}

export function gpuBlockers(s: FrameState): Line[] {
	const n = nextBatch(s);
	if (!n) return [msg('info', 'gpuIdle')];
	if (n.blockedBy) {
		const from = n.blockedBy.from;
		return [msg('info', 'gpuWaits', { n: n.b.id, sem: semName(n.b.wait), i: from.by === 'acquire' ? from.img : '?' })];
	}
	return [];
}

/**
 * The GPU executes the oldest pending batch: its wait completes, its signal and fence fire. The host learns nothing
 * from it: the command buffer, the fence and the semaphore wait stay in use for the host until a fence wait.
 */
export function gpuStep(s: FrameState): Step {
	const { b } = nextBatch(s)!;
	let st = setBatch(s, b.id, { done: true });
	st = mapSigs(st, (sigs) =>
		sigs.map((g) => {
			if (sameClaim(g.claim, { by: 'batch', id: b.id })) return { ...g, used: true };
			if (g.from.by === 'batch' && g.from.id === b.id) return { ...g, fired: true };
			return g;
		}),
	);
	if (b.fence) st = setFence(st, slotOf(b.fence), { signaled: true });
	return { state: st, log: [msg('info', 'gpuRan', { n: b.id, slot: b.slot, i: b.image, sem: semName(b.signal), fence: b.fence ? `in_flight[${slotOf(b.fence)}]` : '-' })] };
}

/** The image the engine shows at the next vblank, and the semaphore its present still waits for. */
function nextShown(s: FrameState): { i: number; waits: string | null } | null {
	const i = s.queue[0];
	if (i === undefined) return null;
	const w = claimedBy(s, { by: 'present', img: i });
	// The present's wait completes when its semaphore is signaled.
	return { i, waits: w && !w.sig.fired ? w.sem : null };
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
	// The present's wait has executed; the app still cannot see it (the signal stays until image i is acquired again).
	st = mapSigs(st, (sigs) => sigs.map((g) => (sameClaim(g.claim, { by: 'present', img: i }) ? { ...g, used: true } : g)));
	st = setImg(st, i, { owner: 'shown' });
	const log: Line[] = [msg('info', 'peShown', { i })];
	const old = s.screen;
	if (old !== null) {
		if (st.imgs[old]!.owner === 'shown') st = setImg(st, old, { owner: 'free' });
		// An acquire that returned the image while it was on screen signals its semaphore now.
		const rel = SEMS.find((id) => st.sems[id]!.sigs.some((g) => g.from.by === 'acquire' && g.from.img === old && !g.fired));
		if (rel) st = mapSigs(st, (sigs, id) => (id === rel ? sigs.map((g) => (g.from.by === 'acquire' && g.from.img === old ? { ...g, fired: true } : g)) : sigs));
		log.push(msg('info', 'peReleased', { i: old, sem: rel ? semName(rel) : '-' }));
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

/**
 * A semaphore as the host sees it: taken by a present (`present`, image) or by a batch (`waited`, or `waitedDone`
 * once the GPU executed that wait but no fence wait told the host), else the state of its newest signal.
 */
export type SemView = { st: 'unsignaled' | 'pending' | 'signaled' } | { st: 'waited' | 'waitedDone'; n: number } | { st: 'present'; i: number };
export function semView(s: FrameState, id: string): SemView {
	const sigs = s.sems[id]!.sigs;
	const p = sigs.find((g) => g.claim?.by === 'present')?.claim;
	if (p?.by === 'present') return { st: 'present', i: p.img };
	const w = sigs.find((g) => g.claim?.by === 'batch');
	if (w?.claim?.by === 'batch') return { st: w.used ? 'waitedDone' : 'waited', n: w.claim.id };
	const last = sigs.at(-1);
	return { st: !last ? 'unsignaled' : last.fired ? 'signaled' : 'pending' };
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
