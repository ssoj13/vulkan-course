/**
 * Model of the "command path" lab: a few frames of a render loop on two lanes, CPU and GPU, with the queue between
 * them. The CPU records a command buffer, submits it with vkQueueSubmit2 (which returns at once), and waits on the
 * frame's fence either right after the submit (1 frame in flight) or only before it reuses that command buffer two
 * frames later (2 frames in flight). Durations are illustrative ticks; what the lab shows is qualitative and follows
 * the spec: a submit only enqueues, the GPU starts the work later, and only the fence tells the CPU it is done.
 * Pure module: the view (components/labs/CommandPathView.tsx) draws the schedule and steps through its events.
 */

/** Frames in flight: how many command buffers (and fences) the loop cycles through. */
export type InFlight = 1 | 2;

/** Illustrative durations, in ticks. Not measured: real times depend on the scene, the GPU and the driver. */
export const TICKS = {
	/** CPU work of a frame that does not touch Vulkan (input, animation, culling). */
	update: 2,
	/** vkBeginCommandBuffer, the vkCmd* calls, vkEndCommandBuffer. */
	record: 3,
	/** The vkQueueSubmit2 call itself: the driver hands the batch to the kernel-mode driver and returns. */
	submit: 1,
	/** From "in the queue" to "the command processor starts on it" when the GPU is idle. */
	pickup: 1,
	/** GPU execution of one frame's command buffer. */
	gpu: 6,
} as const;

/** Frames the lab runs. */
export const FRAMES = 3;

export type Lane = 'cpu' | 'queue' | 'gpu';
export type SegKind = 'update' | 'record' | 'submit' | 'wait' | 'queued' | 'exec';

/** One bar on a lane: [t0, t1) in ticks, belonging to frame `f`. */
export interface Segment {
	lane: Lane;
	kind: SegKind;
	f: number;
	t0: number;
	t1: number;
}

export type EventKind =
	| 'start'
	| 'update'
	| 'record'
	| 'recorded'
	| 'queued'
	| 'gpuStart'
	| 'gpuEnd'
	| 'waitStart'
	| 'waitFree'
	| 'waitEnd'
	| 'done';

/** A moment the timeline stops at; `f` = frame, `cb` = command buffer and fence slot, `g` = frame waited for. */
export interface CpEvent {
	t: number;
	kind: EventKind;
	f: number;
	cb: number;
	g: number;
}

export interface Schedule {
	inFlight: InFlight;
	segments: Segment[];
	/** In time order; the first is `start` (t = 0), the last is `done`. */
	events: CpEvent[];
	/** When the last lane goes quiet. */
	end: number;
}

/** Busy time per kind of bar within [0, t) (all of the run when `t` is left out). */
export function busy(s: Schedule, kinds: readonly SegKind[], t = s.end): number {
	return s.segments.filter((x) => kinds.includes(x.kind)).reduce((sum, x) => sum + Math.max(0, Math.min(x.t1, t) - x.t0), 0);
}

/** Share of [0, end) a lane spends on `kinds`, in whole percent. */
export function percent(s: Schedule, kinds: readonly SegKind[]): number {
	return Math.round((100 * busy(s, kinds)) / s.end);
}

/**
 * The render loop, simulated. Each frame f uses command buffer and fence slot `f % inFlight`.
 *  - inFlight 1: update, record, submit, then vkWaitForFences on this frame's fence at once: the CPU idles while
 *    the GPU runs the frame.
 *  - inFlight 2: before recording into slot k the CPU waits on fence k, which the frame two back signaled (fences are
 *    created signaled, so the first waits return at once); otherwise it never stops.
 * The GPU lane runs one batch at a time, in submission order: a simplification of this model. The spec gives no such
 * promise; batches without synchronization may overlap or finish out of order.
 */
export function simulate(inFlight: InFlight): Schedule {
	const segments: Segment[] = [];
	const events: CpEvent[] = [{ t: 0, kind: 'start', f: 0, cb: 0, g: 0 }];
	const seg = (lane: Lane, kind: SegKind, f: number, t0: number, t1: number) => {
		if (t1 > t0) segments.push({ lane, kind, f, t0, t1 });
	};
	const ev = (t: number, kind: EventKind, f: number, g = f) => events.push({ t, kind, f, cb: f % inFlight, g });
	/** When each frame's fence is signaled (the GPU finished it). */
	const fenceAt: number[] = [];
	let cpu = 0;
	let gpuFree = 0;

	for (let f = 0; f < FRAMES; f++) {
		if (inFlight === 2) {
			// The fence of this slot was last used by frame f - 2; before that it is still in its created-signaled state.
			const g = f - inFlight;
			const ready = g >= 0 ? fenceAt[g]! : 0;
			if (ready > cpu) {
				ev(cpu, 'waitStart', f, g);
				seg('cpu', 'wait', f, cpu, ready);
				cpu = ready;
				ev(cpu, 'waitEnd', f, g);
			} else ev(cpu, 'waitFree', f, g);
		}
		ev(cpu, 'update', f);
		seg('cpu', 'update', f, cpu, cpu + TICKS.update);
		cpu += TICKS.update;
		ev(cpu, 'record', f);
		seg('cpu', 'record', f, cpu, cpu + TICKS.record);
		cpu += TICKS.record;
		ev(cpu, 'recorded', f);
		seg('cpu', 'submit', f, cpu, cpu + TICKS.submit);
		cpu += TICKS.submit;
		ev(cpu, 'queued', f);

		const start = Math.max(cpu + TICKS.pickup, gpuFree);
		seg('queue', 'queued', f, cpu, start);
		seg('gpu', 'exec', f, start, start + TICKS.gpu);
		gpuFree = start + TICKS.gpu;
		fenceAt[f] = gpuFree;
		ev(start, 'gpuStart', f);
		ev(gpuFree, 'gpuEnd', f);

		if (inFlight === 1) {
			ev(cpu, 'waitStart', f);
			seg('cpu', 'wait', f, cpu, gpuFree);
			cpu = gpuFree;
			ev(cpu, 'waitEnd', f);
		}
	}
	const end = Math.max(cpu, gpuFree);
	events.push({ t: end, kind: 'done', f: FRAMES - 1, cb: (FRAMES - 1) % inFlight, g: FRAMES - 1 });
	// Stable sort: at equal times the order of the loop is kept, which is cause before effect (a fence's gpuEnd is
	// always pushed before the waitEnd it releases).
	events.sort((a, b) => a.t - b.t);
	return { inFlight, segments, events, end };
}
