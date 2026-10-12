import { describe, expect, test } from 'bun:test';
import { GLOBAL, StaleAct, act, begin, complete, consistency, counts, decode, encode, goalsOf, replay, undo, why, type Act, type Run } from 'course-kit/lib/trainer.ts';
import { textErrors } from '../../src/lib/text';
import { TEXT } from '../../src/components/labs/frame-assembly.text';
import { ITEMS, frameScenario } from '../../src/lib/frame-trainer';
import { ruleVersion } from '../../src/lib/object-trainer';
import { SYNC, type FrameState } from '../../src/lib/frames';

const V = 'test';
const sc = frameScenario('en', V);

// Act builders: one per host call / time step, options as the trainer stores them.
const create = (k: number, signaled = true): Act => ({ item: `f${k}`, verb: 'create', opts: { signaled } });
const wait = (k: number, timeout: 'inf' | 'zero' = 'inf'): Act => ({ item: `f${k}`, verb: 'wait', opts: { timeout } });
const reset = (k: number): Act => ({ item: `f${k}`, verb: 'reset' });
const acquire = (sem: string): Act => ({ item: GLOBAL, verb: 'acquire', opts: { sem } });
const rec = (k: number, i: number): Act => ({ item: `cb${k}`, verb: 'record', opts: { image: `img${i}` } });
const submit = (k: number, wait: string, signal: string, fence = `f${k}`, stage = 'COLOR_ATTACHMENT_OUTPUT'): Act => ({ item: `cb${k}`, verb: 'submit', opts: { wait, stage, signal, fence } });
const present = (i: number, wait: string): Act => ({ item: `img${i}`, verb: 'present', opts: { wait } });
const gpu: Act = { item: GLOBAL, verb: 'gpu' };
const pe: Act = { item: GLOBAL, verb: 'pe' };

/** One frame of step 10 on slot k, rendering the image `i` the acquire is expected to return. */
const frame = (k: number, i: number): Act[] => [wait(k), acquire(`ia${k}`), reset(k), rec(k, i), submit(k, `ia${k}`, `rf${i}`), present(i, `rf${i}`)];
const SETUP = [create(0), create(1)];
const CORRECT = [...SETUP, ...frame(0, 0), ...frame(1, 1), ...frame(0, 2), ...frame(1, 0)];

/** Plays the acts, asserting status/rule consistency in every state reached. */
function play(acts: readonly Act[], s = sc): Run<FrameState> {
	let r = begin(s);
	for (const a of acts) {
		expect(consistency(s, r.state)).toEqual([]);
		r = act(s, r, a);
	}
	expect(consistency(s, r.state)).toEqual([]);
	return r;
}
const refs = (r: Run<FrameState>) => (r.last?.log ?? []).map((e) => e.ref).filter(Boolean);
const refused = (r: Run<FrameState>) => r.last?.ok === false;

describe('frame assembly: the correct loop of step 10', () => {
	test('4 frames, 2 per slot, no hazard, two frames in flight: complete', () => {
		const r = play(CORRECT);
		expect(r.log.filter((l) => l.kind === 'error')).toEqual([]);
		expect(r.state.frames).toEqual([
			{ slot: 0, image: 0, hazard: false },
			{ slot: 1, image: 1, hazard: false },
			{ slot: 0, image: 2, hazard: false },
			{ slot: 1, image: 0, hazard: false },
		]);
		expect(goalsOf(sc, r.state).map((g) => g.met)).toEqual([true, true, true]);
		expect(complete(sc, r.state)).toBe(true);
	});
	test('frame 3 waits for frame 1 only: the fence wait runs batch #0, batch #1 stays pending', () => {
		const r = play([...SETUP, ...frame(0, 0), ...frame(1, 1), wait(0)]);
		expect(r.state.batches.map((b) => b.done)).toEqual([true, false]);
		expect(r.last?.log.map((e) => e.text)).toContain('-> VK_SUCCESS');
	});
	test('the 4th acquire gets image 0 back only after the presentation engine showed it (FIFO); its semaphore signal waits for the next vblank', () => {
		const r = play([...SETUP, ...frame(0, 0), ...frame(1, 1), ...frame(0, 2), wait(1), acquire('ia1')]);
		expect(r.state.screen).toBe(0);
		expect(r.state.sems.ia1!.sig).toBe('pending');
		const blocked = act(sc, play([...r.acts, reset(1), rec(1, 0), submit(1, 'ia1', 'rf0')]), gpu);
		expect(blocked.last?.log[0]!.text).toStartWith('Batch #3 cannot complete yet: it waits for image_available[1]');
		// The next vblank shows image 1 and releases image 0: its acquire signal fires and the batch can complete.
		const vblank = act(sc, blocked, pe);
		expect(vblank.state.sems.ia1!.sig).toBe('signaled');
		expect(act(sc, vblank, gpu).state.batches.every((x) => x.done)).toBe(true);
	});
	test('undo, reload (encode/decode) and replay rebuild the same run', () => {
		const r = play(CORRECT);
		expect(undo(sc, r).state.frames).toHaveLength(3);
		expect(replay(sc, decode(JSON.parse(JSON.stringify(encode(sc, r))), V)!)).toEqual(r);
	});
});

describe('frame assembly: refusals quote the real rules', () => {
	const f1 = [...SETUP, ...frame(0, 0)];
	test('re-recording a pending command buffer: VUID-00045 and VUID-00049 (status locked with that reason)', () => {
		const r = play([...f1, acquire('ia1'), rec(0, 1)]);
		expect(refused(r)).toBe(true);
		expect(refs(r)).toEqual(['VUID-vkResetCommandBuffer-commandBuffer-00045', 'VUID-vkBeginCommandBuffer-commandBuffer-00049']);
		expect(sc.status(r.state, 'cb0')).toBe('locked');
		expect(why(sc, r.state, 'cb0').map((f) => f.ref)).toContain('VUID-vkResetCommandBuffer-commandBuffer-00045');
	});
	test('submit with a signaled fence (reset forgotten): VUID-vkQueueSubmit2-fence-04894', () => {
		const r = play([...SETUP, wait(0), acquire('ia0'), rec(0, 0), submit(0, 'ia0', 'rf0')]);
		expect(refs(r)).toEqual(['VUID-vkQueueSubmit2-fence-04894']);
	});
	test('submit with the fence of a pending submission: VUID-vkQueueSubmit2-fence-04895', () => {
		const r = play([...f1, wait(1), acquire('ia1'), reset(1), rec(1, 1), submit(1, 'ia1', 'rf1', 'f0')]);
		expect(refs(r)).toEqual(['VUID-vkQueueSubmit2-fence-04895']);
	});
	test('waiting on a semaphore nothing signaled: VUID-vkQueueSubmit2-semaphore-03873', () => {
		const r = play([...SETUP, wait(0), acquire('ia0'), reset(0), rec(0, 0), submit(0, 'ia1', 'rf0')]);
		expect(refs(r)).toEqual(['VUID-vkQueueSubmit2-semaphore-03873']);
	});
	test('the challenge: render_finished per slot instead of per image fails at frame 3 (VUID-03868 + why)', () => {
		const r = play([...SETUP, ...frame(0, 0), ...frame(1, 1), wait(0), acquire('ia0'), reset(0), rec(0, 2), submit(0, 'ia0', 'rf0')]);
		expect(refs(r)).toEqual(['VUID-vkQueueSubmit2-semaphore-03868']);
		expect(r.last?.log[0]!.text).toStartWith('Validation Error: [ VUID-vkQueueSubmit2-semaphore-03868 ] vkQueueSubmit2(): The Vulkan spec states: The semaphore member of any binary semaphore');
		expect(r.last?.log[1]!.text).toStartWith('render_finished[0] is still waited on by the present of image 0');
	});
	test('a signaled semaphore again into acquire: VUID-01286; one a pending submit waits on: VUID-01779', () => {
		expect(refs(play([...SETUP, acquire('ia0'), acquire('ia0')]))).toEqual(['VUID-vkAcquireNextImageKHR-semaphore-01286']);
		// Frame 3 without its fence wait: batch #0 still waits on image_available[0].
		expect(refs(play([...f1, ...frame(1, 1), acquire('ia0')]))).toEqual(['VUID-vkAcquireNextImageKHR-semaphore-01779']);
	});
	test('holding 2 of 3 images (minImageCount 2) and acquiring with UINT64_MAX: VUID-07783', () => {
		const r = play([...SETUP, acquire('ia0'), acquire('ia1'), acquire('rf0')]);
		expect(refs(r)).toEqual(['VUID-vkAcquireNextImageKHR-surface-07783']);
	});
	test('present of an image not acquired, not rendered, or without a signal: VUID-01430 / VUID-03268', () => {
		expect(refs(play([present(0, 'rf0')]))).toEqual(['VUID-VkPresentInfoKHR-pImageIndices-01430', 'VUID-vkQueuePresentKHR-pWaitSemaphores-03268']);
		const r = play([...SETUP, wait(0), acquire('ia0'), reset(0), rec(0, 0), submit(0, 'ia0', 'rf0'), present(0, 'rf1')]);
		expect(refs(r)).toEqual(['VUID-vkQueuePresentKHR-pWaitSemaphores-03268']);
	});
	test('resetting the fence of a pending submission: VUID-vkResetFences-pFences-01123', () => {
		expect(refs(play([...f1, reset(0)]))).toEqual(['VUID-vkResetFences-pFences-01123']);
	});
	test('deadlocks are refused with a labeled explanation; timeout 0 returns VK_TIMEOUT', () => {
		// Fence created unsignaled: the first wait would never return.
		const r = play([create(0, false), create(1), wait(0)]);
		expect(refused(r)).toBe(true);
		expect(r.last?.log[0]!.text).toStartWith('Deadlock (no VUID): in_flight[0] is unsignaled');
		expect(sc.status(r.state, 'f0')).toBe('locked');
		// Reset before the acquire, then wait again: nothing pending will signal it.
		expect(play([...SETUP, wait(0), reset(0), wait(0)]).last?.log[0]!.text).toStartWith('Deadlock');
		const z = play([create(0, false), wait(0, 'zero')]);
		expect(z.last?.ok).toBe(true);
		expect(z.last?.log.map((e) => e.text)).toContain('-> VK_TIMEOUT');
	});
	test('recording with no acquired image is refused (spec: use only between acquire and present)', () => {
		const r = play([...SETUP, { item: 'cb0', verb: 'record' }]);
		expect(r.last?.log[0]!.text).toStartWith('No acquired image to render to.');
	});
});

describe('frame assembly: hazards are accepted and break the frame', () => {
	const upTo = [...SETUP, wait(0), acquire('ia0'), reset(0), rec(0, 0)];
	test('waiting at BOTTOM_OF_PIPE: WRITE_AFTER_READ (as sync validation reported in step 10)', () => {
		const r = play([...upTo, submit(0, 'ia0', 'rf0', 'f0', 'BOTTOM_OF_PIPE'), present(0, 'rf0')]);
		expect(r.log.find((l) => l.ref === SYNC.acquire.id)?.text).toStartWith('vkQueueSubmit2(): WRITE_AFTER_READ hazard detected.');
		expect(r.state.frames[0]!.hazard).toBe(true);
		expect(counts(sc, r).errors).toBe(1);
	});
	test('waiting on another semaphore than the image\'s acquire semaphore is the same hazard', () => {
		const r = play([...SETUP, ...frame(0, 0), wait(1), acquire('ia1'), reset(1), rec(1, 1), submit(1, 'rf0', 'rf1')]);
		expect(r.last?.ok).toBe(false); // rf0 is claimed by the present of image 0: no unclaimed signal
		const r2 = play([...upTo, { item: 'cb0', verb: 'submit', opts: { wait: 'ia0', stage: 'TOP_OF_PIPE', signal: 'rf0', fence: 'f0' } }]);
		expect(r2.last?.log.map((e) => e.ref).filter(Boolean)).toEqual([]);
		expect(r2.last?.log[1]!.text).toStartWith('Valid, but VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT holds the whole batch back');
	});
	test('a present that does not wait on the rendering batch\'s semaphore: PRESENT_AFTER_WRITE', () => {
		const r = play([...upTo, submit(0, 'ia0', 'rf0'), acquire('ia1'), present(0, 'ia1')]);
		expect(r.last?.ok).toBe(true);
		expect(r.last?.log[1]).toMatchObject({ kind: 'error', ref: SYNC.present.id });
		expect(r.last?.log[1]!.text).toContain('presents swapchain image 0');
		expect(r.state.frames[0]!.hazard).toBe(true);
	});
	test('a hazard keeps the run from completing even with 4 frames', () => {
		const acts = [...SETUP, wait(0), acquire('ia0'), reset(0), rec(0, 0), submit(0, 'ia0', 'rf0', 'f0', 'BOTTOM_OF_PIPE'), present(0, 'rf0'), ...frame(1, 1), ...frame(0, 2), ...frame(1, 0)];
		const r = play(acts);
		expect(goalsOf(sc, r.state).map((g) => g.met)).toEqual([true, false, true]);
		expect(complete(sc, r.state)).toBe(false);
	});
});

describe('frame assembly: stored runs and versions', () => {
	test('a run stored for other rules is not replayed', () => {
		const stored = JSON.parse(JSON.stringify(encode(sc, play(CORRECT.slice(0, 5)))));
		expect(decode(stored, V)).toHaveLength(5);
		expect(decode(stored, ruleVersion(['other']))).toBeNull();
	});
	test('replay stops at an act the scenario does not offer and keeps the ones before it', () => {
		const acts = [...SETUP, { item: 'cb0', verb: 'begin' }, wait(0)];
		expect(replay(sc, acts).acts).toEqual(SETUP);
		expect(() => act(sc, begin(sc), { item: 'f0', verb: 'wait' })).toThrow(StaleAct); // not created yet: no wait verb
		expect(() => act(sc, begin(sc), { item: GLOBAL, verb: 'acquire', opts: { sem: 'rf9' } })).toThrow(StaleAct);
	});
	test('items are the fixed objects of step 10', () => {
		expect(ITEMS).toEqual(['f0', 'cb0', 'ia0', 'f1', 'cb1', 'ia1', 'swapchain', 'img0', 'rf0', 'img1', 'rf1', 'img2', 'rf2', 'queue']);
	});
});

describe('frame assembly texts', () => {
	test('every language has the same keys, placeholders and plural forms', () => {
		expect(textErrors(TEXT, 'frame-assembly')).toEqual([]);
	});
	test('Russian remarks and badges', () => {
		const ru = frameScenario('ru', V);
		const r = [create(0, false), wait(0)].reduce((x, a) => act(ru, x, a), begin(ru));
		expect(r.last?.log[0]!.text).toStartWith('Взаимоблокировка (не VUID)');
		expect(ru.badge!(r.state, 'f0')).toBe('unsignaled');
	});
});

describe('frame assembly: random walks', () => {
	test('from any point of the correct run, any offered act with any offered option value keeps statuses consistent and never throws', () => {
		let seed = 12345;
		const rnd = (n: number) => ((seed = (seed * 1103515245 + 12345) % 2147483648), seed % n);
		// Each walk starts from a prefix of the correct run, so the states of later frames are explored too.
		for (let walk = 0; walk < 80; walk++) {
			let r = replay(sc, CORRECT.slice(0, rnd(CORRECT.length + 1)));
			for (let i = 0; i < 40; i++) {
				const items = [GLOBAL, ...ITEMS];
				const item = items[rnd(items.length)]!;
				const verbs = item === GLOBAL ? sc.actions!(r.state) : sc.verbs(r.state, item);
				if (!verbs.length) continue;
				const v = verbs[rnd(verbs.length)]!;
				const opts = Object.fromEntries(
					(v.options ?? []).map((o) => {
						if (o.kind === 'flag') return [o.id, rnd(2) === 0];
						if (o.kind === 'choice') return [o.id, o.choices[rnd(o.choices.length)]!.id];
						if (o.kind === 'number') return [o.id, o.min];
						const c = o.of(r.state);
						return [o.id, c.length ? c[rnd(c.length)]! : ''];
					}),
				);
				r = act(sc, r, { item, verb: v.id, opts });
				expect(consistency(sc, r.state)).toEqual([]);
				for (const id of ITEMS) sc.badge!(r.state, id);
			}
		}
	});
});
