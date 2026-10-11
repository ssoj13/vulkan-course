import { describe, expect, test } from 'bun:test';
import gpu from '../../src/data/gpu.json';
import { textErrors } from '../../src/lib/text';
import {
	MAX_T,
	PRESETS,
	PRESET_IDS,
	QUEUES,
	ISSUE_VUID,
	analyze,
	expand,
	frames,
	pairBinary,
	phases,
	queuesFor,
	raceSpans,
	scenarioErrors,
	simulate,
	stageSupported,
	type IssueKind,
	type Scenario,
	type SubmitOp,
} from '../../src/lib/labs/queue-sync';
import { vuid } from '../../src/lib/vuids';
import { TEXT } from '../../src/components/labs/queue-sync.text';

/** Deep copy of a preset, changed by `f`. */
const edit = (id: keyof typeof PRESETS, f: (sc: Scenario) => void): Scenario => {
	const sc = structuredClone(PRESETS[id]) as Scenario;
	f(sc);
	return sc;
};
const sub = (sc: Scenario, i: number) => sc.ops[i] as SubmitOp;
const kinds = (sc: Scenario): IssueKind[] => analyze(sc).issues.map((i) => i.kind);

describe('data', () => {
	test('the queues come from the real families of the course GPU', () => {
		expect(QUEUES.map((q) => [q.id, q.family])).toEqual([
			['graphics', 0],
			['compute', 2],
			['transfer', 1],
		]);
		expect(gpu.queueFamilies[1]!.queueFlags).not.toContain('VK_QUEUE_COMPUTE_BIT');
		expect(gpu.queueFamilies[2]!.queueFlags).not.toContain('VK_QUEUE_GRAPHICS_BIT');
	});
	test('a binary wait with no signal submitted is the rule the lab names', () => {
		expect(vuid(ISSUE_VUID.binaryNoSignal!)).toContain('must reference a semaphore signal operation that has been submitted for execution');
	});
	test('presets are valid scenarios', () => {
		for (const id of PRESET_IDS) expect(scenarioErrors(PRESETS[id])).toEqual([]);
	});
	test('texts are complete in every language', () => {
		expect(textErrors(TEXT, 'queue-sync')).toEqual([]);
	});
});

describe('stages', () => {
	test('a wait mask covers its stage and the logically later ones of the same pipeline', () => {
		expect(expand('VERTEX_SHADER')).toEqual(['VERTEX_SHADER', 'FRAGMENT_SHADER', 'COLOR_ATTACHMENT_OUTPUT']);
		expect(expand('FRAGMENT_SHADER')).toEqual(['FRAGMENT_SHADER', 'COLOR_ATTACHMENT_OUTPUT']);
		expect(expand('COPY')).toEqual(['COPY']);
		expect(expand('ALL_COMMANDS')).toHaveLength(5);
	});
	test('supported stages follow the spec table and the family flags', () => {
		expect(stageSupported('COPY', 'transfer')).toBe(true);
		expect(stageSupported('COMPUTE_SHADER', 'transfer')).toBe(false);
		expect(stageSupported('FRAGMENT_SHADER', 'compute')).toBe(false);
		expect(stageSupported('ALL_COMMANDS', 'transfer')).toBe(true);
		expect(queuesFor('draw')).toEqual(['graphics']);
		expect(queuesFor('dispatch')).toEqual(['graphics', 'compute']);
		expect(queuesFor('copy')).toEqual(['graphics', 'compute', 'transfer']);
	});
	test('a draw spends one tick in vertex and color output and the rest in fragment shading', () => {
		expect(phases('draw', 5).map((p) => p.len)).toEqual([1, 3, 1]);
	});
});

describe('(a) upload, then render waiting on a binary semaphore', () => {
	const a = analyze(PRESETS.upload);
	test('no issues', () => expect(a.issues).toEqual([]));
	test('the vertex stage runs before the upload ends; fragment shading waits for the signal', () => {
		const upload = a.run.batches[0]!;
		const vs = a.run.segs.find((s) => s.op === 1 && s.what === 'VERTEX_SHADER')!;
		const fs = a.run.segs.find((s) => s.op === 1 && s.what === 'FRAGMENT_SHADER')!;
		expect(vs.from).toBeLessThan(upload.done!);
		expect(fs.from).toBe(upload.done!);
		expect(a.run.segs.some((s) => s.op === 1 && s.what === 'stall')).toBe(true);
		expect(a.run.links).toEqual([{ kind: 'sem', name: 'uploaded', from: { op: 0, t: upload.done! }, to: { op: 1, t: upload.done! } }]);
	});
	test('waiting at COLOR_ATTACHMENT_OUTPUT does not cover the fragment shader read: a race', () => {
		const sc = edit('upload', (sc) => (sub(sc, 1).waits[0]!.stage = 'COLOR_ATTACHMENT_OUTPUT'));
		expect(kinds(sc)).toEqual(['race']);
	});
	test('waiting at VERTEX_SHADER also covers the later fragment stage: no race, the whole draw waits', () => {
		const sc = edit('upload', (sc) => (sub(sc, 1).waits[0]!.stage = 'VERTEX_SHADER'));
		const r = analyze(sc);
		expect(r.issues).toEqual([]);
		expect(r.run.batches[1]!.started).toBe(r.run.batches[0]!.done);
	});
	test('a binary wait submitted before its signal breaks VUID 03873 and never runs', () => {
		const sc = edit('upload', (sc) => sc.ops.reverse());
		const r = analyze(sc);
		expect(r.issues.map((i) => [i.kind, i.op, i.vuid])).toEqual([
			['binaryNoSignal', 0, 'VUID-vkQueueSubmit2-semaphore-03873'],
			['race', 1, undefined],
		]);
		expect(r.run.batches[0]!.done).toBeNull();
		expect(r.run.end).toBeLessThan(MAX_T);
	});
	test('the same order is fine with a timeline semaphore: wait-before-signal is allowed', () => {
		const sc = edit('upload', (sc) => {
			sc.ops.reverse();
			sc.sems[0]!.type = 'timeline';
		});
		const r = analyze(sc);
		expect(r.issues).toEqual([]);
		expect(r.run.batches[0]!.done).not.toBeNull();
	});
	test('two waits on one binary signal: the second has nothing to pair with', () => {
		const sc = edit('upload', (sc) => sc.ops.push(structuredClone(sc.ops[1]!)));
		const p = pairBinary(sc);
		expect([p.get('1:0'), p.get('2:0')]).toEqual(['0:0', null]);
		expect(kinds(sc)).toContain('binaryNoSignal');
	});
	test('signaling a binary semaphore twice without a wait in between breaks VUID 03868', () => {
		const sc = edit('upload', (sc) => (sc.ops[1] = { ...structuredClone(sub(sc, 0)), queue: 'graphics' }));
		const r = analyze(sc).issues.filter((i) => i.kind === 'binarySignaled');
		expect(r.map((i) => [i.op, i.vuid])).toEqual([[1, 'VUID-vkQueueSubmit2-semaphore-03868']]);
	});
	test('a second upload whose signal is consumed in between is fine: the wait takes the first signal', () => {
		const sc = edit('upload', (sc) => sc.ops.splice(1, 0, structuredClone(sc.ops[0]!)));
		expect(pairBinary(sc).get('2:0')).toBe('0:0');
		expect(kinds(sc)).not.toContain('binarySignaled');
	});
});

describe('(b) the same without the semaphore', () => {
	const a = analyze(PRESETS.race);
	test('reading the texture is a data race with the copy that writes it', () => {
		expect(a.issues.map((i) => [i.kind, i.vars.res, i.vars.aStage, i.vars.bStage])).toEqual([['race', 'texture', 'COPY', 'FRAGMENT_SHADER']]);
		expect(a.issues[0]!.vuid).toBeUndefined();
	});
	test('in the drawn schedule the read overlaps the write', () => {
		const { a: w, b: r } = raceSpans(a.run, a.issues[0]!);
		expect(r![0]).toBeLessThan(w![1]);
	});
});

describe('(c) async compute joined by a timeline semaphore', () => {
	const a = analyze(PRESETS.async);
	test('no issues; compute overlaps the G-buffer pass', () => {
		expect(a.issues).toEqual([]);
		const sim = a.run.batches[0]!;
		const gb = a.run.batches[1]!;
		expect(gb.started!).toBeLessThan(sim.done!);
	});
	test('the signal of value 2 waits for the earlier batch on the graphics queue too (submission order scope)', () => {
		const sig = a.run.events.find((e) => e.kind === 'signal' && e.op === 2)!;
		expect(sig.t).toBe(Math.max(a.run.batches[1]!.done!, a.run.batches[2]!.done!));
	});
	test('without its wait, the second simulation races with the draw that reads the particles and with the first simulation', () => {
		const sc = edit('async', (sc) => (sub(sc, 3).waits = []));
		const r = analyze(sc).issues;
		expect(r.filter((i) => i.kind === 'race').map((i) => [i.others[0], i.op, i.vars.a, i.vars.b])).toEqual([
			[0, 3, 'write', 'write'],
			[2, 3, 'read', 'write'],
		]);
		// It now finishes before the draw, so the draw's value 2 arrives after 3: the counter would go down.
		expect(r.find((i) => i.kind === 'timelineNotIncreasing')).toMatchObject({ op: 2, vars: { value: 2, current: 3 } });
	});
	test('a fragment-stage wait on the compute queue breaks VUID 03870 and does not cover the dispatch', () => {
		const sc = edit('async', (sc) => (sub(sc, 3).waits[0]!.stage = 'FRAGMENT_SHADER'));
		expect(kinds(sc).sort()).toEqual(['race', 'race', 'stageUnsupported', 'timelineNotIncreasing']);
	});
	test('waiting for a value nobody signals deadlocks, without a VUID', () => {
		const sc = edit('async', (sc) => {
			sub(sc, 3).waits[0]!.value = 4;
			sub(sc, 3).signals[0]!.value = 5;
		});
		const r = analyze(sc);
		expect(r.issues.map((i) => [i.kind, i.op, i.vuid])).toEqual([['neverRuns', 3, undefined]]);
		expect(r.run.end).toBeLessThan(MAX_T);
	});
	test('a timeline signal must increase the value (VUID-VkSubmitInfo2-semaphore-03882)', () => {
		const sc = edit('async', (sc) => (sub(sc, 3).signals[0]!.value = 2));
		expect(kinds(sc)).toContain('timelineNotIncreasing');
	});
	test('waiting and signaling the same timeline in one batch needs signal > wait (03881)', () => {
		const sc = edit('async', (sc) => (sub(sc, 2).signals[0]!.value = 1));
		expect(kinds(sc)).toContain('timelineSameBatch');
	});
});

describe('(d) frames in flight with fences', () => {
	const a = analyze(PRESETS.frames);
	test('no issues; the CPU blocks on F0 until frame 0 is done', () => {
		expect(a.issues).toEqual([]);
		const wait = a.run.segs.find((s) => s.lane === 'cpu' && s.what === 'wait')!;
		expect(wait.op).toBe(4);
		expect(wait.to).toBe(a.run.batches[1]!.done!);
		expect(a.run.links.filter((l) => l.kind === 'fence').map((l) => l.name)).toEqual(['F0', 'F1']);
	});
	test('two frames are in flight at once', () => {
		expect(a.run.batches[3]!.started!).toBeLessThan(a.run.batches[1]!.done!);
	});
	test('without the wait, frame 2 re-records a command buffer that may be pending and races on the uniforms', () => {
		const sc = edit('frames', (sc) => sc.ops.splice(4, 1));
		const r = analyze(sc).issues;
		expect(r.map((i) => i.kind).sort()).toEqual(['cbPending', 'fencePending', 'race']);
		expect(r.find((i) => i.kind === 'cbPending')!.vuid).toBe('VUID-vkBeginCommandBuffer-commandBuffer-00049');
		expect(r.find((i) => i.kind === 'race')!.vars).toMatchObject({ res: 'uniforms0', aStage: 'VERTEX_SHADER', bStage: 'HOST' });
	});
	test('fences created unsignaled: the first wait blocks forever and nothing is submitted', () => {
		const sc = edit('frames', (sc) => sc.fences.forEach((f) => (f.signaled = false)));
		const r = analyze(sc);
		expect(r.issues.map((i) => [i.kind, i.op])).toEqual([['fenceNeverSubmitted', 0]]);
		expect(r.run.batches.every((b) => !b || b.submitted === null)).toBe(true);
	});
	test('submitting with a fence that is still signaled breaks VUID 04894', () => {
		const sc = edit('frames', (sc) => sc.ops.splice(0, 1));
		expect(analyze(sc).issues.map((i) => i.vuid)).toEqual(['VUID-vkQueueSubmit2-fence-04894']);
	});
});

describe('timeline frames', () => {
	test('one frame per time step, from 0 to the end', () => {
		const a = analyze(PRESETS.upload);
		const fr = frames(a.run);
		expect(fr).toHaveLength(a.run.end + 1);
		expect(fr.at(-1)!.events.map((e) => e.kind)).toContain('done');
	});
	test('a scenario with no operations still has its initial frame', () => {
		expect(frames(simulate({ sems: [], fences: [], ops: [] }))).toHaveLength(1);
	});
});
