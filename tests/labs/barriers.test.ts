import { describe, expect, test } from 'bun:test';
import { GLOBAL, StaleAct, act, begin, complete, consistency, decode, encode, goalsOf, replay, undo, why, type Act, type Opts, type Run } from 'course-kit/lib/trainer.ts';
import { textErrors } from '../../src/lib/text';
import { TEXT } from '../../src/components/labs/barriers.text';
import {
	ACCESSES,
	MASKS,
	PUZZLES,
	STAGES,
	barrier,
	blockers,
	correct,
	execScope,
	extras,
	hazards,
	puzzleOf,
	tally,
	type Barrier,
	type BarrierState,
	type Layout,
} from '../../src/lib/barriers';
import { ITEMS, barrierScenario, entryOf } from '../../src/lib/barrier-trainer';
import { ruleVersion } from '../../src/lib/object-trainer';
import { vuid } from '../../src/lib/vuids';

const V = 'test';
const sc = barrierScenario('en', V);
const vuids = (p: string, slots: Barrier[]) => blockers(puzzleOf(p), slots).map((l) => l.vuid);
const found = (p: string, slots: Barrier[]) => hazards(puzzleOf(p), slots).found.map((f) => `${f.hazard}:${f.miss}`);

/** Acts that set slot k of puzzle p to `b` exactly (every offered flag given), and the layouts for images. */
function setActs(state: BarrierState, p: string, k: number, b: Barrier): Act[] {
	const item = `${p}.b${k}`;
	const acts: Act[] = MASKS.map((mask) => {
		const v = sc.verbs(state, item).find((x) => x.id === mask)!;
		const want = b[mask] as readonly string[];
		return { item, verb: mask, opts: Object.fromEntries(v.options!.map((o) => [o.id, want.includes(o.id.slice(2))])) as Opts };
	});
	if (puzzleOf(p).struct === 'VkImageMemoryBarrier2') acts.push({ item, verb: 'layout', opts: { old: b.oldLayout, new: b.newLayout } });
	return acts;
}
/** Plays the acts, asserting status/rule consistency in every state reached. */
function play(acts: readonly Act[], run: Run<BarrierState> = begin(sc)): Run<BarrierState> {
	let r = run;
	for (const a of acts) {
		expect(consistency(sc, r.state)).toEqual([]);
		r = act(sc, r, a);
	}
	expect(consistency(sc, r.state)).toEqual([]);
	return r;
}
/** Sets every slot of puzzle p (built against the run's current state) and submits. */
function solve(r: Run<BarrierState>, p: string, slots: readonly Barrier[]): Run<BarrierState> {
	for (const [k, b] of slots.entries()) r = play(setActs(r.state, p, k, b), r);
	return play([{ item: p, verb: 'submit' }], r);
}

describe('barriers: texts and data', () => {
	test('EN and RU tables agree', () => expect(textErrors(TEXT, 'barriers')).toEqual([]));
	test('every remark the model can produce has a text, in both languages', () => {
		for (const lang of ['en', 'ru']) {
			const e = entryOf({ kind: 'error', msg: 'hazard_acc2', vars: { hazard: 'READ_AFTER_WRITE', before: 'transition', after: 'vkCmdDraw', stage: 'S', access: 'A', slot: 2, slots: 2 } }, lang);
			expect(e.text).not.toContain('{');
		}
		expect(entryOf({ kind: 'error', msg: 'hazard_acc2', vars: { hazard: 'H', before: 'transition', after: 'X', stage: 'S', access: 'A', slot: 2, slots: 2 } }, 'en').text).toStartWith('Barrier 2: H: X accesses');
	});
	test('every VUID the model quotes was synced (texts come only from vuids.json)', () => {
		for (const id of ['VUID-VkImageMemoryBarrier2-oldLayout-01197', 'VUID-VkImageMemoryBarrier2-newLayout-01198', 'VUID-vkCmdDraw-None-09600', 'VUID-VkCopyBufferToImageInfo2-dstImageLayout-00180', 'VUID-vkCmdBeginRendering-pRenderingInfo-09588'])
			expect(vuid(id).length).toBeGreaterThan(20);
		// The access-mask rule names the stages that may carry the flag (registry text, e.g. 03915 for TRANSFER_WRITE).
		expect(vuid('VUID-VkMemoryBarrier2-srcAccessMask-03915')).toContain('VK_PIPELINE_STAGE_2_COPY_BIT');
		expect(vuid('VUID-VkMemoryBarrier2-srcAccessMask-03915')).toContain('VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT');
	});
	test('puzzle offers are subsets of the full lists, ids unique', () => {
		expect(new Set(PUZZLES.map((p) => p.id)).size).toBe(PUZZLES.length);
		for (const p of PUZZLES) {
			expect(p.stages.every((s) => STAGES.includes(s))).toBe(true);
			expect(p.accesses.every((a) => ACCESSES.includes(a))).toBe(true);
		}
	});
});

describe('barriers: scopes (spec rules)', () => {
	test('the source scope grows to logically earlier stages, the destination scope to later ones', () => {
		expect(execScope(['COLOR_ATTACHMENT_OUTPUT'], 'src').has('VERTEX_SHADER')).toBe(true);
		expect(execScope(['FRAGMENT_SHADER'], 'dst').has('COLOR_ATTACHMENT_OUTPUT')).toBe(true);
		expect(execScope(['FRAGMENT_SHADER'], 'dst').has('VERTEX_SHADER')).toBe(false);
	});
	test('transfer stages are not ordered against graphics stages, nor among themselves', () => {
		expect(execScope(['COLOR_ATTACHMENT_OUTPUT'], 'src').has('BLIT')).toBe(false);
		expect(execScope(['BLIT'], 'src').has('COPY')).toBe(false);
	});
	test('TOP_OF_PIPE is NONE as a source and ALL_COMMANDS as a destination; BOTTOM_OF_PIPE the other way round', () => {
		expect(execScope(['TOP_OF_PIPE'], 'src').size).toBe(0);
		expect(execScope(['TOP_OF_PIPE'], 'dst').has('COPY')).toBe(true);
		expect(execScope(['BOTTOM_OF_PIPE'], 'src').has('COMPUTE_SHADER')).toBe(true);
		expect(execScope(['BOTTOM_OF_PIPE'], 'dst').size).toBe(0);
	});
	test('access scopes do not grow by logical order: LATE alone does not publish a write made at EARLY', () => {
		const p = { ...puzzleOf('depth'), seq: [{ ...puzzleOf('depth').seq[0]!, acc: [{ stage: 'EARLY_FRAGMENT_TESTS' as const, access: 'DEPTH_STENCIL_ATTACHMENT_WRITE' as const, write: true }] }, ...puzzleOf('depth').seq.slice(1)] };
		const h = hazards(p as never, [{ ...puzzleOf('depth').solution[0]! }]).found;
		expect(h.map((f) => `${f.hazard}:${f.miss}`)).toEqual(['WRITE_AFTER_WRITE:acc1']);
	});
});

describe('barriers: valid usage (refused at the submit)', () => {
	test('an access flag needs a stage that performs it (03915, 03902, 03911)', () => {
		expect(vuids('upload', [barrier({ srcAccess: ['TRANSFER_WRITE'], dstStage: ['VERTEX_ATTRIBUTE_INPUT'], dstAccess: ['VERTEX_ATTRIBUTE_READ'] })])).toEqual(['VUID-VkMemoryBarrier2-srcAccessMask-03915']);
		expect(vuids('upload', [barrier({ srcStage: ['COPY'], srcAccess: ['TRANSFER_WRITE'], dstStage: ['VERTEX_SHADER'], dstAccess: ['VERTEX_ATTRIBUTE_READ'] })])).toEqual(['VUID-VkMemoryBarrier2-dstAccessMask-03902']);
		const s = puzzleOf('sample').solution[0]!;
		expect(vuids('sample', [{ ...s, srcStage: ['FRAGMENT_SHADER'] }])).toEqual(['VUID-VkImageMemoryBarrier2-srcAccessMask-03911']);
	});
	test('MEMORY_READ / MEMORY_WRITE and meta stages are accepted (ALL_TRANSFER, ALL_GRAPHICS, ALL_COMMANDS)', () => {
		expect(vuids('upload', [barrier({ srcStage: ['ALL_TRANSFER'], srcAccess: ['MEMORY_WRITE'], dstStage: ['ALL_GRAPHICS'], dstAccess: ['MEMORY_READ', 'VERTEX_ATTRIBUTE_READ'] })])).toEqual([]);
	});
	test('old layout must be UNDEFINED or the current one (01197); new layout not UNDEFINED (01198)', () => {
		const s = puzzleOf('sample').solution[0]!;
		expect(vuids('sample', [{ ...s, oldLayout: 'TRANSFER_SRC_OPTIMAL' }])).toContain('VUID-VkImageMemoryBarrier2-oldLayout-01197');
		expect(vuids('sample', [{ ...s, newLayout: 'UNDEFINED' }])).toContain('VUID-VkImageMemoryBarrier2-newLayout-01198');
	});
	test('a layout needs its usage flag (01212: TRANSFER_SRC; 01208: COLOR_ATTACHMENT on a depth image)', () => {
		const s = puzzleOf('sample').solution[0]!;
		expect(vuids('sample', [{ ...s, newLayout: 'TRANSFER_SRC_OPTIMAL' }])).toContain('VUID-VkImageMemoryBarrier2-oldLayout-01212');
		const d = puzzleOf('depth').solution[0]!;
		expect(vuids('depth', [{ ...d, newLayout: 'COLOR_ATTACHMENT_OPTIMAL' }])).toContain('VUID-VkImageMemoryBarrier2-oldLayout-01208');
	});
	test('the later command must find the layout it names (09600, 00180, 09588)', () => {
		const s = puzzleOf('sample').solution[0]!;
		expect(vuids('sample', [{ ...s, newLayout: 'GENERAL' }])).toEqual(['VUID-vkCmdDraw-None-09600']);
		const [t0, t1] = puzzleOf('texture').solution as [Barrier, Barrier];
		expect(vuids('texture', [{ ...t0, newLayout: 'GENERAL' }, t1])).toEqual(['VUID-VkCopyBufferToImageInfo2-dstImageLayout-00180']);
		const d = puzzleOf('depth').solution[0]!;
		expect(vuids('depth', [{ ...d, newLayout: 'GENERAL' }])).toEqual(['VUID-vkCmdBeginRendering-pRenderingInfo-09588']);
	});
	test('equal layouts are ignored (sync2): no 01197 even when they are not the current layout, and no transition', () => {
		const s = puzzleOf('sample').solution[0]!;
		expect(vuids('sample', [{ ...s, oldLayout: 'TRANSFER_SRC_OPTIMAL', newLayout: 'TRANSFER_SRC_OPTIMAL' }])).toEqual(['VUID-vkCmdDraw-None-09600']);
	});
});

describe('barriers: hazards (accepted, reported)', () => {
	test('empty barrier: the read-after-write is open (the copy is not in the first scope)', () => expect(found('upload', [barrier()])).toEqual(['READ_AFTER_WRITE:exec1']));
	test('RAW needs the memory half: without srcAccessMask the write is never made available', () =>
		expect(found('upload', [barrier({ srcStage: ['COPY'], dstStage: ['VERTEX_ATTRIBUTE_INPUT'], dstAccess: ['VERTEX_ATTRIBUTE_READ'] })])).toEqual(['READ_AFTER_WRITE:acc1']));
	test('RAW: without dstAccessMask the write is never made visible to the reader', () =>
		expect(found('upload', [barrier({ srcStage: ['COPY'], srcAccess: ['TRANSFER_WRITE'], dstStage: ['VERTEX_ATTRIBUTE_INPUT'] })])).toEqual(['READ_AFTER_WRITE:acc2']));
	test('WAR is closed by an execution dependency alone; without the source stage it stays open', () => {
		expect(found('war', [barrier({ srcStage: ['VERTEX_ATTRIBUTE_INPUT'], dstStage: ['COMPUTE_SHADER'] })])).toEqual([]);
		expect(found('war', [barrier({ srcStage: ['COMPUTE_SHADER'], dstStage: ['COMPUTE_SHADER'] })])).toEqual(['WRITE_AFTER_READ:exec1']);
		// VERTEX_SHADER pulls the logically earlier VERTEX_ATTRIBUTE_INPUT in: correct, but wider than needed.
		const wide = [barrier({ srcStage: ['VERTEX_SHADER'], dstStage: ['COMPUTE_SHADER'] })];
		expect(found('war', wide)).toEqual([]);
		expect(extras(puzzleOf('war'), wide)).toEqual([{ slot: 0, mask: 'srcStage', flag: 'VERTEX_SHADER', why: 'narrow', alt: 'VERTEX_ATTRIBUTE_INPUT' }]);
	});
	test('WAW needs a memory dependency on both sides', () => expect(found('waw', [barrier({ srcStage: ['COPY'], srcAccess: ['TRANSFER_WRITE'], dstStage: ['COPY'] })])).toEqual(['WRITE_AFTER_WRITE:acc2']));
	test('a layout transition is a write: the later read must see it (RAW against the transition)', () => {
		const s = puzzleOf('sample').solution[0]!;
		const h = hazards(puzzleOf('sample'), [{ ...s, dstAccess: [] }]).found;
		expect(h.map((f) => [f.hazard, f.before, f.miss])).toEqual([['READ_AFTER_WRITE', 'transition', 'acc2']]);
	});
	test('texture: barrier 1 needs dstAccessMask TRANSFER_WRITE although nothing came before (WAW copy after transition)', () => {
		const [t0, t1] = puzzleOf('texture').solution as [Barrier, Barrier];
		expect(found('texture', [{ ...t0, dstAccess: [] }, t1])).toEqual(['WRITE_AFTER_WRITE:acc2']);
	});
	test('old layout UNDEFINED where the contents are needed: valid, but the puzzle stays open', () => {
		const s = puzzleOf('sample').solution[0]!;
		const b = [{ ...s, oldLayout: 'UNDEFINED' as Layout }];
		expect(vuids('sample', b)).toEqual([]);
		expect(hazards(puzzleOf('sample'), b).discard).toEqual([0]);
		expect(correct(puzzleOf('sample'), b)).toBe(false);
	});
});

describe('barriers: grading', () => {
	test('every reference solution is correct and tight', () => {
		for (const p of PUZZLES) {
			expect(correct(p, p.solution)).toBe(true);
			expect(extras(p, p.solution)).toEqual([]);
		}
	});
	test('step 22\'s barrier (EARLY|LATE both sides, READ|WRITE) is correct but wider than needed', () => {
		const b = barrier({
			srcStage: ['EARLY_FRAGMENT_TESTS', 'LATE_FRAGMENT_TESTS'],
			srcAccess: ['DEPTH_STENCIL_ATTACHMENT_WRITE'],
			dstStage: ['EARLY_FRAGMENT_TESTS', 'LATE_FRAGMENT_TESTS'],
			dstAccess: ['DEPTH_STENCIL_ATTACHMENT_READ', 'DEPTH_STENCIL_ATTACHMENT_WRITE'],
			oldLayout: 'UNDEFINED',
			newLayout: 'DEPTH_ATTACHMENT_OPTIMAL',
		});
		expect(correct(puzzleOf('depth'), [b])).toBe(true);
		expect(extras(puzzleOf('depth'), [b]).map((e) => `${e.mask}:${e.flag}`)).toEqual(['srcStage:EARLY_FRAGMENT_TESTS', 'dstStage:LATE_FRAGMENT_TESTS', 'dstAccess:DEPTH_STENCIL_ATTACHMENT_READ']);
	});
	test('ALL_COMMANDS / MEMORY_* are accepted as correct and reported as broader than needed', () => {
		const b = barrier({ srcStage: ['ALL_COMMANDS'], srcAccess: ['MEMORY_WRITE'], dstStage: ['ALL_COMMANDS'], dstAccess: ['MEMORY_READ'] });
		expect(correct(puzzleOf('upload'), [b])).toBe(true);
		expect(extras(puzzleOf('upload'), [b]).map((e) => e.why)).toEqual(['part', 'part', 'part', 'part']);
	});
	test('TOP_OF_PIPE as destination = ALL_COMMANDS: correct for the WAR, reported; TOP_OF_PIPE as source adds nothing', () => {
		const b = barrier({ srcStage: ['VERTEX_ATTRIBUTE_INPUT', 'TOP_OF_PIPE'], dstStage: ['TOP_OF_PIPE'] });
		expect(correct(puzzleOf('war'), [b])).toBe(true);
		expect(extras(puzzleOf('war'), [b]).map((e) => `${e.mask}:${e.flag}:${e.why}`)).toEqual(['srcStage:TOP_OF_PIPE:drop', 'dstStage:TOP_OF_PIPE:all']);
	});
	test('MEMORY_READ counts as every read valid on its stage: broader than SAMPLED on FRAGMENT_SHADER, equal to TRANSFER_WRITE-only cases', () => {
		const s = puzzleOf('sample').solution[0]!;
		expect(extras(puzzleOf('sample'), [{ ...s, dstAccess: ['MEMORY_READ'] }]).map((e) => `${e.flag}:${e.why}`)).toEqual(['MEMORY_READ:part']);
		const c = puzzleOf('compute').solution[0]!;
		expect(extras(puzzleOf('compute'), [{ ...c, dstAccess: ['MEMORY_READ'] }]).map((e) => `${e.flag}:${e.why}`)).toEqual(['MEMORY_READ:part']);
		// On COPY the only valid write is TRANSFER_WRITE: MEMORY_WRITE means exactly that, so it is not reported.
		const u = puzzleOf('upload').solution[0]!;
		expect(extras(puzzleOf('upload'), [{ ...u, srcAccess: ['MEMORY_WRITE'] }])).toEqual([]);
	});
	test('of two flags that cover each other only the broader one is reported (greedy, broadest first)', () => {
		const u = puzzleOf('upload').solution[0]!;
		expect(extras(puzzleOf('upload'), [{ ...u, srcStage: ['COPY', 'ALL_TRANSFER'] }]).map((e) => `${e.flag}:${e.why}`)).toEqual(['ALL_TRANSFER:drop']);
		const c = puzzleOf('compute').solution[0]!;
		expect(extras(puzzleOf('compute'), [{ ...c, dstAccess: ['SHADER_STORAGE_READ', 'SHADER_READ'] }]).map((e) => `${e.flag}:${e.why}`)).toEqual(['SHADER_READ:drop']);
	});
	// Review fix: the advice must never call the needed flag removable when a broad one next to it covers it.
	test('a broad flag next to the needed one is named removable, never the needed one', () => {
		const ex = (p: string, b: Barrier) => extras(puzzleOf(p), [b]).map((e) => `${e.mask}:${e.flag}:${e.why}`);
		const war = puzzleOf('war').solution[0]!;
		expect(ex('war', { ...war, dstStage: ['TOP_OF_PIPE', 'COMPUTE_SHADER'] })).toEqual(['dstStage:TOP_OF_PIPE:all']);
		expect(ex('war', { ...war, srcStage: ['VERTEX_ATTRIBUTE_INPUT', 'BOTTOM_OF_PIPE'] })).toEqual(['srcStage:BOTTOM_OF_PIPE:all']);
		const u = puzzleOf('upload').solution[0]!;
		expect(ex('upload', { ...u, srcStage: ['COPY', 'ALL_COMMANDS'] })).toEqual(['srcStage:ALL_COMMANDS:drop']);
		expect(ex('upload', { ...u, srcAccess: ['TRANSFER_WRITE', 'MEMORY_WRITE'] })).toEqual(['srcAccess:MEMORY_WRITE:drop']);
		// Dropping every reported flag (and replacing TOP/BOTTOM by nothing) leaves a correct barrier.
		const b = { ...war, dstStage: ['TOP_OF_PIPE', 'COMPUTE_SHADER'] as Barrier['dstStage'] };
		let left: Barrier = b;
		for (const e of extras(puzzleOf('war'), [b])) left = { ...left, [e.mask]: (left[e.mask] as readonly string[]).filter((f) => f !== e.flag) } as Barrier;
		expect(correct(puzzleOf('war'), [left])).toBe(true);
	});
	test('a read in srcAccessMask has nothing to publish: reported as not needed', () => {
		const b = barrier({ srcStage: ['VERTEX_ATTRIBUTE_INPUT'], srcAccess: ['VERTEX_ATTRIBUTE_READ'], dstStage: ['COMPUTE_SHADER'] });
		expect(extras(puzzleOf('war'), [b]).map((e) => `${e.mask}:${e.flag}`)).toEqual(['srcAccess:VERTEX_ATTRIBUTE_READ']);
	});
});

describe('barriers: the scenario', () => {
	const all = (): Run<BarrierState> => {
		let r = begin(sc);
		for (const p of PUZZLES) r = solve(r, p.id, p.solution);
		return r;
	};
	test('solving every puzzle with its tight solution completes both goals, no error lines', () => {
		const r = all();
		expect(tally(r.state)).toEqual({ solved: 7, tight: 7, total: 7 });
		expect(goalsOf(sc, r.state).map((g) => g.met)).toEqual([true, true]);
		expect(complete(sc, r.state)).toBe(true);
		expect(r.log.filter((l) => l.kind === 'error')).toEqual([]);
		for (const id of ITEMS) expect(sc.status(r.state, id)).toBe('done');
	});
	test('real layer messages follow the matching hazard; a silent run is said to be silent', () => {
		const war = solve(begin(sc), 'war', [barrier({ dstStage: ['COMPUTE_SHADER'] })]);
		expect(war.last!.log.map((e) => e.ref ?? e.kind)).toEqual(['call', 'call', 'SYNC-HAZARD-WRITE-AFTER-READ', 'info', 'SYNC-HAZARD-WRITE-AFTER-READ']);
		expect(war.last!.log[4]!.text).toStartWith('vkQueueSubmit2(): WRITE_AFTER_READ hazard detected. vkCmdDispatch');
		const d = puzzleOf('depth').solution[0]!;
		const depth = solve(begin(sc), 'depth', [{ ...d, srcStage: [], srcAccess: [] }]);
		expect(depth.last!.log.some((e) => e.text.includes('which was previously written by vkCmdEndRendering'))).toBe(true);
		// Without a transition the hazard is prev -> next, not the barrier's write: the step 28 text does not fit it.
		const noT = solve(begin(sc), 'depth', [{ ...d, srcStage: [], srcAccess: [], oldLayout: 'DEPTH_ATTACHMENT_OPTIMAL' }]);
		expect(noT.last!.log.some((e) => e.text.includes('previously written by vkCmdEndRendering'))).toBe(false);
		const [t0, t1] = puzzleOf('texture').solution as [Barrier, Barrier];
		const tex = solve(begin(sc), 'texture', [t0, { ...t1, srcAccess: [] }]);
		expect(tex.last!.log.some((e) => e.text.startsWith('vkCmdPipelineBarrier2(): WRITE_AFTER_WRITE hazard detected. vkCmdPipelineBarrier2 performs image layout transition'))).toBe(true);
		expect(tex.last!.log.some((e) => e.text.startsWith('Barrier 2: WRITE_AFTER_WRITE'))).toBe(true);
		// Step 26's silent run lacked only visibility (the draw's barriers chained the dispatches): its remark follows an
		// access-scope miss of the second dispatch, never an execution miss the run did not have.
		const comp = solve(begin(sc), 'compute', [barrier({ srcStage: ['COMPUTE_SHADER'], srcAccess: ['SHADER_STORAGE_WRITE'], dstStage: ['COMPUTE_SHADER'] })]);
		expect(comp.last!.log.at(-1)!.text).toStartWith('In a run of step 26_particles with this barrier removed, synchronization validation');
		expect(comp.last!.log.at(-1)!.text).toContain('reported nothing');
		for (const slots of [[barrier()], [barrier({ srcStage: ['COMPUTE_SHADER'], dstStage: ['COMPUTE_SHADER'], dstAccess: ['SHADER_STORAGE_READ'] })]]) {
			const r = solve(begin(sc), 'compute', slots);
			expect(r.last!.log.some((e) => e.text.includes('reported nothing'))).toBe(false);
		}
	});
	test('evidence says what the probe run changed in the barrier (removed, or the field it set)', () => {
		const [t0, t1] = puzzleOf('texture').solution as [Barrier, Barrier];
		const tex = solve(begin(sc), 'texture', [{ ...t0, dstAccess: [] }, t1]);
		expect(tex.last!.log.some((e) => e.text === 'Barrier 1: The real layer, in a run of step 21_texture with dstAccessMask = VK_ACCESS_2_NONE in this barrier (first message, handles shortened):')).toBe(true);
		const war = solve(begin(sc), 'war', [barrier({ dstStage: ['COMPUTE_SHADER'] })]);
		expect(war.last!.log[3]!.text).toBe('The real layer, in a run of step 26_particles with this barrier removed (first message, handles shortened):');
		const ru = barrierScenario('ru', V);
		const r = act(ru, begin(ru), { item: 'war', verb: 'submit' });
		expect(r.last!.log.some((e) => e.text.startsWith('Настоящий слой при запуске шага 26_particles без этого барьера'))).toBe(true);
		for (const p of PUZZLES) for (const e of p.evidence ?? []) expect(e.fault === 'removed' || /^(src|dst)(Stage|Access)Mask = VK_/.test(e.fault)).toBe(true);
	});
	test('pressing "Set" without a change keeps a solved puzzle solved', () => {
		let r = solve(begin(sc), 'upload', puzzleOf('upload').solution);
		r = play(setActs(r.state, 'upload', 0, puzzleOf('upload').solution[0]!), r);
		expect(tally(r.state).solved).toBe(1);
	});
	test('an over-synchronized barrier is accepted: solved, no ★, warned', () => {
		const r = solve(begin(sc), 'upload', [barrier({ srcStage: ['ALL_COMMANDS'], srcAccess: ['MEMORY_WRITE'], dstStage: ['ALL_COMMANDS'], dstAccess: ['MEMORY_READ'] })]);
		expect(tally(r.state)).toEqual({ solved: 1, tight: 0, total: 7 });
		expect(sc.badge!(r.state, 'upload')).toBe('wider than needed');
		expect(r.last!.log.some((l) => l.kind === 'warn')).toBe(true);
	});
	test('a hazard is accepted and logged with its SYNC id; the puzzle and the barrier show it; an edit makes it stale', () => {
		let r = play([{ item: 'upload', verb: 'submit' }]);
		expect(r.last!.ok).toBe(true);
		expect(r.last!.log.find((l) => l.kind === 'error')!.ref).toBe('SYNC-HAZARD-READ-AFTER-WRITE');
		expect(sc.status(r.state, 'upload')).toBe('error');
		expect(sc.status(r.state, 'upload.b0')).toBe('error');
		r = play(setActs(r.state, 'upload', 0, barrier({ srcStage: ['COPY'] })), r);
		expect(sc.status(r.state, 'upload')).toBe('ready');
		expect(sc.badge!(r.state, 'upload')).toBe('edited, submit again');
	});
	test('a refused submit quotes the VUID and changes nothing; the puzzle card is locked with that reason', () => {
		let r = play(setActs(begin(sc).state, 'upload', 0, barrier({ srcAccess: ['TRANSFER_WRITE'] })));
		expect(sc.status(r.state, 'upload')).toBe('locked');
		expect(why(sc, r.state, 'upload').map((e) => e.ref)).toEqual(['VUID-VkMemoryBarrier2-srcAccessMask-03915']);
		const before = r.state;
		r = play([{ item: 'upload.b0', verb: 'submit' }], r);
		expect(r.last!.ok).toBe(false);
		expect(r.state).toBe(before);
		expect(r.last!.log[0]!.text).toStartWith('Validation Error: [ VUID-VkMemoryBarrier2-srcAccessMask-03915 ] vkCmdPipelineBarrier2(): The Vulkan spec states: If srcAccessMask includes VK_ACCESS_2_TRANSFER_WRITE_BIT');
	});
	test('"show all" offers every flag; switching back keeps a flag that is set but no longer shown', () => {
		let r = play([{ item: GLOBAL, verb: 'all' }]);
		const verbs = sc.verbs(r.state, 'upload.b0');
		expect(verbs.find((x) => x.id === 'srcStage')!.options!.length).toBe(STAGES.length);
		expect(verbs.find((x) => x.id === 'srcAccess')!.options!.length).toBe(ACCESSES.length);
		r = play([{ item: 'upload.b0', verb: 'srcStage', opts: { s_BLIT: true, s_COPY: true } }, { item: GLOBAL, verb: 'all' }], r);
		expect(r.state.puzzles.upload!.slots[0]!.srcStage).toEqual(['COPY', 'BLIT']);
		// The narrow editor has no BLIT box: setting COPY off keeps BLIT.
		r = play([{ item: 'upload.b0', verb: 'srcStage', opts: { s_COPY: false } }], r);
		expect(r.state.puzzles.upload!.slots[0]!.srcStage).toEqual(['BLIT']);
	});
	test('undo, reload (encode/decode) and replay rebuild the same run; another version or a stale act do not', () => {
		const r = all();
		expect(tally(undo(sc, r).state).solved).toBe(6);
		expect(replay(sc, decode(JSON.parse(JSON.stringify(encode(sc, r))), V)!)).toEqual(r);
		expect(decode(encode(sc, r), 'other')).toBeNull();
		const acts: Act[] = [{ item: 'upload', verb: 'submit' }, { item: 'upload.b0', verb: 'layout', opts: { old: 'GENERAL' } }, { item: 'upload', verb: 'submit' }];
		expect(replay(sc, acts).acts).toHaveLength(1);
		expect(() => act(sc, begin(sc), { item: 'nope', verb: 'submit' })).toThrow(StaleAct);
	});
	test('random walks from prefixes of the solved run keep status and rules consistent', () => {
		const full = all().acts;
		let seed = 12345;
		const rnd = (n: number) => ((seed = (seed * 1103515245 + 12345) % 2147483648), seed % n);
		for (let w = 0; w < 30; w++) {
			let r = replay(sc, full.slice(0, rnd(full.length + 1)));
			for (let i = 0; i < 25; i++) {
				expect(consistency(sc, r.state)).toEqual([]);
				const item = rnd(ITEMS.length + 1) === ITEMS.length ? GLOBAL : ITEMS[rnd(ITEMS.length)]!;
				const verbs = item === GLOBAL ? sc.actions!(r.state) : sc.verbs(r.state, item);
				const v = verbs[rnd(verbs.length)]!;
				const opts: Record<string, boolean | string> = {};
				for (const o of v.options ?? []) {
					if (o.kind === 'flag') opts[o.id] = rnd(4) === 0;
					else if (o.kind === 'choice') opts[o.id] = o.choices[rnd(o.choices.length)]!.id;
				}
				r = act(sc, r, { item, verb: v.id, opts });
			}
			expect(consistency(sc, r.state)).toEqual([]);
		}
	});
	test('the version is a hash of the rule sources', () => expect(ruleVersion(['a'])).not.toBe(ruleVersion(['b'])));
});
