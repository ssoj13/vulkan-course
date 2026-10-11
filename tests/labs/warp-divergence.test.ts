import { describe, expect, test } from 'bun:test';
import { MAX_X, PROGRAM, WARP, condSrc, inputs, run, takes } from '../../src/lib/labs/warp-divergence';
import { textErrors } from '../../src/lib/text';
import { TEXT } from '../../src/components/labs/warp-divergence.text';

const lanes = inputs('lane', 1);
const pre = PROGRAM.pre.length;
const post = PROGRAM.post.length;
const nIf = PROGRAM.if.length;
const nElse = PROGRAM.else.length;

describe('warp-divergence model', () => {
	test('inputs: lane ids, or seeded values in 0..MAX_X', () => {
		expect(lanes).toEqual(Array.from({ length: WARP }, (_, i) => i));
		const a = inputs('random', 7);
		expect(a).toHaveLength(WARP);
		expect(a.every((x) => Number.isInteger(x) && x >= 0 && x <= MAX_X)).toBe(true);
		expect(inputs('random', 7)).toEqual(a);
		expect(inputs('random', 8)).not.toEqual(a);
	});

	test('conditions', () => {
		expect(takes('lt', 16, 15)).toBe(true);
		expect(takes('lt', 16, 16)).toBe(false);
		expect(takes('even', 0, 4)).toBe(true);
		expect(takes('even', 0, 5)).toBe(false);
		expect(lanes.some((x) => takes('never', 0, x))).toBe(false);
		expect(condSrc('lt', 9)).toBe('x < 9');
	});

	test('a divergent warp issues both branches, each with the other lanes masked', () => {
		const r = run(lanes, 'lt', 16);
		expect(r.divergent).toBe(true);
		expect(r.cycles).toBe(pre + nIf + nElse + post);
		expect(r.frames).toHaveLength(r.cycles + 1);
		const ifFrame = r.frames.find((f) => f.at?.block === 'if')!;
		const elseFrame = r.frames.find((f) => f.at?.block === 'else')!;
		expect(ifFrame.active).toEqual(lanes.map((x) => x < 16));
		expect(elseFrame.active).toEqual(lanes.map((x) => x >= 16));
		// Before the branch and after reconvergence every lane runs.
		for (const f of r.frames.filter((f) => f.at?.block === 'pre' || f.at?.block === 'post')) expect(f.active.every(Boolean)).toBe(true);
		expect(r.efficiency).toBeCloseTo((WARP * (pre + post) + 16 * nIf + 16 * nElse) / (WARP * r.cycles));
	});

	test('one instruction per slot: cycles count up by one, frame 0 issues nothing', () => {
		const r = run(lanes, 'even', 0);
		expect(r.frames[0]!.at).toBeNull();
		expect(r.frames[0]!.active.some(Boolean)).toBe(false);
		r.frames.forEach((f, i) => expect(f.cycle).toBe(i));
	});

	test('a branch no lane takes is skipped', () => {
		const none = run(lanes, 'never', 0);
		expect(none.divergent).toBe(false);
		expect(none.cycles).toBe(none.allElse);
		expect(none.frames.some((f) => f.at?.block === 'if')).toBe(false);
		expect(none.efficiency).toBe(1);
		const all = run(lanes, 'lt', WARP);
		expect(all.cycles).toBe(all.allIf);
		expect(all.frames.some((f) => f.at?.block === 'else')).toBe(false);
	});

	test('a warp has exactly 32 lanes', () => {
		expect(() => run([1, 2, 3], 'lt', 1)).toThrow();
	});

	test('texts', () => {
		expect(textErrors(TEXT, 'warp-divergence')).toEqual([]);
	});
});
