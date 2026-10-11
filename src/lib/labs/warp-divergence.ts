/**
 * Model of the warp-divergence lab: one warp of 32 lanes runs a tiny shader with an if/else, SIMT style. Every issue
 * slot sends ONE instruction to the whole warp; lanes whose thread is not on that path are masked off (CUDA
 * Programming Guide, "SIMT Architecture": "the warp executes each branch path taken, disabling threads that are not
 * on that path"). A branch nobody takes is skipped. Instruction costs are illustrative: one slot each.
 * Pure module: no DOM, no React, tested with `bun test`.
 */

/** Lanes in an NVIDIA warp (AMD wavefronts are 32 or 64). */
export const WARP = 32;

/** Input values are 0..MAX_X, so the threshold slider covers every split from "none" to "all". */
export const MAX_X = WARP - 1;

/** Which part of the program an instruction belongs to. */
export type Block = 'pre' | 'if' | 'else' | 'post';

/** Condition presets the reader picks from; `never` is `x < 0`, false for every input. */
export const CONDS = ['lt', 'even', 'never'] as const;
export type Cond = (typeof CONDS)[number];

/** The source line of a condition, as the program listing shows it. */
export function condSrc(cond: Cond, t: number): string {
	return cond === 'lt' ? `x < ${t}` : cond === 'even' ? 'x % 2 == 0' : 'x < 0';
}

/** True when a lane with input `x` takes the if-path. */
export function takes(cond: Cond, t: number, x: number): boolean {
	return cond === 'lt' ? x < t : cond === 'even' ? x % 2 === 0 : x < 0;
}

/**
 * The program, one line per instruction. The two branches have different lengths on purpose: a divergent warp pays
 * for both. `{cond}` is replaced with the condition's source.
 */
export const PROGRAM: Readonly<Record<Block, readonly string[]>> = {
	pre: ['float x = data[lane];', 'bool c = {cond};'],
	if: ['float a = x * 2.0;', 'a = a + 1.0;', 'a = a * a;', 'out = a;'],
	else: ['float b = x - 1.0;', 'b = b * 3.0;', 'b = b + x;', 'b = b * b;', 'b = b - 2.0;', 'out = b;'],
	post: ['result[lane] = out;'],
};

/** Order the warp walks the blocks in. Real hardware may run either branch first; the cost is the same. */
const ORDER: readonly Block[] = ['pre', 'if', 'else', 'post'];

/** Inputs: x = the lane id, or a random value per lane (seeded, so a seed always gives the same warp). */
export type Inputs = 'lane' | 'random';

/** mulberry32: a tiny deterministic PRNG, enough for reproducible "random" inputs. */
function rng(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let r = Math.imul(a ^ (a >>> 15), 1 | a);
		r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
		return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
	};
}

/** The x of every lane. */
export function inputs(kind: Inputs, seed: number): number[] {
	if (kind === 'lane') return Array.from({ length: WARP }, (_, i) => i);
	const r = rng(seed);
	return Array.from({ length: WARP }, () => Math.floor(r() * (MAX_X + 1)));
}

/** One issue slot of the timeline (frame 0 is the warp before its first instruction). */
export interface WarpFrame {
	/** Block and line of the instruction issued in this slot; null on frame 0. */
	at: { block: Block; line: number } | null;
	/** Lanes that execute this instruction (all false on frame 0). */
	active: readonly boolean[];
	/** Issue slots used so far, this one included. */
	cycle: number;
	/** Lane-slots that did useful work so far (sum of active lanes). */
	busy: number;
}

export interface WarpRun {
	/** Branch each lane takes: true = if. */
	taken: boolean[];
	frames: WarpFrame[];
	/** Issue slots of the whole run. */
	cycles: number;
	/** Cost if every lane agreed on the if-path, and on the else-path. */
	allIf: number;
	allElse: number;
	/** Both branches were executed by this warp. */
	divergent: boolean;
	/** Useful lane-slots / (WARP × cycles), 0..1. */
	efficiency: number;
}

const len = (b: Block) => PROGRAM[b].length;

/** Runs the warp: which lanes take which branch, and one frame per issued instruction. */
export function run(xs: readonly number[], cond: Cond, t: number): WarpRun {
	if (xs.length !== WARP) throw new Error(`run: a warp has ${WARP} lanes, got ${xs.length}`);
	const taken = xs.map((x) => takes(cond, t, x));
	const all = taken.map(() => true);
	const mask: Record<Block, boolean[]> = { pre: all, if: taken, else: taken.map((c) => !c), post: all };
	const frames: WarpFrame[] = [{ at: null, active: taken.map(() => false), cycle: 0, busy: 0 }];
	let cycle = 0;
	let busy = 0;
	for (const block of ORDER) {
		const active = mask[block];
		const n = active.filter(Boolean).length;
		// A branch no lane takes is jumped over: it costs nothing.
		if (n === 0) continue;
		for (let line = 0; line < len(block); line++) {
			cycle++;
			busy += n;
			frames.push({ at: { block, line }, active, cycle, busy });
		}
	}
	const base = len('pre') + len('post');
	return {
		taken,
		frames,
		cycles: cycle,
		allIf: base + len('if'),
		allElse: base + len('else'),
		divergent: taken.some(Boolean) && !taken.every(Boolean),
		efficiency: busy / (WARP * cycle),
	};
}
