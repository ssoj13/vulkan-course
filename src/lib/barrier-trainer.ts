/**
 * The barriers trainer as an engine scenario (course-kit lib/trainer.ts): the model of lib/barriers.ts supplies the
 * rules (`blockers` with the real VUID texts, `hazards`, `extras`) and the transitions; this file names the verbs, words
 * the remarks in the reader's language and maps the state onto card statuses, badges and goals. Validation reports stay
 * verbatim English. Pure: tested in tests/labs/barriers.test.ts.
 *
 * Item ids (stable: stored in readers' runs): `<puzzle>` = the puzzle card (its submit), `<puzzle>.b<k>` = barrier k of
 * the puzzle (the editor). Command cards `<puzzle>.<command>` are on the map but not in play. Verb ids: `srcStage`,
 * `srcAccess`, `dstStage`, `dstAccess` (one per mask), `layout`, `submit`; option ids `s_<STAGE>`, `a_<ACCESS>` (flags),
 * `old` / `new` (layouts).
 */
import type { Entry, Goal, Scenario, Status, Verb } from 'course-kit/lib/trainer.ts';
import { pick, tfmt } from './text';
import {
	INIT,
	LAYOUTS,
	MASKS,
	PUZZLES,
	offeredAccesses,
	offeredStages,
	puzzleOf,
	setLayouts,
	setMask,
	shortText,
	slotsOf,
	submit,
	submitBlockers,
	tally,
	type BarrierState,
	type Layout,
	type MaskKey,
	type Line,
	type Step,
} from './barriers';
import { TEXT, type BarrierText } from '../components/labs/barriers.text';

export const textOf = (lang: string): BarrierText => pick(TEXT, lang, 'barriers');

/** A text by a key built from a puzzle or command id (`p_<puzzle>`, `c_<puzzle>_<command>` ...); throws when missing. */
export function textKey(t: BarrierText, key: string): string {
	const v = (t as Readonly<Record<string, string>>)[key];
	if (v === undefined) throw new Error(`barriers: no text "${key}"`);
	return v;
}

/** Items in play: every puzzle and its barriers. */
export const ITEMS: readonly string[] = PUZZLES.flatMap((p) => [p.id, ...Array.from({ length: slotsOf(p) }, (_, k) => `${p.id}.b${k}`)]);

/** A model line as a console entry: calls and reports verbatim (ref = VUID or hazard id), remarks worded in `lang`. */
export function entryOf(l: Line, lang: string): Entry {
	const ref = l.vuid ?? l.sync;
	if ('text' in l) return ref ? { kind: l.kind, text: l.text, ref } : { kind: l.kind, text: l.text };
	const t = textOf(lang);
	const party = (x: string | number | undefined) => (x === 'transition' ? t.transition : String(x ?? ''));
	const vars = { ...l.vars, before: party(l.vars.before), after: party(l.vars.after) };
	// Puzzles with several barriers name the one a remark is about.
	const prefix = Number(l.vars.slots) > 1 && l.vars.slot !== undefined ? tfmt(lang, t.slotPrefix, { slot: l.vars.slot }) : '';
	const text = prefix + tfmt(lang, t[`m_${l.msg}`], vars);
	return ref ? { kind: l.kind, text, ref } : { kind: l.kind, text };
}

/** `<puzzle>.b<k>` -> [puzzle, k]; null for puzzle and command ids. */
export function slotOf(id: string): [string, number] | null {
	const m = /^([\w-]+)\.b(\d+)$/.exec(id);
	return m ? [m[1]!, Number(m[2])] : null;
}

/** Card status: the submit's verdict for puzzles, the last verdict (or "open") for barriers. */
export function barrierStatus(s: BarrierState, id: string): Status {
	const at = slotOf(id);
	const ps = s.puzzles[at ? at[0] : id]!;
	const solved = !ps.dirty && (ps.grade === 'tight' || ps.grade === 'wide');
	if (at) return solved ? 'done' : !ps.dirty && ps.grade === 'hazard' && ps.broken[at[1]] ? 'error' : 'ready';
	if (submitBlockers(s, id).length) return 'locked';
	if (solved) return 'done';
	return !ps.dirty && ps.grade === 'hazard' ? 'error' : 'ready';
}

/** The state word on a card: the verdict for a puzzle, the barrier in short form for a barrier. */
export function barrierBadge(s: BarrierState, id: string, t: BarrierText): string {
	const at = slotOf(id);
	if (at) {
		const p = puzzleOf(at[0]);
		return shortText(s.puzzles[p.id]!.slots[at[1]]!, p.struct === 'VkImageMemoryBarrier2');
	}
	const ps = s.puzzles[id]!;
	if (ps.grade === null) return ps.dirty ? t.badge_dirty : t.badge_new;
	if (ps.dirty) return t.badge_dirty;
	return t[`badge_${ps.grade}`];
}

/** The exercise. `version` = `ruleVersion` of barriers.ts and this file (computed by Barriers.astro). */
export function barrierScenario(lang: string, version: string): Scenario<BarrierState> {
	const t = textOf(lang);
	const words = (ls: readonly Line[]) => ls.map((l) => entryOf(l, lang));
	const step = (r: Step) => ({ state: r.state, log: words(r.log) });

	const submitVerb = (id: string, primary: boolean): Verb<BarrierState> => ({
		id: 'submit',
		label: t.v_submit,
		primary,
		check: (x) => words(submitBlockers(x, id)),
		apply: (x) => step(submit(x, id)),
	});

	/** Editor of one mask: the offered stages or access types as checkboxes holding the current mask. */
	const maskVerb = (s: BarrierState, id: string, k: number, mask: MaskKey): Verb<BarrierState> => {
		const p = puzzleOf(id);
		const cur = s.puzzles[id]!.slots[k]![mask] as readonly string[];
		const stage = mask.endsWith('Stage');
		const list: readonly string[] = stage ? offeredStages(s, p) : offeredAccesses(s, p);
		const prefix = stage ? 's_' : 'a_';
		return {
			id: mask,
			label: tfmt(lang, t.v_set, { mask: `${mask}Mask` }),
			options: list.map((x) => ({ kind: 'flag' as const, id: `${prefix}${x}`, label: x, initial: cur.includes(x) })),
			apply: (x, o) => step(setMask(x, id, k, mask, list.filter((f) => o[`${prefix}${f}`] === true))),
		};
	};

	const layoutVerb = (s: BarrierState, id: string, k: number): Verb<BarrierState> => {
		const p = puzzleOf(id);
		const b = s.puzzles[id]!.slots[k]!;
		// The current values stay choosable even when they are outside the puzzle's selection (set under "show all").
		const list = (cur: Layout) => (s.all ? LAYOUTS : LAYOUTS.filter((l) => p.layouts!.includes(l) || l === cur)).map((l) => ({ id: l, label: l }));
		return {
			id: 'layout',
			label: t.v_layout,
			options: [
				{ kind: 'choice', id: 'old', label: t.o_old, choices: list(b.oldLayout), initial: b.oldLayout },
				{ kind: 'choice', id: 'new', label: t.o_new, choices: list(b.newLayout), initial: b.newLayout },
			],
			apply: (x, o) => step(setLayouts(x, id, k, o.old as Layout, o.new as Layout)),
		};
	};

	const verbs = (s: BarrierState, item: string): Verb<BarrierState>[] => {
		const at = slotOf(item);
		if (!at) return [submitVerb(item, true)];
		const [id, k] = at;
		const image = puzzleOf(id).struct === 'VkImageMemoryBarrier2';
		return [...MASKS.map((m) => maskVerb(s, id, k, m)), ...(image ? [layoutVerb(s, id, k)] : []), submitVerb(id, false)];
	};

	const goals: Goal<BarrierState>[] = [
		{ id: 'solved', label: tfmt(lang, t.goalSolved, { n: PUZZLES.length }), met: (s) => tally(s).solved === PUZZLES.length },
		{ id: 'tight', label: t.goalTight, met: (s) => tally(s).tight === PUZZLES.length },
	];

	return {
		id: 'barriers',
		version,
		init: INIT,
		items: ITEMS,
		verbs,
		actions: (s) => [
			{
				id: 'all',
				label: s.all ? t.a_allOff : t.a_allOn,
				apply: (x) => ({ state: { ...x, all: !x.all }, log: [] }),
			},
		],
		status: barrierStatus,
		badge: (s, id) => barrierBadge(s, id, t),
		goals,
	};
}
