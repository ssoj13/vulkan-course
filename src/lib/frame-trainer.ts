/**
 * The frame assembly trainer as an engine scenario (course-kit lib/trainer.ts): the model of lib/frames.ts supplies
 * the rules (`*Blockers`, with the real VUID texts) and the transitions; this file names the verbs, words the remarks
 * in the reader's language and maps the state onto card statuses, badges and goals. Validation reports stay verbatim
 * English. Pure: tested in tests/labs/frame-assembly.test.ts.
 *
 * Item ids (stable: stored in readers' runs): `f<k>` fence, `cb<k>` command buffer, `ia<k>` image_available of slot k,
 * `img<i>` swapchain image, `rf<i>` its render_finished, `swapchain`, `queue`.
 */
import type { Entry, Goal, Scenario, Status, Verb } from 'course-kit/lib/trainer.ts';
import { pick, tfmt } from './text';
import {
	IMAGES,
	INIT,
	SEMS,
	SLOTS,
	STAGES,
	acquire,
	acquireBlockers,
	createFence,
	createdFences,
	fenceState,
	frameCounts,
	gpuBlockers,
	gpuStep,
	heldImages,
	peBlockers,
	peStep,
	present,
	presentBlockers,
	record,
	recordBlockers,
	resetBlockers,
	resetFence,
	semName,
	semState,
	stageName,
	submit,
	submitBlockers,
	waitBlockers,
	waitFence,
	type FrameState,
	type Line,
	type Stage,
	type Step,
} from './frames';
import { TEXT, type FrameText } from '../components/labs/frame-assembly.text';

export const ITEMS = [...SLOTS.flatMap((k) => [`f${k}`, `cb${k}`, `ia${k}`]), 'swapchain', ...IMAGES.flatMap((i) => [`img${i}`, `rf${i}`]), 'queue'] as const;

export const textOf = (lang: string): FrameText => pick(TEXT, lang, 'frame-assembly');

/** A model line as a console entry: calls and reports verbatim (ref = VUID or hazard id), remarks worded in `lang`. */
export function entryOf(l: Line, lang: string): Entry {
	if ('text' in l) {
		const ref = l.vuid ?? l.sync;
		return ref ? { kind: l.kind, text: l.text, ref } : { kind: l.kind, text: l.text };
	}
	return { kind: l.kind, text: tfmt(lang, textOf(lang)[`m_${l.msg}`], l.vars) };
}

const num = (id: string) => Number(id.replace(/^\D+/, ''));
const imgId = (i: number) => `img${i}`;

/**
 * Present's wait candidates: semaphores holding a signal no one waits for first (render_finished before
 * image_available), then the rest; the first is the default, and the status must agree with it.
 */
function presentWaits(s: FrameState): string[] {
	const free = (id: string) => s.sems[id]!.sig !== 'none' && s.sems[id]!.claim === null;
	const order = [...SEMS.filter((id) => id.startsWith('rf')), ...SEMS.filter((id) => id.startsWith('ia'))];
	return [...order.filter(free), ...order.filter((id) => !free(id))];
}

/** Card status: the main verb's verdict, `done` where the item's job is done (and for items with no verbs). */
export function frameStatus(s: FrameState, id: string): Status {
	const n = num(id);
	if (id.startsWith('f')) {
		const st = fenceState(s, n);
		if (st === 'none') return 'ready';
		if (st === 'signaled') return 'done';
		return waitBlockers(s, n, true).length ? 'locked' : 'ready';
	}
	if (id.startsWith('cb')) {
		const cb = s.cbs[n]!;
		if (cb.state === 'executable') return 'done';
		return recordBlockers(s, n, heldImages(s)[0] ?? null).length ? 'locked' : 'ready';
	}
	if (id.startsWith('img')) {
		const o = s.imgs[n]!.owner;
		if (o === 'queued' || o === 'shown') return 'done';
		return presentBlockers(s, n, presentWaits(s)[0]!).length ? 'locked' : 'ready';
	}
	return 'done';
}

/** The state word on a card. */
export function frameBadge(s: FrameState, id: string, t: FrameText, lang: string): string {
	const n = num(id);
	const f = (text: string, vars: Record<string, string | number>) => tfmt(lang, text, vars);
	if (id.startsWith('f')) {
		const st = fenceState(s, n);
		return st === 'none' ? t.b_none : st === 'pending' ? f(t.b_pending, { n: s.fences[n]!.batch! }) : t[`b_${st}`];
	}
	if (id.startsWith('cb')) {
		const cb = s.cbs[n]!;
		if (cb.state === 'executable') return f(t.cb_executable, { i: cb.image! });
		if (cb.state === 'pending') return f(t.cb_pending, { n: s.batches.filter((b) => b.slot === n && !b.done)[0]!.id });
		return t[`cb_${cb.state}`];
	}
	if (id.startsWith('ia') || id.startsWith('rf')) {
		const st = semState(s, id);
		const c = s.sems[id]!.claim;
		if (st === 'waited' && c?.by === 'batch') return f(t.s_waited, { n: c.id });
		if (st === 'present' && c?.by === 'present') return f(t.s_present, { i: c.img });
		return t[`s_${st as 'unsignaled' | 'pending' | 'signaled'}`];
	}
	if (id.startsWith('img')) {
		const m = s.imgs[n]!;
		if (m.owner === 'app') return m.rendered === null ? t.i_app : f(t.i_rendered, { n: m.rendered });
		return t[`i_${m.owner}`];
	}
	if (id === 'swapchain') return f(t.sw_badge, { screen: s.screen ?? t.none, queued: s.queue.length ? s.queue.join(', ') : t.none });
	const pending = s.batches.filter((b) => !b.done);
	return pending.length ? f(t.q_pending, { list: pending.map((b) => `#${b.id} (cmd[${b.slot}])`).join(', ') }) : t.q_idle;
}

/** The exercise. `version` = `ruleVersion` of frames.ts and this file (computed by FrameAssembly.astro). */
export function frameScenario(lang: string, version: string): Scenario<FrameState> {
	const t = textOf(lang);
	const words = (ls: readonly Line[]) => ls.map((l) => entryOf(l, lang));
	const step = (r: Step) => ({ state: r.state, log: words(r.log) });

	const fenceVerbs = (s: FrameState, k: number): Verb<FrameState>[] =>
		s.fences[k]!.created
			? [
					{
						id: 'wait',
						label: t.v_wait,
						primary: true,
						options: [{ kind: 'choice', id: 'timeout', label: t.o_timeout, choices: [{ id: 'inf', label: t.c_inf }, { id: 'zero', label: t.c_zero }], initial: 'inf' }],
						check: (x, o) => words(waitBlockers(x, k, o.timeout === 'inf')),
						apply: (x, o) => step(waitFence(x, k, o.timeout === 'inf')),
					},
					{ id: 'reset', label: t.v_reset, check: (x) => words(resetBlockers(x, k)), apply: (x) => step(resetFence(x, k)) },
				]
			: [
					{
						id: 'create',
						label: t.v_create,
						primary: true,
						options: [{ kind: 'flag', id: 'signaled', label: t.o_signaled, initial: true }],
						apply: (x, o) => step(createFence(x, k, o.signaled === true)),
					},
				];

	const cbVerbs = (k: number): Verb<FrameState>[] => [
		{
			id: 'record',
			label: t.v_record,
			primary: true,
			options: [{ kind: 'item', id: 'image', label: t.o_image, of: (x) => heldImages(x).map(imgId) }],
			check: (x, o) => words(recordBlockers(x, k, o.image ? num(String(o.image)) : null)),
			apply: (x, o) => step(record(x, k, num(String(o.image)))),
		},
		{
			id: 'submit',
			label: t.v_submit,
			options: [
				{ kind: 'item', id: 'wait', label: t.o_wait, of: () => SEMS },
				{ kind: 'choice', id: 'stage', label: t.o_stage, choices: STAGES.map((st) => ({ id: st, label: stageName(st) })), initial: STAGES[0] },
				{ kind: 'item', id: 'signal', label: t.o_signal, of: () => SEMS },
				{ kind: 'item', id: 'fence', label: t.o_fence, of: createdFences },
			],
			check: (x, o) => words(submitBlockers(x, k, { wait: String(o.wait), stage: o.stage as Stage, signal: String(o.signal), fence: String(o.fence) })),
			apply: (x, o) => step(submit(x, k, { wait: String(o.wait), stage: o.stage as Stage, signal: String(o.signal), fence: String(o.fence) })),
		},
	];

	const imgVerbs = (i: number): Verb<FrameState>[] => [
		{
			id: 'present',
			label: t.v_present,
			primary: true,
			options: [{ kind: 'item', id: 'wait', label: t.o_wait, of: presentWaits }],
			check: (x, o) => words(presentBlockers(x, i, String(o.wait))),
			apply: (x, o) => step(present(x, i, String(o.wait))),
		},
	];

	const goals: Goal<FrameState>[] = [
		{ id: 'frames', label: t.goalFrames, met: (s) => frameCounts(s).perSlot.every((n) => n >= 2) },
		{ id: 'clean', label: t.goalClean, met: (s) => frameCounts(s).perSlot.every((n) => n >= 2) && frameCounts(s).hazards === 0 },
		{ id: 'overlap', label: t.goalOverlap, met: (s) => s.overlap },
	];

	return {
		id: 'frame-assembly',
		version,
		init: INIT,
		items: ITEMS,
		verbs: (s, id) => (id.startsWith('f') ? fenceVerbs(s, num(id)) : id.startsWith('cb') ? cbVerbs(num(id)) : id.startsWith('img') ? imgVerbs(num(id)) : []),
		actions: () => [
			{
				id: 'acquire',
				label: t.a_acquire,
				options: [{ kind: 'item', id: 'sem', label: t.o_sem, of: () => SEMS }],
				check: (x, o) => words(acquireBlockers(x, String(o.sem))),
				apply: (x, o) => step(acquire(x, String(o.sem))),
			},
			{ id: 'gpu', label: t.a_gpu, check: (x) => words(gpuBlockers(x)), apply: (x) => step(gpuStep(x)) },
			{ id: 'pe', label: t.a_pe, check: (x) => words(peBlockers(x)), apply: (x) => step(peStep(x)) },
		],
		status: frameStatus,
		badge: (s, id) => frameBadge(s, id, t, lang),
		goals,
	};
}

/** Labels of the items (Vulkan names, the same in every language except "image"). */
export function itemLabel(id: string, lang: string): string {
	const n = num(id);
	if (id.startsWith('f')) return `in_flight[${n}]`;
	if (id.startsWith('cb')) return `cmd[${n}]`;
	if (id.startsWith('ia') || id.startsWith('rf')) return semName(id);
	if (id.startsWith('img')) return tfmt(lang, textOf(lang).image, { i: n });
	return id === 'swapchain' ? 'VkSwapchainKHR' : 'VkQueue';
}
