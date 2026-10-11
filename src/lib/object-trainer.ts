/**
 * The ObjectGraph "build" exercise as an engine trainer scenario (course-kit lib/trainer.ts): the Vulkan simulator of
 * lib/objects.ts supplies the rules (blockers / destroyBlockers, with the real VUID texts) and the transitions
 * (create / destroy); this file only words its remarks in the reader's language and names the verbs. Validation
 * reports stay verbatim English, like the real layer prints them. Pure: tested in tests/object-trainer.test.ts.
 */
import { allDone, type Entry, type Scenario, type Status, type Verb } from 'course-kit/lib/trainer.ts';
import { EMPTY, blockers, create, destroy, destroyBlockers, obj, type LogLine, type SimState } from './objects';
import { tfmt } from './text';
import { SIM_KEY, courseText } from '../i18n';

/** Instance extension the exercise offers at vkCreateInstance (the debug messenger needs it). */
export const DEBUG_UTILS = 'VK_EXT_debug_utils';

/** A simulator line as a console entry: API calls and validation reports verbatim, remarks worded in `lang`. */
export function entryOf(l: LogLine, lang: string): Entry {
	if ('text' in l) return l.vuid ? { kind: l.kind, text: l.text, ref: l.vuid } : { kind: l.kind, text: l.text };
	return { kind: l.kind, text: tfmt(lang, courseText(lang)[SIM_KEY[l.msg]], l.vars) };
}

/** Card status: dangling = broken, alive = done, otherwise whether vkCreate* would be accepted now. */
export function objectStatus(state: SimState, id: string): Status {
	if (state.dangling.includes(id)) return 'error';
	if (state.alive.includes(id)) return 'done';
	return blockers(state, id).length ? 'locked' : 'ready';
}

/** The exercise on the objects `ids` (one of each kind, as the simulator keeps them). */
export function objectScenario(ids: readonly string[], lang: string): Scenario<SimState> {
	const t = courseText(lang);
	const words = (ls: LogLine[]) => ls.map((l) => entryOf(l, lang));
	const verbs = (state: SimState, id: string): Verb<SimState>[] => {
		const o = obj(id);
		return [
			{
				id: o.how,
				label: t[o.how],
				primary: true,
				options: id === 'instance' && !state.alive.includes('instance') ? [{ id: 'debugUtils', label: tfmt(lang, t.enableExt, { ext: DEBUG_UTILS }), initial: true }] : undefined,
				check: (s) => words(blockers(s, id)),
				apply: (s, opts) => {
					const r = create(s, id, opts.debugUtils ? [DEBUG_UTILS] : []);
					return { state: r.state, log: words(r.log) };
				},
			},
			{
				id: 'destroy',
				label: o.destroy ? (o.how === 'allocate' ? t.free : t.destroy) : t.destroyQ,
				check: (s) => words(destroyBlockers(s, id)),
				apply: (s) => {
					const r = destroy(s, id);
					return { state: r.state, log: words(r.log) };
				},
			},
		];
	};
	return {
		id: `objects:${ids.join(',')}`,
		init: EMPTY,
		items: ids,
		verbs,
		status: objectStatus,
		goals: [allDone(t.goalAll)],
	};
}
