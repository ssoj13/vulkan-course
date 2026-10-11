/**
 * The Vulkan object tree on the engine's concept tree (course-kit TreeView: spine / tree / outline views, details in
 * a floating window).
 *   map:   every object in scope, colored by its stage; click for what it is.
 *   build: objects start as "not created"; the reader creates and destroys them from the details window like an app
 *          would, and the simulator (lib/objects.ts) answers with the calls made or with the real validation error.
 *          Card states (any view): ghost = cannot be created yet, ready = can be created now, alive, dangling.
 * Items (labels, parents, notes) come from ObjectGraph.astro; this island adds the simulator state.
 */
import { useState } from 'react';
import TreeView, { type TreeGroup, type TreeItem } from 'course-kit/components/graph/TreeView.tsx';
import './objects.css';
import { EMPTY, blockers, create, destroy, labelOf, obj, type LogLine, type SimState } from '../lib/objects';
import { tfmt } from '../lib/text';
import { HOW_KEY, SIM_KEY, courseText } from '../i18n';

export interface ObjectGraphViewProps {
	lang: string;
	mode: 'map' | 'build';
	items: TreeItem[];
	groups: TreeGroup[];
	/** term id -> glossary URL (resolved at build time). */
	termHref: Record<string, string>;
}

/** Instance extension the build exercise offers at vkCreateInstance. */
const DEBUG_UTILS = 'VK_EXT_debug_utils';

type Status = 'ghost' | 'alive' | 'dangling' | 'ready';

export default function ObjectGraphView({ lang, mode, items, groups, termHref }: ObjectGraphViewProps) {
	const t = courseText(lang);
	/** A console line in the reader's language: API calls and validation reports stay verbatim. */
	const line = (l: LogLine) => ('text' in l ? l.text : tfmt(lang, t[SIM_KEY[l.msg]], l.vars));
	const [state, setState] = useState<SimState>(EMPTY);
	const [log, setLog] = useState<LogLine[]>([]);
	/** The result of the last action and the object it was taken on, shown in that object's window. */
	const [last, setLast] = useState<{ id: string; log: LogLine[] } | null>(null);
	const [debugUtils, setDebugUtils] = useState(true);
	const ids = items.map((i) => i.id);

	const status = (id: string): Status => {
		if (state.dangling.includes(id)) return 'dangling';
		if (state.alive.includes(id)) return 'alive';
		return blockers(state, id).length ? 'ghost' : 'ready';
	};
	const apply = (id: string, r: { state: SimState; log: LogLine[] }) => {
		setState(r.state);
		setLog((l) => [...l, ...r.log].slice(-40));
		setLast({ id, log: r.log });
	};
	const restart = () => {
		setState(EMPTY);
		setLog([]);
		setLast(null);
	};
	const created = ids.filter((id) => state.alive.includes(id)).length;
	const errors = log.filter((l) => l.kind === 'error').length;
	const STATE_KEY = { alive: 'stateAlive', ready: 'stateReady', ghost: 'stateGhost', dangling: 'stateDangling' } as const;

	const details = (it: TreeItem) => {
		const o = obj(it.id);
		const s = status(it.id);
		// Why it cannot be created: the missing parents as the validation layer would report them.
		const why = s === 'ghost' ? blockers(state, it.id).filter((l) => l.kind !== 'info') : [];
		return (
			<>
				<p>{it.def}</p>
				<p className="vk-facts">
					<span className="vk-badge">{t[HOW_KEY[o.how]]}</span> <code>{o.call}</code>
					{o.destroy ? <> · {tfmt(lang, t.releasedWith, { call: o.destroy })}</> : null}
					{o.parents.length > 0 && (
						<> · {tfmt(lang, t.madeFrom, { parents: o.parents.map((p) => labelOf(obj(p.id)) + (p.optional ? ` ${t.optional}` : '')).join(', ') })}</>
					)}
					{o.term && termHref[o.term] ? (
						<>
							{' '}
							· <a href={termHref[o.term]}>{t.glossary}</a>
						</>
					) : null}
				</p>
				{mode === 'build' && (
					<>
						<p className={`vk-state is-${s}`}>{t[STATE_KEY[s]]}</p>
						{why.length > 0 && (
							<ol className="vk-log">
								{why.map((l, i) => (
									<li key={i} className={`is-${l.kind}`}>
										{line(l)}
									</li>
								))}
							</ol>
						)}
						<div className="lab-row vk-actions">
							{o.id === 'instance' && !state.alive.includes('instance') && (
								<label className="vk-check">
									<input type="checkbox" checked={debugUtils} onChange={(e) => setDebugUtils(e.target.checked)} /> {tfmt(lang, t.enableExt, { ext: DEBUG_UTILS })}
								</label>
							)}
							<button type="button" className="lab-button is-primary" onClick={() => apply(o.id, create(state, o.id, debugUtils ? [DEBUG_UTILS] : []))}>
								{t[o.how]}
							</button>
							<button type="button" className="lab-button" onClick={() => apply(o.id, destroy(state, o.id))}>
								{o.destroy ? (o.how === 'allocate' ? t.free : t.destroy) : t.destroyQ}
							</button>
						</div>
						{last?.id === o.id && (
							<div aria-live="polite">
								<p className="vk-last">{t.lastResult}</p>
								<ol className="vk-log">
									{last.log.map((l, i) => (
										<li key={i} className={`is-${l.kind}`}>
											{line(l)}
										</li>
									))}
								</ol>
							</div>
						)}
					</>
				)}
			</>
		);
	};

	return (
		<div className="vk-objects">
			<TreeView
				lang={lang}
				items={items}
				groups={groups}
				label={t.mapLabel}
				hint={t.hint}
				// Build exercises show every object; the full map starts with the device's children in view.
				reveal={mode === 'build' ? 'all' : items.filter((i) => i.parent === 'device').map((i) => i.id)}
				cardClass={mode === 'build' ? (it) => `is-${status(it.id)}` : undefined}
				renderDetails={details}
			>
				{mode === 'build' && (
					<div className="vk-console">
						<div className="lab-row vk-console-head">
							<span>{tfmt(lang, t.progress, { n: created, total: ids.length, errors })}</span>
							<button type="button" className="lab-button" onClick={restart}>
								{t.startOver}
							</button>
						</div>
						<ol className="vk-log" aria-live="polite">
							{log.length === 0 && <li className="is-info">{t.startHint}</li>}
							{log.map((l, i) => (
								<li key={i} className={`is-${l.kind}`}>
									{line(l)}
								</li>
							))}
						</ol>
					</div>
				)}
			</TreeView>
		</div>
	);
}
