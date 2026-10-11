/**
 * The Vulkan object tree on the engine's concept tree (course-kit TreeView: spine / tree / outline views, details in
 * a floating window).
 *   map:   every object in scope, colored by its stage; click for what it is.
 *   build: the engine's Trainer on the scenario of lib/object-trainer.ts: objects start as "not created"; the reader
 *          creates and destroys them from the details window like an app would, and the simulator (lib/objects.ts)
 *          answers with the calls made or with the real validation error. Card statuses (any view): locked = cannot be
 *          created yet, ready = can be created now, done = alive, error = dangling. Undo, start over, the goal and the
 *          stored progress come from the engine.
 * Items (labels, parents, notes) come from ObjectGraph.astro; this island adds the object details and the scenario.
 */
import { useMemo } from 'react';
import TreeView, { type TreeGroup, type TreeItem } from 'course-kit/components/graph/TreeView.tsx';
import Trainer from 'course-kit/components/trainer/Trainer.tsx';
import './objects.css';
import { labelOf, obj } from '../lib/objects';
import { objectScenario } from '../lib/object-trainer';
import { tfmt } from '../lib/text';
import { HOW_KEY, courseText } from '../i18n';

export interface ObjectGraphViewProps {
	lang: string;
	mode: 'map' | 'build';
	items: TreeItem[];
	groups: TreeGroup[];
	/** term id -> glossary URL (resolved at build time). */
	termHref: Record<string, string>;
	/** Version of the build exercise's rules (lib/object-trainer.ts `ruleVersion`, computed at build time). */
	version: string;
}

const STATE_KEY = { done: 'stateAlive', ready: 'stateReady', locked: 'stateGhost', error: 'stateDangling' } as const;
const TAG_KEY = { done: 'tagAlive', ready: 'tagReady', locked: 'tagGhost', error: 'tagDangling' } as const;

export default function ObjectGraphView({ lang, mode, items, groups, termHref, version }: ObjectGraphViewProps) {
	const t = courseText(lang);
	const ids = items.map((i) => i.id);
	const scenario = useMemo(() => objectScenario(ids, lang, version), [ids.join(), lang, version]); // eslint-disable-line react-hooks/exhaustive-deps

	/** What the object is: its note and how it is obtained, released and made. */
	const body = (it: TreeItem) => {
		const o = obj(it.id);
		return (
			<>
				<p>{it.def}</p>
				<p className="vk-facts">
					<span className="vk-badge">{t[HOW_KEY[o.how]]}</span> <code>{o.call}</code>
					{o.destroy ? <> · {tfmt(lang, t.releasedWith, { call: o.destroy })}</> : null}
					{o.parents.length > 0 && <> · {tfmt(lang, t.madeFrom, { parents: o.parents.map((p) => labelOf(obj(p.id)) + (p.optional ? ` ${t.optional}` : '')).join(', ') })}</>}
					{o.term && termHref[o.term] ? (
						<>
							{' '}
							· <a href={termHref[o.term]}>{t.glossary}</a>
						</>
					) : null}
				</p>
			</>
		);
	};

	return (
		<div className="vk-objects">
			{mode === 'build' ? (
				<Trainer
					lang={lang}
					scenario={scenario}
					items={items}
					groups={groups}
					label={t.mapLabel}
					hint={t.hint}
					body={body}
					startHint={t.startHint}
					statusText={(s) => t[STATE_KEY[s]]}
					statusTag={(s) => t[TAG_KEY[s]]}
					progressText={(c, state) => tfmt(lang, t.progress, { n: ids.filter((id) => state.alive.includes(id)).length, total: c.total, errors: c.errors })}
				/>
			) : (
				<TreeView
					lang={lang}
					items={items}
					groups={groups}
					label={t.mapLabel}
					hint={t.hint}
					// The full map starts with the device's children in view.
					reveal={items.filter((i) => i.parent === 'device').map((i) => i.id)}
					renderDetails={body}
				/>
			)}
		</div>
	);
}
