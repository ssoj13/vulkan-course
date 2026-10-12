/**
 * Barriers trainer: the engine's Trainer (course-kit, on TreeView) running the scenario of src/lib/barrier-trainer.ts.
 * Each puzzle is a small command stream drawn as a chain: the puzzle card (its submit and verdict), then its commands and
 * barrier slots in submission order. Barrier cards hold the editor (stage and access masks as checkboxes, layouts as
 * choices) and show the current struct as their badge; command cards are fixed context. The global action switches
 * the editors between each puzzle's selection of flags and every flag. Undo, start over, goals and the stored run come
 * from the engine.
 */
import { useMemo } from 'react';
import Trainer from 'course-kit/components/trainer/Trainer.tsx';
import { groupColor, type TreeGroup, type TreeItem } from 'course-kit/components/graph/TreeView.tsx';
import type { Status } from 'course-kit/lib/trainer.ts';
import { tfmt } from '../../lib/text';
import { PUZZLES, tally } from '../../lib/barriers';
import { barrierScenario, slotOf, textKey, textOf } from '../../lib/barrier-trainer';

export interface BarriersViewProps {
	lang: string;
	/** Version of the rules (lib/object-trainer.ts `ruleVersion` over the sources, computed at build time). */
	version: string;
}

export default function BarriersView({ lang, version }: BarriersViewProps) {
	const t = textOf(lang);
	const w = (k: string) => textKey(t, k);
	const scenario = useMemo(() => barrierScenario(lang, version), [lang, version]);
	const { items, groups } = useMemo(() => {
		const items: TreeItem[] = [];
		for (const p of PUZZLES) {
			items.push({ id: p.id, label: w(`p_${p.id}`), sub: w(`k_${p.id}`), def: w(`q_${p.id}`), group: p.id, parent: null });
			// The stream as a chain: every command or slot hangs off the one before it.
			let parent = p.id;
			let k = 0;
			for (const x of p.seq) {
				const id = x.kind === 'cmd' ? `${p.id}.${x.id}` : `${p.id}.b${k}`;
				if (x.kind === 'cmd') {
					const sub = x.acc.map((a) => `${a.stage} · ${a.access} (${a.write ? t.write : t.read})`).join(', ');
					items.push({ id, label: x.call, code: true, sub, def: w(`c_${p.id}_${x.id}`), group: p.id, parent, rel: parent === p.id ? undefined : t.then });
				} else {
					items.push({ id, label: p.struct, code: true, sub: w(`b_${p.id}_${k}`), def: w(`q_${p.id}`), group: p.id, parent, rel: parent === p.id ? undefined : t.then });
					k++;
				}
				parent = id;
			}
		}
		// One palette entry per puzzle (index 5 = neutral gray is skipped, like ObjectGraph).
		const groups: TreeGroup[] = PUZZLES.map((p, i) => ({ id: p.id, label: w(`p_${p.id}`), color: groupColor(i < 5 ? i : i + 1) }));
		return { items, groups };
	}, [t]);

	const statusText = (s: Status, it: TreeItem) => {
		if (s === 'done') return t.stDone;
		if (s === 'error') return t.stError;
		if (slotOf(it.id)) return t.stBarrier;
		return s === 'locked' ? t.stLocked : t.stReady;
	};
	const statusTag = (s: Status) => ({ locked: t.tagLocked, ready: t.tagReady, done: t.tagDone, error: t.tagError })[s];

	return (
		<div className="vk-barriers">
			<Trainer
				lang={lang}
				scenario={scenario}
				items={items}
				groups={groups}
				label={t.mapLabel}
				hint={t.hint}
				body={(it) => <p>{it.def}</p>}
				startHint={t.startHint}
				statusText={statusText}
				statusTag={statusTag}
				progressText={(c, state) => {
					const n = tally(state);
					return tfmt(lang, t.progress, { solved: n.solved, total: n.total, tight: n.tight, errors: c.errors });
				}}
			/>
		</div>
	);
}
