/**
 * Frame assembly trainer: the engine's Trainer (course-kit, on TreeView) running the scenario of
 * src/lib/frame-trainer.ts. Cards: the two frame slots (fence, command buffer, image_available), the swapchain with its
 * images and their render_finished semaphores, and the queue; every card shows its exact state as a badge. Host calls
 * are verbs on the cards (fence: create / wait / reset, command buffer: record / submit, image: present) or global
 * actions (acquire, GPU step, vblank). Undo, start over, goals and the stored run come from the engine.
 */
import { useMemo } from 'react';
import Trainer from 'course-kit/components/trainer/Trainer.tsx';
import { groupColor, type TreeGroup, type TreeItem } from 'course-kit/components/graph/TreeView.tsx';
import type { Status } from 'course-kit/lib/trainer.ts';
import { tfmt } from '../../lib/text';
import { IMAGES, SLOTS, frameCounts } from '../../lib/frames';
import { frameScenario, itemLabel, textOf } from '../../lib/frame-trainer';

export interface FrameAssemblyViewProps {
	lang: string;
	/** Version of the rules (lib/object-trainer.ts `ruleVersion` over the sources, computed at build time). */
	version: string;
}

/** Kind of a card, from its id (see frame-trainer.ts). */
const kindOf = (id: string) => (id.startsWith('cb') ? 'cb' : id.startsWith('img') ? 'img' : /^f\d/.test(id) ? 'fence' : 'info');

export default function FrameAssemblyView({ lang, version }: FrameAssemblyViewProps) {
	const t = textOf(lang);
	const scenario = useMemo(() => frameScenario(lang, version), [lang, version]);
	const { items, groups } = useMemo(() => {
		const card = (id: string, group: string, parent: string | null, sub: string, def: string): TreeItem => ({ id, label: itemLabel(id, lang), group, parent, code: true, sub, def });
		const items: TreeItem[] = [
			...SLOTS.flatMap((k) => [
				{ id: `slot${k}`, label: tfmt(lang, t.slot, { k }), group: `slot${k}`, parent: null, def: t.def_slot },
				card(`f${k}`, `slot${k}`, `slot${k}`, t.sub_fence, t.def_fence),
				card(`cb${k}`, `slot${k}`, `slot${k}`, t.sub_cb, t.def_cb),
				card(`ia${k}`, `slot${k}`, `slot${k}`, t.sub_sem, t.def_ia),
			]),
			card('swapchain', 'swapchain', null, t.g_swapchain, t.def_swapchain),
			...IMAGES.flatMap((i) => [card(`img${i}`, 'swapchain', 'swapchain', t.sub_img, t.def_img), card(`rf${i}`, 'swapchain', `img${i}`, t.sub_sem, t.def_rf)]),
			card('queue', 'queue', null, t.g_queue, t.def_queue),
		];
		// Palette entries: the two slots, the swapchain, the queue (5 = neutral gray is skipped, like ObjectGraph).
		const groups: TreeGroup[] = [
			...SLOTS.map((k) => ({ id: `slot${k}`, label: tfmt(lang, t.g_slot, { k }), color: groupColor(k) })),
			{ id: 'swapchain', label: t.g_swapchain, color: groupColor(6) },
			{ id: 'queue', label: t.g_queue, color: groupColor(4) },
		];
		return { items, groups };
	}, [lang, t]);

	const statusText = (s: Status, it: TreeItem) => {
		if (s === 'locked') return t.stLocked;
		if (s === 'ready') return t.stReady;
		const k = kindOf(it.id);
		return k === 'fence' ? t.stFence : k === 'cb' ? t.stCb : k === 'img' ? t.stImg : t.stInfo;
	};
	const statusTag = (s: Status, it: TreeItem) => {
		if (s === 'locked') return t.tagLocked;
		if (s === 'ready') return t.tagReady;
		// The badge already says what "done" means for a fence (signaled) or an image (waiting for vblank, on screen);
		// semaphores, the swapchain and the queue have no step of their own.
		return kindOf(it.id) === 'cb' ? t.tagCb : '';
	};

	return (
		<div className="vk-frames">
			<Trainer
				lang={lang}
				scenario={scenario}
				items={items}
				groups={groups}
				label={t.mapLabel}
				hint={t.hint}
				body={(it) => <p>{it.def}</p>}
				startHint={t.startHint}
				// A frame prints ~10 lines and the engine console keeps its scroll at the top: show only the newest ones.
				lines={8}
				statusText={statusText}
				statusTag={statusTag}
				progressText={(c, state) => {
					const f = frameCounts(state);
					return tfmt(lang, t.progress, { n: state.frames.length, h: f.hazards, errors: c.errors });
				}}
			/>
		</div>
	);
}
