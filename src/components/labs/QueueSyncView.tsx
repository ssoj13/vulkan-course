/**
 * Queue-sync lab (React island): a host program of vkQueueSubmit2 calls and fence waits on the graphics, async
 * compute and transfer queues of the course GPU, drawn as Gantt lanes (CPU + one lane per queue) with semaphore and
 * fence arrows, a scrubber over time, and the problems the rules find (data races, VUIDs, deadlocks). The reader edits
 * waits, stage masks, signals, fences and the order of the calls; everything is recomputed by lib/labs/queue-sync.ts.
 */
import { useMemo, useState } from 'react';
import { LabFrame } from 'course-kit/components/lab/LabFrame.tsx';
import { Scrubber } from 'course-kit/components/lab/Scrubber.tsx';
import { useTimeline } from 'course-kit/components/lab/useTimeline.ts';
import { fmt } from 'course-kit/i18n.mjs';
import { pick } from '../../lib/text';
import {
	MASKS,
	MAX_DUR,
	MIN_DUR,
	PRESETS,
	PRESET_IDS,
	QUEUES,
	STAGES,
	analyze,
	frames,
	queue,
	queuesFor,
	raceSpans,
	stageName,
	type Issue,
	type Lane,
	type Mask,
	type PresetId,
	type QueueId,
	type Run,
	type Scenario,
	type SimEvent,
	type SubmitOp,
} from '../../lib/labs/queue-sync';
import { TEXT } from './queue-sync.text';
import './queue-sync.css';

type Text = (typeof TEXT)['en'];

const LANES: Lane[] = ['cpu', ...QUEUES.map((q) => q.id)];
const PRESET_KEY: Record<PresetId, keyof Text> = { upload: 'p_upload', race: 'p_race', async: 'p_async', frames: 'p_frames' };

/** Chart geometry, px: time unit, label column, row height, gap between lanes, axis height. */
const U = 30;
const LABEL_W = 156;
const ROW = 26;
const GAP = 10;
const AXIS = 22;

/** Display name of a submitted batch in the reader's language (`frame 2`). */
function who(t: Text, op: SubmitOp): string {
	return fmt(t[`w_${op.name}` as keyof Text], { n: op.n ?? '' });
}

/** One line per event of a time step, in the reader's language. */
function eventText(t: Text, sc: Scenario, e: SimEvent): string {
	const op = sc.ops[e.op]!;
	const w = op.kind === 'submit' ? who(t, op) : '';
	switch (e.kind) {
		case 'submit':
			return fmt(t.ev_submit, { who: w, queue: (op as SubmitOp).queue });
		case 'start':
			return fmt(t.ev_start, { who: w });
		case 'done':
			return fmt(t.ev_done, { who: w });
		case 'signal':
			return fmt(e.value === undefined ? t.ev_signal : t.ev_signalT, { who: w, sem: e.sem, value: e.value ?? '' });
		case 'waitDone':
			return fmt(e.value === undefined ? t.ev_waitDone : t.ev_waitDoneT, { who: w, sem: e.sem, value: e.value ?? '' });
		case 'blocked':
			return fmt(t.ev_blocked, { who: w, sem: e.sem, stage: e.stage });
		case 'fenceSignal':
			return fmt(t.ev_fence, { who: w, fence: e.fence });
		case 'cpuWait':
			return fmt(t.ev_cpuWait, { fence: e.fence });
		case 'cpuWaitDone':
			return fmt(t.ev_cpuWaitDone, { fence: e.fence });
	}
}

/** The text of a problem, in the reader's language; VUID texts stay verbatim (English, from the registry). */
function issueText(t: Text, sc: Scenario, run: Run, i: Issue): string {
	const op = sc.ops[i.op]!;
	const w = op.kind === 'submit' ? who(t, op) : '';
	if (i.kind !== 'race') return fmt(t[`i_${i.kind}` as keyof Text], { ...i.vars, who: w });
	const side = (opIdx: number, acc: string, stage: string) => {
		const o = sc.ops[opIdx] as SubmitOp;
		const where = stage === 'HOST' ? t.onHost : fmt(t.atStage, { stage });
		return `${who(t, o)} ${acc === 'write' ? t.acc_write : t.acc_read} ${where}`;
	};
	const a = side(i.others[0]!, String(i.vars.a), String(i.vars.aStage));
	const b = side(i.op, String(i.vars.b), String(i.vars.bStage));
	const sp = raceSpans(run, i);
	const overlap = sp.a && sp.b && sp.a[0] < sp.b[1] && sp.b[0] < sp.a[1];
	return `${fmt(t.i_race, { res: String(i.vars.res), a, b })} ${overlap ? t.raceOverlap : t.raceApart}`;
}

export default function QueueSyncView({ lang }: { lang: string }) {
	const t = pick(TEXT, lang, 'queue-sync');
	const [preset, setPreset] = useState<PresetId>('upload');
	const [sc, setSc] = useState<Scenario>(() => structuredClone(PRESETS.upload));
	const [hot, setHot] = useState<number | null>(null);

	const a = useMemo(() => analyze(sc), [sc]);
	const fr = useMemo(() => frames(a.run), [a]);
	const tl = useTimeline(fr);
	const now = tl.frame.t;

	const load = (id: PresetId) => {
		setPreset(id);
		setSc(structuredClone(PRESETS[id]));
		setHot(null);
		tl.reset();
	};
	/** Immutable edit: `f` changes a copy of the scenario. */
	const edit = (f: (s: Scenario) => void) => {
		setSc((prev) => {
			const next = structuredClone(prev);
			f(next);
			return next;
		});
		setHot(null);
	};

	// Lane layout: y of each lane's first row, from the number of overlapping batches in it.
	const laneY = useMemo(() => {
		const ys = {} as Record<Lane, number>;
		let y = AXIS;
		for (const l of LANES) {
			ys[l] = y;
			y += a.laneRows[l] * ROW + GAP;
		}
		return { ys, height: y };
	}, [a]);
	const rowY = (lane: Lane, op: number) => laneY.ys[lane] + (lane === 'cpu' ? 0 : (a.rows[op] ?? 0)) * ROW;
	const x = (time: number) => LABEL_W + time * U;
	const width = x(a.run.end) + U;

	const hotOps = new Set(hot === null ? [] : [a.issues[hot]!.op, ...a.issues[hot]!.others]);
	const issueOps = new Set(a.issues.flatMap((i) => [i.op, ...i.others]));
	const caption = (
		<>
			<b>{fmt(t.time, { t: now })}</b>: {tl.frame.events.length ? tl.frame.events.map((e) => eventText(t, sc, e)).join('; ') : t.nothing}
		</>
	);

	const laneLabel = (l: Lane) => (l === 'cpu' ? t.lane_cpu : fmt(t[`lane_${l}` as keyof Text], { f: queue(l).family }));

	return (
		<LabFrame lang={lang} title={t.title} onReset={() => load(preset)}>
			<div className="qs">
			<p className="qs-question">{t.question}</p>
			<div className="lab-row qs-presets" role="group" aria-label={t.scenario}>
				{PRESET_IDS.map((id) => (
					<button key={id} type="button" className={`lab-button${id === preset ? ' is-primary' : ''}`} aria-pressed={id === preset} onClick={() => load(id)}>
						{t[PRESET_KEY[id]]}
					</button>
				))}
			</div>

			<div className="qs-chart">
				<svg width={width} height={laneY.height + 4} viewBox={`0 0 ${width} ${laneY.height + 4}`} role="img" aria-label={t.chart}>
					<defs>
						<pattern id="qs-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
							<line x1="0" y1="0" x2="0" y2="6" className="qs-hatch-line" />
						</pattern>
						<marker id="qs-arrow-sem" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
							<path d="M0,0 L8,4 L0,8 z" className="qs-arrowhead is-sem" />
						</marker>
						<marker id="qs-arrow-fence" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
							<path d="M0,0 L8,4 L0,8 z" className="qs-arrowhead is-fence" />
						</marker>
					</defs>

					{/* Time axis */}
					{Array.from({ length: a.run.end + 1 }, (_, i) => (
						<g key={i}>
							<line x1={x(i)} x2={x(i)} y1={AXIS - 4} y2={laneY.height - GAP} className="qs-grid" />
							{(a.run.end <= 24 || i % 2 === 0) && (
								<text x={x(i)} y={AXIS - 8} className="qs-tick">
									{i}
								</text>
							)}
						</g>
					))}
					<text x={4} y={AXIS - 8} className="qs-tick is-axis">
						{t.axis}
					</text>

					{/* Lanes */}
					{LANES.map((l) => (
						<g key={l}>
							<rect x={0} y={laneY.ys[l] - 2} width={width} height={a.laneRows[l] * ROW + 4} className="qs-lane" />
							<text x={6} y={laneY.ys[l] + ROW / 2 + 4} className="qs-lane-label">
								{laneLabel(l)}
							</text>
						</g>
					))}

					{/* Segments: past solid, future faint */}
					{a.run.segs.flatMap((s, k) => {
						const y = rowY(s.lane, s.op) + 3;
						const parts: [number, number, boolean][] = [];
						if (s.from < now) parts.push([s.from, Math.min(s.to, now), false]);
						if (s.to > now) parts.push([Math.max(s.from, now), s.to, true]);
						const op = sc.ops[s.op]!;
						const tip = op.kind === 'submit' ? `${who(t, op)} · ${s.what}` : `${t.waitOp} · ${op.fence}`;
						return parts.map(([from, to, future], m) => (
							<rect
								key={`${k}-${m}`}
								x={x(from)}
								y={y}
								width={(to - from) * U}
								height={ROW - 6}
								rx={3}
								className={`qs-seg is-${s.what}${future ? ' is-future' : ''}${hotOps.has(s.op) ? ' is-hot' : ''}`}
								fill={s.what === 'stall' ? 'url(#qs-hatch)' : undefined}
							>
								<title>{tip}</title>
							</rect>
						));
					})}

					{/* Batch names, at the submit time */}
					{sc.ops.map((op, i) => {
						const r = a.run.batches[i];
						if (op.kind !== 'submit' || !r || r.submitted === null) return null;
						return (
							<text key={i} x={x(r.submitted) + 4} y={rowY(op.queue, i) + ROW / 2 + 4} className={`qs-name${r.submitted >= now ? ' is-future' : ''}`}>
								{who(t, op)}
							</text>
						);
					})}

					{/* Semaphore and fence arrows: signal point -> wait point */}
					{a.run.links.map((l, k) => {
						const fromOp = sc.ops[l.from.op] as SubmitOp;
						const x1 = x(l.from.t);
						const y1 = rowY(fromOp.queue, l.from.op) + ROW / 2;
						const toOp = sc.ops[l.to.op]!;
						const x2 = x(l.to.t);
						const y2 = l.kind === 'fence' ? rowY('cpu', l.to.op) + ROW / 2 : rowY((toOp as SubmitOp).queue, l.to.op) + ROW / 2;
						const bend = Math.max(14, Math.abs(y2 - y1) / 2);
						const d = `M${x1},${y1} C${x1 + bend},${y1} ${x2 - bend},${y2} ${x2},${y2}`;
						return (
							<g key={k} className={`qs-link is-${l.kind}${l.to.t > now ? ' is-future' : ''}`}>
								<path d={d} markerEnd={`url(#qs-arrow-${l.kind})`} />
								<circle cx={x1} cy={y1} r={3.5} />
								<title>{`${l.name}: t ${l.from.t} → ${l.to.t}`}</title>
							</g>
						);
					})}

					{/* Problem markers on the operations involved */}
					{sc.ops.map((op, i) => {
						if (!issueOps.has(i)) return null;
						const r = a.run.batches[i];
						const lane: Lane = op.kind === 'submit' ? op.queue : 'cpu';
						const at = op.kind === 'submit' ? (r?.submitted ?? null) : (a.run.segs.find((s) => s.lane === 'cpu' && s.op === i)?.from ?? a.run.events.find((e) => e.op === i && e.kind === 'cpuWait')?.t ?? null);
						if (at === null) return null;
						return (
							<g key={i} className="qs-mark" transform={`translate(${x(at) - 2},${rowY(lane, i) + 2})`}>
								<circle r={7} />
								<text y={4}>!</text>
							</g>
						);
					})}

					<line x1={x(now)} x2={x(now)} y1={AXIS - 4} y2={laneY.height - GAP + 2} className="qs-cursor" />
				</svg>
			</div>

			<Scrubber lang={lang} tl={tl} caption={caption} />

			<ul className="qs-legend" aria-label="Legend">
				{STAGES.map((s) => (
					<li key={s}>
						<span className={`qs-swatch is-${s}`} aria-hidden="true" />
						{s}
					</li>
				))}
				<li>
					<span className="qs-swatch is-pending" aria-hidden="true" />
					{t.legend_pending}
				</li>
				<li>
					<span className="qs-swatch is-stall" aria-hidden="true" />
					{t.legend_stall}
				</li>
				<li>
					<span className="qs-swatch is-record" aria-hidden="true" />
					{t.legend_record}
				</li>
				<li>
					<span className="qs-swatch is-wait" aria-hidden="true" />
					{t.legend_wait}
				</li>
				<li>
					<span className="qs-swatch is-line-sem" aria-hidden="true" />
					{t.legend_sem}
				</li>
				<li>
					<span className="qs-swatch is-line-fence" aria-hidden="true" />
					{t.legend_fence}
				</li>
			</ul>

			<section className="qs-issues" aria-live="polite">
				{a.issues.length === 0 ? (
					<p className="qs-ok">{t.noIssues}</p>
				) : (
					<>
						<p className="qs-issues-head">{fmt(t.issues, { n: a.issues.length })}</p>
						<ol>
							{a.issues.map((i, k) => (
								<li key={k} className={hot === k ? 'is-hot' : undefined}>
									<button type="button" className="qs-issue" aria-pressed={hot === k} onClick={() => setHot(hot === k ? null : k)}>
										{issueText(t, sc, a.run, i)}
									</button>
									{i.vuid ? (
										<p className="qs-vuid">
											<code>{i.vuid}</code>: {i.vuidText}
										</p>
									) : (
										<p className="qs-vuid is-none">{i.kind === 'race' ? t.noVuidRace : t.noVuidHang}</p>
									)}
								</li>
							))}
						</ol>
					</>
				)}
			</section>

			<Program t={t} sc={sc} edit={edit} issueOps={issueOps} />

			<p className="qs-note lab-muted">{t.note}</p>
			</div>
		</LabFrame>
	);
}

/** The editable host program: semaphores, fences, and the calls in order. */
function Program({ t, sc, edit, issueOps }: { t: Text; sc: Scenario; edit: (f: (s: Scenario) => void) => void; issueOps: Set<number> }) {
	const timeline = (sem: string) => sc.sems.find((s) => s.id === sem)?.type === 'timeline';
	const move = (i: number, d: number) =>
		edit((s) => {
			const [op] = s.ops.splice(i, 1);
			s.ops.splice(i + d, 0, op!);
		});
	const sub = (s: Scenario, i: number) => s.ops[i] as SubmitOp;
	const nextValue = (sem: string) => 1 + Math.max(0, ...sc.ops.flatMap((o) => (o.kind === 'submit' ? o.signals.filter((x) => x.sem === sem).map((x) => x.value) : [])));

	return (
		<section className="qs-program">
			<p className="qs-program-head">{t.program}</p>
			<div className="lab-row qs-objects">
				{sc.sems.length > 0 && (
					<span className="qs-objects-group">
						{t.sems}:
						{sc.sems.map((s, k) => (
							<label key={s.id} className="qs-inline">
								<code>{s.id}</code>
								<select value={s.type} onChange={(e) => edit((x) => (x.sems[k]!.type = e.currentTarget.value as 'binary' | 'timeline'))}>
									<option value="binary">{t.binary}</option>
									<option value="timeline">{t.timeline}</option>
								</select>
							</label>
						))}
					</span>
				)}
				{sc.fences.length > 0 && (
					<span className="qs-objects-group">
						{t.fences}:
						{sc.fences.map((f, k) => (
							<label key={f.id} className="qs-inline">
								<input type="checkbox" checked={f.signaled} onChange={(e) => edit((x) => (x.fences[k]!.signaled = e.currentTarget.checked))} />
								<code>{f.id}</code> {t.createdSignaled}
							</label>
						))}
					</span>
				)}
			</div>
			<ol className="qs-ops">
				{sc.ops.map((op, i) => (
					<li key={i} className={`qs-op${issueOps.has(i) ? ' has-issue' : ''}`}>
						<div className="qs-op-head">
							<span className="qs-op-index">{i + 1}</span>
							{op.kind === 'wait' ? (
								<span>
									<code>{t.waitOp}</code>{' '}
									<select aria-label={t.fence} value={op.fence} onChange={(e) => edit((s) => ((s.ops[i] as { fence: string }).fence = e.currentTarget.value))}>
										{sc.fences.map((f) => (
											<option key={f.id} value={f.id}>
												{f.id}
											</option>
										))}
									</select>
								</span>
							) : (
								<span>
									<code>vkQueueSubmit2</code> <b>{who(t, op)}</b> <span className="lab-muted">{fmt(t.records, { cb: op.cb })}</span>
								</span>
							)}
							<span className="qs-op-tools">
								<button type="button" className="lab-button" aria-label={t.up} title={t.up} aria-disabled={i === 0} onClick={() => i > 0 && move(i, -1)}>
									↑
								</button>
								<button type="button" className="lab-button" aria-label={t.down} title={t.down} aria-disabled={i === sc.ops.length - 1} onClick={() => i < sc.ops.length - 1 && move(i, 1)}>
									↓
								</button>
								<button type="button" className="lab-button is-danger" aria-label={t.remove} title={t.remove} onClick={() => edit((s) => s.ops.splice(i, 1))}>
									✕
								</button>
							</span>
						</div>
						{op.kind === 'submit' && (
							<div className="qs-op-body">
								<label className="qs-inline">
									{t.queue}
									<select value={op.queue} onChange={(e) => edit((s) => (sub(s, i).queue = e.currentTarget.value as QueueId))}>
										{queuesFor(op.work).map((q) => (
											<option key={q} value={q}>
												{q}
											</option>
										))}
									</select>
								</label>
								<label className="qs-inline">
									{t.dur}
									<input
										type="number"
										min={MIN_DUR[op.work]}
										max={MAX_DUR}
										value={op.dur}
										onChange={(e) => {
											const v = Math.round(Number(e.currentTarget.value));
											if (Number.isFinite(v)) edit((s) => (sub(s, i).dur = Math.min(MAX_DUR, Math.max(MIN_DUR[op.work], v))));
										}}
									/>
								</label>
								<label className="qs-inline">
									{t.fence}
									<select value={op.fence ?? ''} onChange={(e) => edit((s) => (sub(s, i).fence = e.currentTarget.value || undefined))}>
										<option value="">{t.none}</option>
										{sc.fences.map((f) => (
											<option key={f.id} value={f.id}>
												{f.id}
											</option>
										))}
									</select>
								</label>
								<div className="qs-sync">
									<span className="qs-sync-head">{t.waits}:</span>
									{op.waits.map((w, k) => (
										<span key={k} className="qs-chip">
											<select aria-label={t.sems} value={w.sem} onChange={(e) => edit((s) => (sub(s, i).waits[k]!.sem = e.currentTarget.value))}>
												{sc.sems.map((s) => (
													<option key={s.id} value={s.id}>
														{s.id}
													</option>
												))}
											</select>
											{timeline(w.sem) && (
												<label className="qs-inline">
													≥
													<input type="number" min={1} max={99} aria-label={t.value} value={w.value} onChange={(e) => edit((s) => (sub(s, i).waits[k]!.value = Math.max(1, Math.round(Number(e.currentTarget.value)) || 1)))} />
												</label>
											)}
											<select aria-label="stageMask" value={w.stage} onChange={(e) => edit((s) => (sub(s, i).waits[k]!.stage = e.currentTarget.value as Mask))}>
												{MASKS.map((m) => (
													<option key={m} value={m}>
														{m}
													</option>
												))}
											</select>
											<button type="button" className="qs-chip-x" aria-label={fmt(t.removeWait, { sem: w.sem })} title={fmt(t.removeWait, { sem: w.sem })} onClick={() => edit((s) => sub(s, i).waits.splice(k, 1))}>
												✕
											</button>
										</span>
									))}
									{sc.sems.length > 0 ? (
										<button type="button" className="lab-button qs-add" onClick={() => edit((s) => sub(s, i).waits.push({ sem: s.sems[0]!.id, value: 1, stage: 'ALL_COMMANDS' }))}>
											{t.addWait}
										</button>
									) : (
										<span className="lab-muted">{t.noSems}</span>
									)}
								</div>
								<div className="qs-sync">
									<span className="qs-sync-head">{t.signals}:</span>
									{op.signals.map((g, k) => (
										<span key={k} className="qs-chip">
											<select aria-label={t.sems} value={g.sem} onChange={(e) => edit((s) => (sub(s, i).signals[k]!.sem = e.currentTarget.value))}>
												{sc.sems.map((s) => (
													<option key={s.id} value={s.id}>
														{s.id}
													</option>
												))}
											</select>
											{timeline(g.sem) && (
												<label className="qs-inline">
													=
													<input type="number" min={1} max={99} aria-label={t.value} value={g.value} onChange={(e) => edit((s) => (sub(s, i).signals[k]!.value = Math.max(1, Math.round(Number(e.currentTarget.value)) || 1)))} />
												</label>
											)}
											<button type="button" className="qs-chip-x" aria-label={fmt(t.removeSignal, { sem: g.sem })} title={fmt(t.removeSignal, { sem: g.sem })} onClick={() => edit((s) => sub(s, i).signals.splice(k, 1))}>
												✕
											</button>
										</span>
									))}
									{sc.sems.length > 0 && (
										<button type="button" className="lab-button qs-add" onClick={() => edit((s) => sub(s, i).signals.push({ sem: s.sems[0]!.id, value: nextValue(s.sems[0]!.id) }))}>
											{t.addSignal}
										</button>
									)}
								</div>
								<p className="qs-uses lab-muted">
									{[
										...op.hostWrites.map((res) => fmt(t.hostWrites, { res })),
										...op.uses.map((u) => fmt(u.write ? t.writes : t.reads, { res: u.res, stage: stageName(u.stage) })),
									].join(' · ')}
								</p>
							</div>
						)}
					</li>
				))}
			</ol>
			{sc.fences.length > 0 && (
				<button type="button" className="lab-button qs-add" onClick={() => edit((s) => s.ops.push({ kind: 'wait', fence: s.fences[0]!.id }))}>
					{t.addFenceWait}
				</button>
			)}
		</section>
	);
}
