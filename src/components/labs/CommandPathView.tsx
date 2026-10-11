/**
 * Command-path lab (React island): the render loop of lib/labs/command-path.ts on three lanes (CPU, queue, GPU),
 * stepped event by event. The reader picks when the CPU waits on the frame's fence and sees the bars, the idle
 * gaps and the utilization change. Both modes share one time scale, so the shorter run is visibly shorter.
 */
import { useMemo, useState } from 'react';
import { LabFrame } from 'course-kit/components/lab/LabFrame.tsx';
import { Scrubber } from 'course-kit/components/lab/Scrubber.tsx';
import { useTimeline } from 'course-kit/components/lab/useTimeline.ts';
import { FRAMES, percent, simulate, type CpEvent, type InFlight, type Lane, type Schedule } from '../../lib/labs/command-path';
import { pick, tfmt } from '../../lib/text';
import { TEXT } from './command-path.text';
import './command-path.css';

type T = (typeof TEXT)['en'];

const LANES: readonly Lane[] = ['cpu', 'queue', 'gpu'];
const LANE_KEY = { cpu: 'laneCpu', queue: 'laneQueue', gpu: 'laneGpu' } as const;
const RUNS: Record<InFlight, Schedule> = { 1: simulate(1), 2: simulate(2) };
/** Common scale of both modes: the longer run fills the width. */
const SCALE = Math.max(RUNS[1].end, RUNS[2].end);
const WORK = ['update', 'record', 'submit'] as const;

/** The caption of an event in the reader's language. */
function caption(lang: string, t: T, s: Schedule, e: CpEvent): string {
	const v = { f: e.f + 1, cb: e.cb, g: e.g + 1 };
	switch (e.kind) {
		case 'start':
			return t.capStart;
		case 'update':
			return tfmt(lang, t.capUpdate, v);
		case 'record':
			return tfmt(lang, t.capRecord, v);
		case 'recorded':
			return tfmt(lang, t.capRecorded, v);
		case 'queued':
			return tfmt(lang, t.capQueued, v);
		case 'gpuStart':
			return tfmt(lang, t.capGpuStart, v);
		case 'gpuEnd':
			return tfmt(lang, t.capGpuEnd, v);
		case 'waitStart':
			return tfmt(lang, s.inFlight === 1 ? t.capWaitStart1 : t.capWaitStart2, v);
		case 'waitEnd':
			return tfmt(lang, s.inFlight === 1 ? t.capWaitEnd1 : t.capWaitEnd2, v);
		case 'waitFree':
			return tfmt(lang, e.g < 0 ? t.capWaitNew : t.capWaitDone, v);
		case 'done':
			return tfmt(lang, s.inFlight === 1 ? t.capDone1 : t.capDone2, { n: FRAMES, t: s.end, gpu: percent(s, ['exec']), wait: percent(s, ['wait']) });
	}
}

const pct = (x: number) => `${(100 * x) / SCALE}%`;

export default function CommandPathView({ lang }: { lang: string }) {
	const t = pick(TEXT, lang, 'command-path');
	const [mode, setMode] = useState<InFlight>(1);
	const run = RUNS[mode];
	const tl = useTimeline(run.events);
	const now = tl.frame.t;
	const ticks = useMemo(() => Array.from({ length: Math.floor(SCALE / 5) + 1 }, (_, i) => i * 5), []);

	// Switching modes keeps the moment: jump to the last event of the other run at or before the current tick.
	const choose = (m: InFlight) => {
		const evs = RUNS[m].events;
		const i = evs.findLastIndex((e) => e.t <= now);
		setMode(m);
		tl.seek(Math.max(0, i));
	};

	return (
		<LabFrame lang={lang} title={t.title} onReset={tl.reset}>
			<fieldset className="cp-mode">
				<legend>{t.mode}</legend>
				{([1, 2] as const).map((m) => (
					<label key={m}>
						<input type="radio" name={`cp-mode-${lang}`} checked={mode === m} onChange={() => choose(m)} />
						{m === 1 ? t.mode1 : t.mode2}
					</label>
				))}
			</fieldset>

			<div className="cp-chart" role="img" aria-label={tfmt(lang, t.lanes, { t: now, end: run.end })}>
				{LANES.map((lane) => (
					<div className="cp-row" key={lane}>
						<span className="cp-lane">{t[LANE_KEY[lane]]}</span>
						<div className="cp-track">
							{run.segments
								.filter((s) => s.lane === lane && s.t0 < now)
								.map((s) => {
									const label = tfmt(lang, t.bar, { f: s.f + 1, what: t[s.kind], t0: s.t0, t1: s.t1 });
									return (
										<span
											key={`${s.kind}-${s.f}`}
											className={`cp-bar is-${s.kind} is-f${s.f % 2}`}
											style={{ left: pct(s.t0), width: pct(Math.min(s.t1, now) - s.t0) }}
											title={label}
										>
											{s.kind === 'exec' || s.kind === 'record' ? `F${s.f + 1}` : ''}
										</span>
									);
								})}
							<span className="cp-now" style={{ left: pct(now) }} aria-hidden="true" />
						</div>
					</div>
				))}
				<div className="cp-row cp-axis" aria-hidden="true">
					<span className="cp-lane" />
					<div className="cp-track">
						{ticks.map((x) => (
							<span key={x} className="cp-tick" style={{ left: pct(x) }}>
								{x}
							</span>
						))}
					</div>
				</div>
			</div>

			<ul className="cp-legend" aria-hidden="true">
				{(['update', 'record', 'submit', 'wait', 'queued', 'exec'] as const).map((k) => (
					<li key={k}>
						<span className={`cp-swatch is-${k}`} />
						{t[k]}
					</li>
				))}
			</ul>

			<Scrubber lang={lang} tl={tl} caption={caption(lang, t, run, tl.frame)} />

			<div className="lab-stats">
				<div className="lab-stat">
					<b>{percent(run, WORK)}%</b>
					<span>{t.statCpu}</span>
				</div>
				<div className="lab-stat">
					<b>{percent(run, ['wait'])}%</b>
					<span>{t.statWait}</span>
				</div>
				<div className="lab-stat">
					<b>{percent(run, ['exec'])}%</b>
					<span>{t.statGpu}</span>
				</div>
				<div className="lab-stat">
					<b>{run.end}</b>
					<span>{tfmt(lang, t.statTime, { n: FRAMES })}</span>
				</div>
			</div>
			<p className="lab-muted">{t.illustrative}</p>
		</LabFrame>
	);
}
