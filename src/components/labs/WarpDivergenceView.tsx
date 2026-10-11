/**
 * Warp-divergence lab (React island): 32 lanes of one warp run a small if/else shader, one instruction per issue
 * slot for the whole warp. The reader picks the inputs and the condition; the timeline shows which lanes are lit
 * (execute) and which are masked in every slot, and the cost against a warp whose lanes agree.
 * Model: lib/labs/warp-divergence.ts.
 */
import { useMemo, useState } from 'react';
import { LabFrame } from 'course-kit/components/lab/LabFrame.tsx';
import { Scrubber } from 'course-kit/components/lab/Scrubber.tsx';
import { useTimeline } from 'course-kit/components/lab/useTimeline.ts';
import { pick, tfmt } from '../../lib/text';
import { CONDS, MAX_X, PROGRAM, WARP, condSrc, inputs, run, type Block, type Cond, type Inputs, type WarpFrame } from '../../lib/labs/warp-divergence';
import { TEXT } from './warp-divergence.text';
import './warp-divergence.css';

type Text = (typeof TEXT)['en'];

const COND_KEY: Record<Cond, keyof Text> = { lt: 'condLt', even: 'condEven', never: 'condNever' };

/** Listing rows: instructions of the program plus the brace lines that are not instructions. */
type Row = { block: Block; line: number } | { brace: string };
const ROWS: Row[] = [
	...PROGRAM.pre.map((_, line) => ({ block: 'pre' as const, line })),
	{ brace: 'if (c) {' },
	...PROGRAM.if.map((_, line) => ({ block: 'if' as const, line })),
	{ brace: '} else {' },
	...PROGRAM.else.map((_, line) => ({ block: 'else' as const, line })),
	{ brace: '}' },
	...PROGRAM.post.map((_, line) => ({ block: 'post' as const, line })),
];

export default function WarpDivergenceView({ lang }: { lang: string }) {
	const t = pick(TEXT, lang, 'warp-divergence');
	const [kind, setKind] = useState<Inputs>('lane');
	const [seed, setSeed] = useState(1);
	const [cond, setCond] = useState<Cond>('lt');
	const [thr, setThr] = useState(16);

	const xs = useMemo(() => inputs(kind, seed), [kind, seed]);
	const r = useMemo(() => run(xs, cond, thr), [xs, cond, thr]);
	const tl = useTimeline<WarpFrame>(r.frames, { ms: 700 });
	const f = tl.frame;
	const src = condSrc(cond, thr);
	const code = (block: Block, line: number) => PROGRAM[block][line]!.replace('{cond}', src);
	const path = (taken: boolean) => (taken ? t.pathIf : t.pathElse);
	const nIf = r.taken.filter(Boolean).length;
	const nActive = f.active.filter(Boolean).length;

	const caption = (() => {
		if (!f.at) return tfmt(lang, t.capStart, { nIf, nElse: r.taken.length - nIf });
		const vars = { cycle: f.cycle, code: code(f.at.block, f.at.line) };
		const now =
			f.at.block === 'pre'
				? tfmt(lang, t.capPre, vars)
				: f.at.block === 'post'
					? tfmt(lang, t.capPost, vars)
					: tfmt(lang, t.capBranch, { ...vars, path: path(f.at.block === 'if'), n: nActive, off: r.taken.length - nActive });
		if (tl.index < tl.count - 1) return now;
		const sum = r.divergent
			? tfmt(lang, t.sumDivergent, { cycles: r.cycles, eff: Math.round(r.efficiency * 100), allIf: r.allIf, allElse: r.allElse })
			: tfmt(lang, t.sumUniform, { cycles: r.cycles, path: path(nIf > 0) });
		return `${now} ${sum}`;
	})();

	const reset = () => {
		setKind('lane');
		setSeed(1);
		setCond('lt');
		setThr(16);
		tl.reset();
	};

	return (
		<LabFrame lang={lang} title={t.title} onReset={reset}>
			<div className="wd-controls">
				<label className="lab-field">
					<span>{t.inputs}</span>
					<select value={kind} onChange={(e) => setKind(e.currentTarget.value as Inputs)}>
						<option value="lane">{t.inLane}</option>
						<option value="random">{t.inRandom}</option>
					</select>
				</label>
				<label className="lab-field">
					<span>{t.cond}</span>
					<select value={cond} onChange={(e) => setCond(e.currentTarget.value as Cond)}>
						{CONDS.map((c) => (
							<option key={c} value={c}>
								{t[COND_KEY[c]]}
							</option>
						))}
					</select>
				</label>
				<label className="lab-field wd-thr">
					<span>{tfmt(lang, t.threshold, { t: thr })}</span>
					<input
						type="range"
						min={0}
						max={MAX_X + 1}
						step={1}
						value={thr}
						disabled={cond !== 'lt'}
						onChange={(e) => setThr(Number(e.currentTarget.value))}
					/>
				</label>
				{kind === 'random' && (
					<button type="button" className="lab-button wd-shuffle" onClick={() => setSeed((s) => s + 1)}>
						{t.shuffle}
					</button>
				)}
			</div>

			<div className="wd-main">
				<figure className="wd-code">
					<figcaption className="lab-muted">{t.program}</figcaption>
					<pre>
						{ROWS.map((row, i) => {
							if ('brace' in row) {
								return (
									<span key={i} className="wd-line is-brace">
										{row.brace}
									</span>
								);
							}
							const cur = f.at?.block === row.block && f.at.line === row.line;
							const nested = row.block === 'if' || row.block === 'else';
							return (
								<span key={i} className={`wd-line is-${row.block}${cur ? ' is-cur' : ''}`} aria-current={cur ? 'step' : undefined}>
									{nested ? '    ' : ''}
									{code(row.block, row.line)}
								</span>
							);
						})}
					</pre>
				</figure>

				<div className="wd-warp">
					<p className="lab-muted">{t.lanes}</p>
					<ol className="wd-lanes">
						{xs.map((x, lane) => {
							const tip = tfmt(lang, t.laneTip, { lane, x, path: path(r.taken[lane]!) });
							return (
								<li
									key={lane}
									className={`wd-lane ${r.taken[lane] ? 'is-if' : 'is-else'}${f.active[lane] ? ' is-on' : ''}`}
									title={tip}
									aria-label={tip}
								>
									<b>{x}</b>
								</li>
							);
						})}
					</ol>
					<p className="wd-legend lab-muted">
						<span className="wd-key is-if" aria-hidden="true" /> {t.legendIf} <span className="wd-key is-else" aria-hidden="true" /> {t.legendElse}
						<br />
						{t.legendOff}
					</p>
					<div className="lab-stats">
						<div className="lab-stat">
							<b>{f.cycle}</b>
							<span>{t.cycles}</span>
						</div>
						<div className="lab-stat">
							<b>{nActive}/{WARP}</b>
							<span>{t.active}</span>
						</div>
						<div className="lab-stat">
							<b>{f.cycle ? Math.round((f.busy / (WARP * f.cycle)) * 100) : 0}%</b>
							<span>{t.useful}</span>
						</div>
					</div>
				</div>
			</div>

			<Scrubber lang={lang} tl={tl} caption={caption} />
			<p className="lab-muted">{t.note}</p>
		</LabFrame>
	);
}
