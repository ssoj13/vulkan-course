/**
 * GL-vs-Vulkan lab (React island): the reader edits the draws of a frame; the model (lib/labs/gl-vs-vulkan.ts)
 * plays load time, frame 1 draw by draw and frame 2 on a GL-style driver and on Vulkan side by side, with the calls
 * each side makes, the draws where GL may wait for a shader compile, and the pipelines Vulkan built up front.
 */
import { useMemo, useState } from 'react';
import { LabFrame } from 'course-kit/components/lab/LabFrame.tsx';
import { Scrubber } from 'course-kit/components/lab/Scrubber.tsx';
import { useTimeline } from 'course-kit/components/lab/useTimeline.ts';
import { fmt } from 'course-kit/i18n.mjs';
import {
	DEPTH_DYNAMIC,
	MAX_DRAWS,
	SCENE,
	SHADERS,
	VERTEX_FORMATS,
	dynamicByVersion,
	simulate,
	type Draw,
	type DrawState,
	type Result,
	type Step,
} from '../../lib/labs/gl-vs-vulkan';
import { pick } from '../../lib/text';
import { TEXT } from './gl-vs-vulkan.text';
import './gl-vs-vulkan.css';

type T = (typeof TEXT)['en'];

const DYNAMIC = dynamicByVersion();

/** "lit shader, blending, mesh vertices" in the reader's language. */
function combo(t: T, d: DrawState): string {
	return fmt(t.combo, { shader: t[`sh_${d.shader}` as const], blend: d.blend ? t.blendOn : t.blendOff, vertex: t[`vx_${d.vertex}` as const] });
}

/** A pipeline key ("lit|blend|mesh|depth") in words, for the pipeline list. */
function pipeLabel(t: T, key: string): string {
	const [shader, blend, vertex, depth] = key.split('|') as [Draw['shader'], string, Draw['vertex'], string | undefined];
	const base = combo(t, { shader, blend: blend === 'blend', vertex, depth: false });
	return depth ? `${base}, ${t.colDepth.toLowerCase()} ${depth === 'depth' ? t.on : t.off}` : base;
}

function caption(t: T, draws: readonly Draw[], r: Result, s: Step, dynDepth: boolean): string {
	if (s.kind === 'load') return fmt(t.capLoad, { programs: r.programs.length, pipelines: r.pipelines.length });
	if (s.kind === 'frame2') return t.capFrame2;
	const d = draws[s.i]!;
	const gl = fmt(s.gl.compile ? t.capGlNew : t.capGlSeen, { i: s.i + 1, name: t[`n_${d.name}` as const], combo: combo(t, d) });
	const bound = s.vk.calls.some((c) => c.startsWith('vkCmdBindPipeline'));
	const vk = fmt(bound ? t.capVkBind : t.capVkKeep, { p: s.pipeline });
	const dyn = dynDepth && s.i > 0 && s.vk.calls.some((c) => c.startsWith('vkCmdSetDepthTestEnable')) ? ` ${t.capVkDyn}` : '';
	return `${gl} ${vk}${dyn}`;
}

export default function GlVsVulkanView({ lang }: { lang: string }) {
	const t = pick(TEXT, lang, 'gl-vs-vulkan');
	const [draws, setDraws] = useState<Draw[]>(() => [...SCENE]);
	const [dynDepth, setDynDepth] = useState(true);
	const r = useMemo(() => simulate(draws, dynDepth), [draws, dynDepth]);
	const tl = useTimeline(r.steps);
	const s = tl.frame;
	/** Draws of frame 1 already played at this step (all of them on the frame-2 step). */
	const played = s.kind === 'load' ? -1 : s.kind === 'frame2' ? draws.length : s.i;

	const edit = (i: number, patch: Partial<DrawState>) => setDraws((ds) => ds.map((d, j) => (j === i ? { ...d, ...patch } : d)));
	const add = () => setDraws((ds) => (ds.length < MAX_DRAWS ? [...ds, { ...ds[ds.length - 1]!, name: 'extra' }] : ds));
	const remove = (i: number) => setDraws((ds) => (ds.length > 1 ? ds.filter((_, j) => j !== i) : ds));
	const reset = () => {
		setDraws([...SCENE]);
		setDynDepth(true);
		tl.reset();
	};

	const chips = (side: 'gl' | 'vk') => (
		<ol className="gv-chips">
			{draws.map((d, i) => {
				const step = r.steps[i + 1]!;
				const hitch = side === 'gl' && step.gl.compile && i <= played;
				return (
					<li key={i} className={`gv-chip${i <= played ? ' is-done' : ''}${i === played && s.kind === 'draw' ? ' is-now' : ''}${hitch ? ' is-hitch' : ''}`}>
						<span>{i + 1}</span>
						{hitch && (
							<b title={t.hitch} aria-label={t.hitch}>
								⚠
							</b>
						)}
						{side === 'vk' && i <= played && <small>p{step.pipeline}</small>}
					</li>
				);
			})}
		</ol>
	);
	const calls = (side: 'gl' | 'vk') => {
		const list = s[side].calls;
		return <pre className="gv-calls">{list.length ? list.join('\n') : t.noCalls}</pre>;
	};

	return (
		<LabFrame lang={lang} title={t.title} onReset={reset}>
			<div className="gv-scene">
				<p className="lab-muted">{t.scene}</p>
				<div className="gv-table-wrap">
					<table className="gv-table">
						<thead>
							<tr>
								<th>#</th>
								<th>{t.colDraw}</th>
								<th>{t.colShader}</th>
								<th>{t.colBlend}</th>
								<th>{t.colDepth}</th>
								<th>{t.colVertex}</th>
								<th />
							</tr>
						</thead>
						<tbody>
							{draws.map((d, i) => (
								<tr key={i} className={i === played && s.kind === 'draw' ? 'is-now' : undefined}>
									<td>{i + 1}</td>
									<td>{t[`n_${d.name}` as const]}</td>
									<td>
										<select aria-label={`${t.colShader} ${i + 1}`} value={d.shader} onChange={(e) => edit(i, { shader: e.currentTarget.value as Draw['shader'] })}>
											{SHADERS.map((x) => (
												<option key={x} value={x}>
													{t[`sh_${x}` as const]}
												</option>
											))}
										</select>
									</td>
									<td>
										<input type="checkbox" aria-label={`${t.colBlend} ${i + 1}`} checked={d.blend} onChange={(e) => edit(i, { blend: e.currentTarget.checked })} />
									</td>
									<td>
										<input type="checkbox" aria-label={`${t.colDepth} ${i + 1}`} checked={d.depth} onChange={(e) => edit(i, { depth: e.currentTarget.checked })} />
									</td>
									<td>
										<select aria-label={`${t.colVertex} ${i + 1}`} value={d.vertex} onChange={(e) => edit(i, { vertex: e.currentTarget.value as Draw['vertex'] })}>
											{VERTEX_FORMATS.map((x) => (
												<option key={x} value={x}>
													{t[`vx_${x}` as const]}
												</option>
											))}
										</select>
									</td>
									<td>
										<button type="button" className="lab-button gv-remove" onClick={() => remove(i)} disabled={draws.length <= 1} aria-label={fmt(t.remove, { i: i + 1 })} title={fmt(t.remove, { i: i + 1 })}>
											×
										</button>
									</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>
				<div className="lab-row">
					<button type="button" className="lab-button" onClick={add} disabled={draws.length >= MAX_DRAWS}>
						{t.add}
					</button>
					<label className="gv-dyn">
						<input type="checkbox" checked={dynDepth} onChange={(e) => setDynDepth(e.currentTarget.checked)} />
						<span>{fmt(t.dynDepth, { state: DEPTH_DYNAMIC })}</span>
					</label>
				</div>
			</div>

			<div className="gv-sides">
				<section className="gv-side is-gl" aria-label={t.glSide}>
					<h4>{t.glSide}</h4>
					<p className="lab-muted">{t.glFrame}</p>
					{chips('gl')}
					<p className="lab-muted">{t.calls}</p>
					{calls('gl')}
				</section>
				<section className="gv-side is-vk" aria-label={t.vkSide}>
					<h4>{t.vkSide}</h4>
					<p className="lab-muted">{t.vkFrame}</p>
					{chips('vk')}
					<p className="lab-muted">{t.calls}</p>
					{calls('vk')}
					<p className="lab-muted">{t.pipelines}</p>
					<ol className="gv-pipes" start={0}>
						{r.pipelines.map((k, i) => (
							<li key={k} className={i === s.pipeline ? 'is-now' : undefined}>
								<code>pipelines[{i}]</code> {pipeLabel(t, k)}
							</li>
						))}
					</ol>
				</section>
			</div>

			<Scrubber lang={lang} tl={tl} caption={caption(t, draws, r, s, dynDepth)} />

			<div className="lab-stats">
				<div className="lab-stat">
					<b>{r.programs.length}</b>
					<span>{t.statGlLoad}</span>
				</div>
				<div className={`lab-stat${s.glFrameCompiles > 0 ? ' gv-warn' : ''}`}>
					<b>{s.glFrameCompiles}</b>
					<span>{t.statGlFrame}</span>
				</div>
				<div className="lab-stat">
					<b>{r.pipelines.length}</b>
					<span>{t.statVkLoad}</span>
				</div>
				<div className="lab-stat">
					{/* Pipelines are complete objects: binding one never compiles. */}
					<b>0</b>
					<span>{t.statVkFrame}</span>
				</div>
			</div>
			<p className="lab-muted">{t.model}</p>

			<details className="gv-dynlist">
				<summary>{t.dynTitle}</summary>
				{DYNAMIC.map(([v, names]) => (
					<div key={v}>
						<p className="lab-muted">{fmt(t.dynSince, { v })}</p>
						<ul>
							{names.map((n) => (
								<li key={n} className={n === DEPTH_DYNAMIC ? 'is-now' : undefined}>
									<code>{n}</code>
								</li>
							))}
						</ul>
					</div>
				))}
			</details>
		</LabFrame>
	);
}
