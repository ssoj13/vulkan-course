/**
 * Graphics-pipeline lab (React island) over lib/labs/graphics-pipeline.ts, in three parts:
 * the stage strip (click a stage: what it does, the VkGraphicsPipelineCreateInfo member that configures it, its core
 * dynamic states), one triangle walked through the stages on a tiny framebuffer (scrubbable), and the pipeline count
 * of a sample scene with baked vs dynamic state.
 */
import { useMemo, useState } from 'react';
import { LabFrame } from 'course-kit/components/lab/LabFrame.tsx';
import { Scrubber } from 'course-kit/components/lab/Scrubber.tsx';
import { useTimeline } from 'course-kit/components/lab/useTimeline.ts';
import {
	AXES,
	CORE_DYNAMIC,
	CULL_MODES,
	DYNAMIC,
	FB,
	FULLY_STATIC,
	SCENE,
	SHIFT_X,
	STAGES,
	VERTS,
	WALL,
	WHOLE,
	count,
	dynamicOf,
	rgb,
	shortState,
	stage,
	toFramebuffer,
	vertexShader,
	versionOf,
	walk,
	type Axis,
	type CullMode,
	type DynChoice,
	type StageId,
	type WalkFrame,
} from '../../lib/labs/graphics-pipeline';
import { pick, tfmt } from '../../lib/text';
import { TEXT } from './graphics-pipeline.text';
import './graphics-pipeline.css';

type T = (typeof TEXT)['en'];

/** Pixel size of one framebuffer cell in the SVG. */
const CELL = 22;
const num = (x: number) => +x.toFixed(2);

function StageDetail({ lang, t, id }: { lang: string; t: T; id: StageId }) {
	const s = stage(id);
	const dyn = dynamicOf(id);
	return (
		<div className="gp-detail" aria-live="polite">
			<p className="gp-detail-head">
				<b>{t[`name.${id}`]}</b>
				<span className={`gp-tag is-${s.kind}`}>{s.kind === 'shader' ? t.kindShader : t.kindFixed}</span>
				{s.optional && <span className="gp-tag is-optional">{t.optional}</span>}
			</p>
			<p>{t[`what.${id}`]}</p>
			<h4>{t.configuredBy}</h4>
			<ul className="gp-config">
				{s.config.map((c) => (
					<li key={c.member + c.struct}>
						<code>{c.member}</code> → <code>{c.struct}</code>
					</li>
				))}
			</ul>
			<h4>{t.dynamicTitle}</h4>
			{dyn.length === 0 ? (
				<p className="lab-muted">{s.kind === 'shader' ? t.noDynamicShader : t.noDynamic}</p>
			) : (
				<>
					<p className="lab-muted">{tfmt(lang, t.dynamicCount, { n: dyn.length, total: CORE_DYNAMIC.length })}</p>
					<ul className="gp-dyn">
						{dyn.map((d) => (
							<li key={d.state}>
								<code>{shortState(d.state)}</code> <span className="gp-ver">{tfmt(lang, t.since, { v: versionOf(d.state) })}</span>
								<span className="gp-dyn-sub">
									{tfmt(lang, t.setWith, { cmd: d.cmd })}; {tfmt(lang, t.replaces, { what: d.replaces })}
								</span>
							</li>
						))}
					</ul>
				</>
			)}
		</div>
	);
}

/** The framebuffer picture of one walk frame. */
function WalkPicture({ t, f }: { t: T; f: WalkFrame }) {
	const W = FB.w * CELL;
	const H = FB.h * CELL;
	const before = f.step === 'vertex-input' || f.step === 'input-assembly';
	// Before the vertex shader the vertices are drawn where they would land without the model transform.
	const pre = VERTS.map((v) => toFramebuffer([v.pos[0], v.pos[1], v.pos[2], 1]));
	const pts = before ? pre : f.fb;
	const poly = (p: readonly (readonly number[])[]) => p.map((v) => `${v[0]! * CELL},${v[1]! * CELL}`).join(' ');
	const afterRaster = ['rasterization', 'fragment-shader', 'fragment-ops', 'color-blend', 'attachments'].includes(f.step);
	const colored = f.step !== 'rasterization';
	const showWall = ['fragment-ops', 'color-blend', 'attachments'].includes(f.step);
	const final = f.step === 'attachments';
	return (
		<svg className="gp-fb" viewBox={`-4 -4 ${W + 8} ${H + 8}`} role="img" aria-label={t.svgLabel}>
			<rect x={0} y={0} width={W} height={H} className="gp-fb-bg" />
			{showWall && <rect x={WALL.fromX * CELL} y={0} width={(FB.w - WALL.fromX) * CELL} height={H} className="gp-wall" />}
			{afterRaster &&
				f.frags.map((p) => (
					<rect
						key={`${p.x},${p.y}`}
						x={p.x * CELL + 1}
						y={p.y * CELL + 1}
						width={CELL - 2}
						height={CELL - 2}
						rx={2}
						className={colored ? 'gp-frag is-colored' : 'gp-frag'}
						style={colored ? { fill: rgb(p.color) } : undefined}
					/>
				))}
			{f.step === 'fragment-ops' &&
				f.rejected.map((p) => (
					<g key={`r${p.x},${p.y}`} className="gp-rejected">
						<rect x={p.x * CELL + 1} y={p.y * CELL + 1} width={CELL - 2} height={CELL - 2} rx={2} />
						<path d={`M${p.x * CELL + 5},${p.y * CELL + 5}l${CELL - 10},${CELL - 10}m0,${-(CELL - 10)}l${-(CELL - 10)},${CELL - 10}`} />
					</g>
				))}
			{!final && (
				<g className="gp-grid" aria-hidden="true">
					{Array.from({ length: FB.w - 1 }, (_, i) => (
						<line key={`v${i}`} x1={(i + 1) * CELL} y1={0} x2={(i + 1) * CELL} y2={H} />
					))}
					{Array.from({ length: FB.h - 1 }, (_, i) => (
						<line key={`h${i}`} x1={0} y1={(i + 1) * CELL} x2={W} y2={(i + 1) * CELL} />
					))}
				</g>
			)}
			{f.step === 'vertex-shader' && <polygon points={poly(pre)} className="gp-tri is-ghost" />}
			{f.step !== 'vertex-input' && !final && <polygon points={poly(pts)} className={`gp-tri${f.culled && afterRaster ? ' is-culled' : ''}`} />}
			{!final &&
				pts.map((v, i) => (
					<g key={i}>
						<circle cx={v[0]! * CELL} cy={v[1]! * CELL} r={5} style={{ fill: rgb(VERTS[i]!.color) }} className="gp-vtx" />
						<text x={v[0]! * CELL + 8} y={v[1]! * CELL - 6} className="gp-vtx-label">
							{i}
						</text>
					</g>
				))}
		</svg>
	);
}

function walkCaption(lang: string, t: T, f: WalkFrame, cull: CullMode): string {
	const n = f.step === 'rasterization' || f.step === 'fragment-shader' ? f.frags.length : 0;
	const empty = f.culled && ['fragment-shader', 'fragment-ops', 'color-blend', 'attachments'].includes(f.step);
	if (empty) return t.walkEmpty;
	switch (f.step) {
		case 'vertex-shader':
			return tfmt(lang, t['walk.vertex-shader'], { dx: SHIFT_X });
		case 'rasterization':
			return tfmt(lang, t['walk.rasterization'], {
				w: FB.w,
				h: FB.h,
				area: num(f.area),
				facing: f.area > 0 ? t.facingFront : t.facingBack,
				result: f.culled ? tfmt(lang, t.rasterCulled, { mode: t[`cull.${cull}`] }) : tfmt(lang, t.rasterCovered, { n }),
			});
		case 'fragment-shader':
			return tfmt(lang, t['walk.fragment-shader'], { n });
		case 'fragment-ops':
			return tfmt(lang, t['walk.fragment-ops'], { wall: WALL.depth, rejected: f.rejected.length, kept: f.frags.length });
		case 'color-blend':
			return tfmt(lang, t['walk.color-blend'], { kept: f.frags.length });
		default:
			return t[`walk.${f.step}`];
	}
}

function Walk({ t, lang }: { t: T; lang: string }) {
	const [cull, setCull] = useState<CullMode>('back');
	const frames = useMemo(() => walk(cull), [cull]);
	const tl = useTimeline(frames);
	const f = tl.frame;
	const afterVs = tl.index >= frames.findIndex((x) => x.step === 'vertex-shader');
	return (
		<section className="gp-walk" aria-label={t.walkTitle}>
			<h4>{t.walkTitle}</h4>
			<div className="gp-walk-controls lab-row">
				<span>{t.cullLabel}:</span>
				{CULL_MODES.map((m) => (
					<button key={m} type="button" className={`lab-button${m === cull ? ' is-primary' : ''}`} aria-pressed={m === cull} onClick={() => setCull(m)}>
						{t[`cull.${m}`]}
					</button>
				))}
			</div>
			<ol className="gp-walk-strip" aria-hidden="true">
				{STAGES.map((s) => (
					<li key={s.id} className={`${s.optional ? 'is-optional' : ''}${s.id === f.stage || (f.step === 'skipped' && s.id === 'geometry') ? ' is-current' : ''}`}>
						{t[`name.${s.id}`]}
					</li>
				))}
			</ol>
			<div className="gp-walk-body">
				<WalkPicture t={t} f={f} />
				<div className="gp-walk-side">
					<table className="gp-vtx-table">
						<caption>{t.vtxTable}</caption>
						<thead>
							<tr>
								<th>#</th>
								<th>{t.vtxPos}</th>
								<th>{t.vtxColor}</th>
								{afterVs && <th>{t.vtxClip}</th>}
							</tr>
						</thead>
						<tbody>
							{VERTS.map((v, i) => (
								<tr key={i}>
									<td>{i}</td>
									<td>
										<code>({v.pos.join(', ')})</code>
									</td>
									<td>
										<span className="gp-swatch" style={{ background: rgb(v.color) }} />
									</td>
									{afterVs && (
										<td>
											<code>({vertexShader(v).map(num).join(', ')})</code>
										</td>
									)}
								</tr>
							))}
						</tbody>
					</table>
					<ul className="gp-legend">
						<li>
							<span className="gp-key is-frag" /> {t.legendFrag}
						</li>
						<li>
							<span className="gp-key is-rejected" /> {t.legendRejected}
						</li>
						<li>
							<span className="gp-key is-wall" /> {tfmt(lang, t.legendWall, { wall: WALL.depth })}
						</li>
					</ul>
				</div>
			</div>
			<Scrubber lang={lang} tl={tl} caption={walkCaption(lang, t, f, cull)} />
			<p className="lab-muted">{t.walkNote}</p>
		</section>
	);
}

function Count({ lang, t }: { lang: string; t: T }) {
	const [choice, setChoice] = useState<DynChoice>(DYNAMIC);
	const c = count(choice);
	const toggle = (a: Axis) => setChoice((x) => ({ ...x, [a]: !x[a] }));
	const preset = (p: DynChoice, label: string) => {
		const on = AXES.every((a) => p[a] === choice[a]);
		return (
			<button type="button" className={`lab-button${on ? ' is-primary' : ''}`} aria-pressed={on} onClick={() => setChoice(p)}>
				{label}
			</button>
		);
	};
	return (
		<section className="gp-count" aria-label={t.countTitle}>
			<h4>{t.countTitle}</h4>
			<p>{tfmt(lang, t.countScene, { m: SCENE.materials })}</p>
			<div className="lab-row">
				{preset(FULLY_STATIC, t.presetStatic)}
				{preset(DYNAMIC, t.presetDynamic)}
			</div>
			<fieldset className="gp-axes">
				<legend className="lab-muted">{t.axesLabel}</legend>
				{AXES.map((a) => (
					<label key={a} className={`gp-axis${choice[a] ? ' is-on' : ''}`}>
						<input type="checkbox" checked={choice[a]} onChange={() => toggle(a)} />
						{t[`axis.${a}`]}
					</label>
				))}
			</fieldset>
			<div className="gp-count-result">
				<b aria-live="polite">{tfmt(lang, t.countResult, { n: c.pipelines })}</b>
				<span className="lab-muted">
					{tfmt(lang, t.formula, { m: SCENE.materials, cull: choice.cull ? 1 : SCENE.cull, depth: choice.depth ? 1 : SCENE.depth })}
				</span>
				<div className="gp-pipes" aria-hidden="true">
					{Array.from({ length: c.pipelines }, (_, i) => (
						<span key={i} className="gp-pipe" />
					))}
				</div>
			</div>
			<p className={c.perResize ? 'lab-msg is-warn' : 'lab-msg is-ok'}>{c.perResize ? tfmt(lang, t.perResize, { n: c.perResize }) : t.noResize}</p>
			<p>
				{c.dynamicStates.length ? tfmt(lang, t.dynList, { list: c.dynamicStates.map(shortState).join(', ') }) : t.dynNone}
				{c.cmds.length > 0 && (
					<>
						<br />
						{tfmt(lang, t.cmdList, { list: c.cmds.join(', ') })}
					</>
				)}
			</p>
			<p className="lab-muted">{t.countNote}</p>
		</section>
	);
}

export default function GraphicsPipelineView({ lang }: { lang: string }) {
	const t = pick(TEXT, lang, 'graphics-pipeline');
	const [sel, setSel] = useState<StageId>('vertex-input');
	return (
		<LabFrame lang={lang} title={t.title}>
			<section aria-label={t.stagesTitle}>
				<h4>{t.stagesTitle}</h4>
				<p className="lab-muted">{t.stagesHint}</p>
				<div className="gp-stages">
					<ol className="gp-strip" aria-label={t.stagesLabel}>
						{STAGES.map((s, i) => (
							<li key={s.id}>
								<button
									type="button"
									className={`gp-stage is-${s.kind}${s.optional ? ' is-optional' : ''}${s.id === sel ? ' is-selected' : ''}`}
									aria-pressed={s.id === sel}
									onClick={() => setSel(s.id)}
								>
									<span className="gp-stage-n">{i + 1}</span>
									<span>{t[`name.${s.id}`]}</span>
									{dynamicOf(s.id).length > 0 && <span className="gp-stage-dyn">{dynamicOf(s.id).length}</span>}
								</button>
							</li>
						))}
					</ol>
					<StageDetail lang={lang} t={t} id={sel} />
				</div>
				<p className="lab-muted">{tfmt(lang, t.wholePipeline, { layout: WHOLE[0]!.member, dynamic: WHOLE[1]!.member })}</p>
			</section>
			<Walk t={t} lang={lang} />
			<Count lang={lang} t={t} />
		</LabFrame>
	);
}
