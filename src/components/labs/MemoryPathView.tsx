/**
 * Memory-path lab (React island): pick a data size and a path (GPU reads VRAM, GPU reads system RAM over PCIe, CPU
 * writes into the BAR window) and see the route, the memory type and heap of the course GPU it uses, and the time
 * per frame at theoretical peak bandwidth against a 60 fps budget. Model: lib/labs/memory-path.ts.
 */
import { useState } from 'react';
import { LabFrame } from 'course-kit/components/lab/LabFrame.tsx';
import { fmt } from 'course-kit/i18n.mjs';
import { pick } from '../../lib/text';
import {
	BW,
	FRAME_MS,
	HEAPS,
	NODES,
	PATH,
	PATHS,
	SIZES,
	TYPES,
	VUID_HEAP_SIZE,
	bytes,
	shortFlag,
	transfer,
	type NodeId,
	type PathId,
} from '../../lib/labs/memory-path';
import { PATH_KEY, TEXT } from './memory-path.text';
import './memory-path.css';

type Text = (typeof TEXT)['en'];

const NODE_KEY: Record<NodeId, keyof Text> = { ram: 'nodeRam', cpu: 'nodeCpu', pcie: 'nodePcie', gpu: 'nodeGpu', vram: 'nodeVram' };
/** Index of the default size on the slider: 64 MiB, a few large textures. */
const SIZE0 = 6;

export default function MemoryPathView({ lang }: { lang: string }) {
	const t = pick(TEXT, lang, 'memory-path');
	const [si, setSi] = useState(SIZE0);
	const [path, setPath] = useState<PathId>('vram');
	const size = SIZES[si]!;
	const tr = transfer(path, size);
	const num = (n: number) => new Intl.NumberFormat(lang, { maximumSignificantDigits: 3 }).format(n);
	const gbps = (bw: number) => fmt(t.gbps, { gb: num(bw / 1e9) });
	const route = new Set<NodeId>(PATH[path].route);
	// Heaps of the picture, read from the type each path picks: RAM and VRAM boxes show their real sizes.
	const heapOf = (p: PathId) => transfer(p, size).heap;
	const level = tr.budget < 0.5 ? 'is-ok' : tr.budget < 1 ? 'is-warn' : 'is-err';
	const flags = (fs: readonly string[]) => (fs.length ? fs.map(shortFlag).join(' | ') : t.noFlags);

	const sub: Partial<Record<NodeId, string>> = {
		ram: bytes(heapOf('sysram').size, lang),
		pcie: gbps(BW.pcie),
		vram: `${bytes(heapOf('vram').size, lang)} · ${gbps(BW.vram)}`,
	};

	return (
		<LabFrame
			lang={lang}
			title={t.title}
			onReset={() => {
				setSi(SIZE0);
				setPath('vram');
			}}
		>
			<div className="mp-controls">
				<label className="lab-field">
					<span>{fmt(t.size, { size: bytes(size, lang) })}</span>
					<input
						type="range"
						min={0}
						max={SIZES.length - 1}
						step={1}
						value={si}
						aria-valuetext={bytes(size, lang)}
						onChange={(e) => setSi(Number(e.currentTarget.value))}
					/>
				</label>
				<fieldset className="mp-paths">
					<legend className="lab-muted">{t.path}</legend>
					{PATHS.map((p) => (
						<label key={p} className={`mp-path${p === path ? ' is-on' : ''}`}>
							<input type="radio" name="mp-path" value={p} checked={p === path} onChange={() => setPath(p)} />
							{t[PATH_KEY[p]]}
						</label>
					))}
				</fieldset>
			</div>

			<ol className="mp-chain" aria-label={t[PATH_KEY[path]]}>
				{NODES.map((n, i) => {
					const next = NODES[i + 1];
					// A link is lit when both of its ends are on the route.
					const lit = next !== undefined && route.has(n) && route.has(next);
					return (
						<li key={n} className="mp-step">
							<div className={`mp-node${route.has(n) ? ' is-on' : ''}`}>
								<b>{t[NODE_KEY[n]]}</b>
								{sub[n] && <span>{sub[n]}</span>}
								{n === 'vram' && (
									<span className={`mp-bar${path === 'bar' ? ' is-on' : ''}`}>BAR {bytes(heapOf('bar').size, lang)}</span>
								)}
							</div>
							{next && <span className={`mp-link${lit ? ' is-on' : ''}`} aria-hidden="true" />}
						</li>
					);
				})}
			</ol>

			<div className="mp-time">
				<p>{fmt(t.time, { ms: num(tr.ms), bw: gbps(PATH[path].bw), peak: t.peak })}</p>
				<div
					className="mp-meter"
					role="meter"
					aria-valuemin={0}
					aria-valuemax={100}
					aria-valuenow={Math.min(100, Math.round(tr.budget * 100))}
					aria-label={fmt(t.budget, { pct: num(tr.budget * 100), frame: num(FRAME_MS) })}
				>
					<span className={level} style={{ width: `${Math.min(100, tr.budget * 100)}%` }} />
				</div>
				<p className={`lab-msg ${level}`}>
					{tr.budget < 1 ? fmt(t.budget, { pct: num(tr.budget * 100), frame: num(FRAME_MS) }) : fmt(t.over, { x: num(tr.budget) })}
				</p>
			</div>

			{!tr.fits && (
				<div className="lab-msg is-err">
					<p>{fmt(t.noFit, { size: bytes(size, lang), heap: tr.heap.index, heapSize: bytes(tr.heap.size, lang) })}</p>
					<p className="mp-vuid">
						<code>{VUID_HEAP_SIZE.id}</code> {VUID_HEAP_SIZE.text}
					</p>
				</div>
			)}

			<div className="mp-types">
				<h4>{t.heapsHead}</h4>
				<ul>
					{HEAPS.map((h) => (
						<li key={`h${h.index}`} className={h.index === tr.heap.index ? 'is-on' : undefined}>
							{fmt(t.heapRow, { index: h.index, size: bytes(h.size, lang) })} — <code>{flags(h.flags)}</code>
						</li>
					))}
				</ul>
				<p className="lab-muted">{fmt(t.typeHead, { flags: PATH[path].need.map(shortFlag).join(' | ') })}</p>
				<ul>
					{TYPES.map((ty) => (
						<li key={`t${ty.index}`} className={ty.index === tr.type.index ? 'is-on' : undefined} aria-current={ty.index === tr.type.index ? 'true' : undefined}>
							{fmt(t.typeRow, { index: ty.index, heap: ty.heapIndex })} — <code>{flags(ty.flags)}</code>
						</li>
					))}
				</ul>
			</div>

			<p>{fmt(t.why, { fast: num(transfer('vram', size).ms), slow: num(transfer('sysram', size).ms) })}</p>
			<p className="lab-muted">{t.note}</p>
		</LabFrame>
	);
}
