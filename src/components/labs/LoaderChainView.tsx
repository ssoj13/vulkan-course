/**
 * Loader-chain lab (React island): the reader picks a call, the layers and the drivers, then steps the call hop by hop
 * from the program through the loader's trampoline, the active layers and the terminator to the driver(s). The path
 * and its rules come from lib/labs/loader-chain.ts; boxes the call does not pass are drawn as "not on this path".
 */
import { useMemo, useState } from 'react';
import { LabFrame } from 'course-kit/components/lab/LabFrame.tsx';
import { Scrubber } from 'course-kit/components/lab/Scrubber.tsx';
import { useTimeline } from 'course-kit/components/lab/useTimeline.ts';
import { fmt } from 'course-kit/i18n.mjs';
import {
	CALLS,
	ICDS,
	ICD_INFO,
	IMPLICIT_LAYERS,
	LEVEL,
	icdNode,
	implicitActive,
	layerNode,
	trace,
	type CallId,
	type IcdId,
	type NodeId,
	type Setup,
	type Via,
} from '../../lib/labs/loader-chain';
import { pick } from '../../lib/text';
import { TEXT } from './loader-chain.text';
import './loader-chain.css';

const INITIAL: Setup = { call: 'vkCreateInstance', env: new Map(), validation: true, second: false, gpu: 'nvidia', via: 'gdpa' };

export default function LoaderChainView({ lang }: { lang: string }) {
	const t = pick(TEXT, lang, 'loader-chain');
	const [setup, setSetup] = useState<Setup>(INITIAL);
	const tr = useMemo(() => trace(setup), [setup]);
	const tl = useTimeline(tr.hops);

	// A new setup is a new animation: start it from the first hop.
	const change = (patch: Partial<Setup>) => {
		setSetup((s) => ({ ...s, ...patch }));
		tl.reset();
	};
	/** Sets a variable to a value; an empty value unsets it. */
	const setEnv = (name: string, value: string) => {
		const env = new Map(setup.env);
		if (value) env.set(name, value);
		else env.delete(name);
		change({ env });
	};
	const reset = () => {
		setSetup(INITIAL);
		tl.reset();
	};

	const hop = tl.frame;
	const now = new Set(hop.at);
	const seen = new Set(tr.hops.slice(0, tl.index).flatMap((h) => h.at));
	const cls = (id: NodeId) => `lc-box${now.has(id) ? ' is-now' : seen.has(id) ? ' is-seen' : ''}${tr.path.has(id) ? '' : ' is-off'}`;
	const level = LEVEL[setup.call];
	const off = (id: NodeId) => !tr.path.has(id) && <span className="lc-off">{t.offPath}</span>;

	return (
		<LabFrame lang={lang} title={t.title} onReset={reset}>
			<div className="lc-grid">
				<div className="lc-controls">
					<label className="lab-field">
						<span>{t.callLabel}</span>
						<select value={setup.call} onChange={(e) => change({ call: e.currentTarget.value as CallId })}>
							{CALLS.map((c) => (
								<option key={c} value={c}>
									{c}
								</option>
							))}
						</select>
					</label>
					<fieldset className="lc-set" disabled={level !== 'device'}>
						<legend>{t.viaLabel}</legend>
						{(['gdpa', 'export'] as const satisfies readonly Via[]).map((v) => (
							<label key={v}>
								<input type="radio" name="lc-via" checked={setup.via === v} onChange={() => change({ via: v })} />
								{v === 'gdpa' ? t.viaGdpa : t.viaExport}
							</label>
						))}
					</fieldset>
					<fieldset className="lc-set">
						<legend>{t.appLayers}</legend>
						<label>
							<input type="checkbox" checked={setup.validation} onChange={(e) => change({ validation: e.currentTarget.checked })} />
							<code>{t.validation}</code>
						</label>
					</fieldset>
					<fieldset className="lc-set">
						<legend>{t.drivers}</legend>
						<label>
							<input type="checkbox" checked disabled />
							<code>{ICD_INFO.nvidia.dll}</code> ({ICD_INFO.nvidia.gpu})
						</label>
						<label>
							<input type="checkbox" checked={setup.second} onChange={(e) => change({ second: e.currentTarget.checked })} />
							{t.secondIcd}
						</label>
					</fieldset>
					<label className="lab-field">
						<span>{t.gpuLabel}</span>
						<select value={setup.second ? setup.gpu : 'nvidia'} disabled={!setup.second || level === 'instance'} onChange={(e) => change({ gpu: e.currentTarget.value as IcdId })}>
							{ICDS.map((i) => (
								<option key={i} value={i}>
									{ICD_INFO[i].gpu} ({ICD_INFO[i].dll})
								</option>
							))}
						</select>
					</label>
				</div>

				<ol className="lc-chain" aria-label={t.chainLabel}>
					<li className={cls('app')}>
						<b>{t.app}</b>
					</li>
					<li className={cls('trampoline')}>
						<b>{t.trampoline}</b> <span className="lc-sub">{t.trampolineSub}</span>
						{off('trampoline')}
					</li>
					{tr.layers.length === 0 && <li className="lc-note">{level !== 'device' ? t.noLayers : setup.via === 'export' ? t.noLayersDevice : t.noLayersGdpa}</li>}
					{tr.layers.map((l) => (
						<li key={l.name} className={`${cls(layerNode(l.name))} is-layer is-${l.kind}`}>
							<code>{l.name}</code> <span className="lc-tag">{l.kind === 'implicit' ? t.implicit : t.explicit}</span>
						</li>
					))}
					<li className={cls('terminator')}>
						<b>{t.terminator}</b> <span className="lc-sub">{t.terminatorSub}</span>
						{off('terminator')}
					</li>
					<li className="lc-icds">
						{tr.icds.map((i) => (
							<div key={i} className={`${cls(icdNode(i))} is-icd`}>
								<code>{ICD_INFO[i].dll}</code>
								<span className="lc-sub">{ICD_INFO[i].gpu}</span>
								{off(icdNode(i))}
							</div>
						))}
					</li>
				</ol>
			</div>

			<Scrubber lang={lang} tl={tl} caption={fmt(t[hop.key], hop.vars)} />

			<details className="lc-env" open>
				<summary>{t.envTitle}</summary>
				<p className="lab-muted">{t.envHint}</p>
				<div className="lc-table-wrap">
					<table>
						<thead>
							<tr>
								<th>{t.colLayer}</th>
								<th>{t.colBy}</th>
								<th>{t.colEnable}</th>
								<th>{t.colDisable}</th>
								<th>{t.colState}</th>
							</tr>
						</thead>
						<tbody>
							{IMPLICIT_LAYERS.map((l) => {
								const on = implicitActive(l, setup.env);
								// A disable variable counts with any value, so a checkbox (sets "1") is enough.
								const box = (name: string) => (
									<label className="lc-var">
										<input type="checkbox" checked={setup.env.has(name)} onChange={(e) => setEnv(name, e.currentTarget.checked ? '1' : '')} aria-label={fmt(t.setVar, { name })} />
										<code>{name}</code>
									</label>
								);
								// An enable variable must match the manifest's value, so the reader types the value.
								const field = ({ name, value }: { name: string; value: string }) => (
									<label className="lc-var">
										<code>{name}</code> =
										<input
											type="text"
											size={4}
											value={setup.env.get(name) ?? ''}
											placeholder={t.envUnset}
											onChange={(e) => setEnv(name, e.currentTarget.value)}
											aria-label={fmt(t.envValue, { name, value })}
										/>
									</label>
								);
								return (
									<tr key={l.name} className={on ? 'is-on' : 'is-out'}>
										<td>
											<code>{l.name}</code>
										</td>
										<td>{l.by}</td>
										<td>{l.enableEnv ? field(l.enableEnv) : t.none}</td>
										<td>{box(l.disableEnv)}</td>
										<td>{on ? t.stateOn : t.stateOff}</td>
									</tr>
								);
							})}
						</tbody>
					</table>
				</div>
			</details>
			<p className="lab-muted lc-simplified">{t.simplified}</p>
		</LabFrame>
	);
}
