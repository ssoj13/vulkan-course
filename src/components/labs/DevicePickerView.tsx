/**
 * Device-picker lab island: the course GPU's properties, features and limits (real, gpu.json) in tabs, and step 03's
 * picking rule as editable inputs evaluated over the real GPU plus two hypothetical ones, with a per-device rule trace.
 * All logic lives in lib/labs/device-picker.ts.
 */
import { useMemo, useState, type KeyboardEvent } from 'react';
import { LabFrame } from 'course-kit/components/lab/LabFrame.tsx';
import gpu from '../../data/gpu.json';
import {
	CANDIDATES,
	STEP03_FEATURES,
	STEP03_RULE,
	VENDORS,
	featureGroups,
	featureMinor,
	hex,
	limitRows,
	nvidiaDriver,
	pick,
	shortType,
	usable,
	type Candidate,
	type Check,
	type Rule,
	type Step03Feature,
} from '../../lib/labs/device-picker';
import { pick as pickText, tfmt } from '../../lib/text';
import { TEXT } from './device-picker.text';
import './device-picker.css';

const TABS = ['props', 'features', 'limits', 'picker'] as const;
type Tab = (typeof TABS)[number];
const TAB_KEY = { props: 'tabProps', features: 'tabFeatures', limits: 'tabLimits', picker: 'tabPicker' } as const;
type Texts = (typeof TEXT)['en'];
type TypeKey = Extract<keyof Texts, `type_${string}`>;
type UnitKey = Extract<keyof Texts, `unit_${string}`>;
type LimitKey = Extract<keyof Texts, `L_${string}`>;
type CandKey = Extract<keyof Texts, `cand_${string}`>;
/** Hypothetical devices are named in the reader's language; the real one by its driver-reported name. */
const label = (t: Texts, c: Candidate) => c.name ?? t[`cand_${c.id}` as CandKey];

const ALL_ON = Object.fromEntries(CANDIDATES.map((c) => [c.id, true]));

export default function DevicePickerView({ lang }: { lang: string }) {
	const t = pickText(TEXT, lang, 'device-picker');
	const [tab, setTab] = useState<Tab>('props');
	const [fq, setFq] = useState('');
	const [fOff, setFOff] = useState(false);
	const [lq, setLq] = useState('');
	const [lExpl, setLExpl] = useState(false);
	const [rule, setRule] = useState<Rule>(STEP03_RULE);
	const [on, setOn] = useState<Record<string, boolean>>(ALL_ON);

	const reset = () => {
		setRule(STEP03_RULE);
		setOn(ALL_ON);
		setFq('');
		setFOff(false);
		setLq('');
		setLExpl(false);
	};

	/** Arrow keys move between tabs (WAI-ARIA tabs pattern). */
	const onTabKey = (e: KeyboardEvent<HTMLDivElement>) => {
		const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
		if (!d) return;
		e.preventDefault();
		const next = TABS[(TABS.indexOf(tab) + d + TABS.length) % TABS.length]!;
		setTab(next);
		document.getElementById(`vk-dp-tab-${next}`)?.focus();
	};

	return (
		<LabFrame lang={lang} title={t.title} onReset={reset}>
			<div className="vk-dp-tabs" role="tablist" aria-label={t.tabs} onKeyDown={onTabKey}>
				{TABS.map((k) => (
					<button
						key={k}
						id={`vk-dp-tab-${k}`}
						type="button"
						role="tab"
						aria-selected={tab === k}
						aria-controls={`vk-dp-panel-${k}`}
						tabIndex={tab === k ? 0 : -1}
						className={`lab-button${tab === k ? ' is-primary' : ''}`}
						onClick={() => setTab(k)}
					>
						{t[TAB_KEY[k]]}
					</button>
				))}
			</div>
			<div className="vk-dp-panel" role="tabpanel" id={`vk-dp-panel-${tab}`} aria-labelledby={`vk-dp-tab-${tab}`}>
				{tab === 'props' && <Props lang={lang} t={t} />}
				{tab === 'features' && <Features lang={lang} t={t} q={fq} setQ={setFq} off={fOff} setOff={setFOff} />}
				{tab === 'limits' && <Limits lang={lang} t={t} q={lq} setQ={setLq} expl={lExpl} setExpl={setLExpl} />}
				{tab === 'picker' && <Picker lang={lang} t={t} rule={rule} setRule={setRule} on={on} setOn={setOn} />}
			</div>
			<p className="lab-muted">{tfmt(lang, t.source, { date: gpu.meta.date, sdk: gpu.meta.sdk })}</p>
		</LabFrame>
	);
}

function Props({ lang, t }: { lang: string; t: Texts }) {
	const d = gpu.device;
	const type = shortType(d.type);
	const rows: [string, string, string][] = [
		[t.pName, d.name, ''],
		[t.pType, type, t[`type_${type}` as TypeKey] ?? ''],
		[t.pApi, d.apiVersion, tfmt(lang, t.pApiNote, { api: d.apiVersion, usable: usable(4, d.apiVersion) })],
		[t.pDriver, `${hex(d.driverVersionRaw, 8)} = ${nvidiaDriver(d.driverVersionRaw)}`, tfmt(lang, t.pDriverNote, { raw: hex(d.driverVersionRaw, 8), ver: d.driverVersion })],
		[t.pVendor, hex(d.vendorID), tfmt(lang, t.pVendorNote, { vendor: VENDORS[d.vendorID] ?? '?' })],
		[t.pDevice, hex(d.deviceID), t.pDeviceNote],
	];
	return (
		<dl className="vk-dp-props">
			{rows.map(([k, v, note]) => (
				<div key={k}>
					<dt>
						<code>{k}</code>
					</dt>
					<dd>
						<b>{v}</b>
						{note && <span className="lab-muted"> {note}</span>}
					</dd>
				</div>
			))}
		</dl>
	);
}

function Features({ lang, t, q, setQ, off, setOff }: { lang: string; t: Texts; q: string; setQ: (s: string) => void; off: boolean; setOff: (b: boolean) => void }) {
	const groups = useMemo(() => featureGroups(q, off), [q, off]);
	const empty = groups.every((g) => !g.rows.length);
	return (
		<>
			<div className="lab-row">
				<input type="text" className="vk-dp-search" placeholder={t.fSearch} aria-label={t.fSearch} value={q} onChange={(e) => setQ(e.target.value)} />
				<label className="vk-dp-check">
					<input type="checkbox" checked={off} onChange={(e) => setOff(e.target.checked)} /> {t.fOnlyOff}
				</label>
			</div>
			<p className="lab-muted">{t.fNote}</p>
			{empty && <p className="lab-msg">{t.fNone}</p>}
			{groups
				.filter((g) => g.rows.length)
				.map((g) => (
					<section key={g.struct} className="vk-dp-group">
						<h4>{tfmt(lang, t.fGroup, { struct: g.struct, minor: g.minor, n: g.supported, total: g.total })}</h4>
						<ul className="vk-dp-feats">
							{g.rows.map((r) => (
								<li key={r.name} className={`${r.value ? 'is-yes' : 'is-no'}${r.step03 ? ' is-marked' : ''}`}>
									<code>{r.name}</code>
									<span className="vk-dp-val">{r.value ? t.fYes : t.fNo}</span>
									{r.step03 && <span className="vk-dp-badge">{t.fStep03}</span>}
									{r.pickRule && <span className="vk-dp-badge is-accent">{t.fPick}</span>}
								</li>
							))}
						</ul>
					</section>
				))}
		</>
	);
}

function Limits({ lang, t, q, setQ, expl, setExpl }: { lang: string; t: Texts; q: string; setQ: (s: string) => void; expl: boolean; setExpl: (b: boolean) => void }) {
	const rows = useMemo(() => limitRows(q, expl), [q, expl]);
	return (
		<>
			<div className="lab-row">
				<input type="text" className="vk-dp-search" placeholder={t.lSearch} aria-label={t.lSearch} value={q} onChange={(e) => setQ(e.target.value)} />
				<label className="vk-dp-check">
					<input type="checkbox" checked={expl} onChange={(e) => setExpl(e.target.checked)} /> {t.lOnlyExplained}
				</label>
			</div>
			<p className="lab-muted">{t.lNote}</p>
			{rows.length === 0 && <p className="lab-msg">{t.lNone}</p>}
			<ul className="vk-dp-limits">
				{rows.map((r) => (
					<li key={r.name} className={r.unit ? 'is-explained' : undefined}>
						<div className="vk-dp-limit-head">
							<code>{r.name}</code>
							<b>
								{r.value}
								{r.unit && t[`unit_${r.unit}` as UnitKey] ? ` ${tfmt(lang, t[`unit_${r.unit}` as UnitKey], { n: r.n! })}` : ''}
								{r.size ? ` (${r.size})` : ''}
							</b>
						</div>
						{r.unit && <p className="vk-dp-limit-text">{t[`L_${r.name}` as LimitKey]}</p>}
					</li>
				))}
			</ul>
		</>
	);
}

function Picker({
	lang,
	t,
	rule,
	setRule,
	on,
	setOn,
}: {
	lang: string;
	t: Texts;
	rule: Rule;
	setRule: (r: Rule) => void;
	on: Record<string, boolean>;
	setOn: (o: Record<string, boolean>) => void;
}) {
	const present = CANDIDATES.filter((c) => on[c.id]);
	const result = pick(present, rule);
	const verdict = new Map(result.verdicts.map((v) => [v.id, v]));
	const winner = result.picked ? CANDIDATES.find((c) => c.id === result.picked)! : null;
	const winScore = winner ? verdict.get(winner.id)!.score : 0;

	const toggleFeature = (f: Step03Feature, yes: boolean) =>
		setRule({ ...rule, required: STEP03_FEATURES.map((x) => x.name).filter((n) => (n === f ? yes : rule.required.includes(n))) });

	const line = (c: Exclude<Check, { kind: 'score' }>, have: string): string => {
		if (c.kind === 'version') return tfmt(lang, c.ok ? t.trVersionOk : t.trVersionBad, { have: c.have, minor: c.minor });
		if (c.value === 'na') return tfmt(lang, t.trFeatNa, { name: c.name, minor: featureMinor(c.name), have });
		return tfmt(lang, c.value ? t.trFeatOk : t.trFeatFalse, { name: c.name });
	};

	return (
		<div className="vk-dp-picker">
			<p className={`lab-msg vk-dp-result ${winner ? 'is-ok' : 'is-err'}`} aria-live="polite">
				{winner
					? tfmt(lang, rule.preferDiscrete && result.tiedLater.length === 0 ? t.pickedScore : t.pickedFirst, { name: label(t, winner), score: winScore })
					: t.pickedNone}
			</p>
			<fieldset className="vk-dp-rule">
				<legend>{t.rule}</legend>
				<label className="lab-field">
					<span>{t.ruleVersion}</span>
					<select value={rule.minMinor} onChange={(e) => setRule({ ...rule, minMinor: Number(e.target.value) })}>
						{[0, 1, 2, 3, 4].map((m) => (
							<option key={m} value={m}>
								1.{m}
							</option>
						))}
					</select>
				</label>
				<div className="lab-field">
					<span>{t.ruleFeatures}</span>
					{STEP03_FEATURES.map((f) => (
						<label key={f.name} className="vk-dp-check">
							<input type="checkbox" checked={rule.required.includes(f.name)} onChange={(e) => toggleFeature(f.name, e.target.checked)} />
							<code>{f.name}</code> <span className="lab-muted">1.{f.minor}</span>
						</label>
					))}
				</div>
				<label className="vk-dp-check">
					<input type="checkbox" checked={rule.preferDiscrete} onChange={(e) => setRule({ ...rule, preferDiscrete: e.target.checked })} /> {t.rulePrefer}
				</label>
				<button type="button" className="lab-button" onClick={() => setRule(STEP03_RULE)}>
					{t.ruleReset}
				</button>
			</fieldset>

			<div className="vk-dp-cands">
				<h4>{t.cands}</h4>
				<ol>
					{CANDIDATES.map((c) => {
						const v = verdict.get(c.id);
						const won = c.id === result.picked;
						return (
							<li key={c.id} className={`vk-dp-cand${won ? ' is-picked' : ''}${on[c.id] ? '' : ' is-off'}${c.hypothetical ? ' is-hypo' : ''}`}>
								<label className="vk-dp-check">
									<input type="checkbox" checked={on[c.id]} onChange={(e) => setOn({ ...on, [c.id]: e.target.checked })} aria-label={tfmt(lang, t.candOn, { name: label(t, c) })} />
									<b>{label(t, c)}</b>
									<span className={`vk-dp-badge${c.hypothetical ? ' is-hypo' : ' is-accent'}`}>{c.hypothetical ? t.hypo : t.real}</span>
								</label>
								<p className="lab-muted">
									{shortType(c.type)} · apiVersion {c.apiVersion}
								</p>
								<ul className="vk-dp-trace" aria-label={`${t.trace}: ${label(t, c)}`}>
									{!v && <li className="is-skip">{t.trOff}</li>}
									{v?.checks.map((ch, i) =>
										ch.kind === 'score' ? (
											<li key={i} className="is-ok">
												{rule.preferDiscrete ? tfmt(lang, t.trScore, { score: ch.score, type: shortType(c.type) }) : t.trScoreOff}
												{' - '}
												{won ? t.trWin : result.tiedLater.includes(c.id) ? t.trTie : t.trLose}
											</li>
										) : (
											<li key={i} className={ch.ok ? 'is-ok' : 'is-bad'}>
												{line(ch, c.apiVersion)}
											</li>
										),
									)}
								</ul>
							</li>
						);
					})}
				</ol>
				<p className="lab-muted">{t.hypoNote}</p>
			</div>
		</div>
	);
}
