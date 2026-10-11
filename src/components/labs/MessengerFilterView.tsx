/**
 * Messenger-filter lab (React island): severity and type checkboxes build the two masks of
 * VkDebugUtilsMessengerCreateInfoEXT; the real log of step 02 (lib/labs/messenger-filter.ts) shows which lines reach
 * the callback and why the others are dropped, next to the create-info code for those masks.
 */
import { useState } from 'react';
import { LabFrame } from 'course-kit/components/lab/LabFrame.tsx';
import { fmt } from 'course-kit/i18n.mjs';
import { COURSE, LOG, SEVERITIES, TYPES, blockedBy, hex, maskErrors, produced, runTotals, snippet, type Masks } from '../../lib/labs/messenger-filter';
import { vuid } from '../../lib/vuids';
import { pick } from '../../lib/text';
import { TEXT } from './messenger-filter.text';
import './messenger-filter.css';

const ALL: Masks = { severity: SEVERITIES.reduce((a, s) => a | s.bit, 0), type: TYPES.reduce((a, t) => a | t.bit, 0) };

export default function MessengerFilterView({ lang }: { lang: string }) {
	const t = pick(TEXT, lang, 'messenger-filter');
	const [masks, setMasks] = useState<Masks>(COURSE);
	const [leak, setLeak] = useState(false);
	const [code, setCode] = useState<'c' | 'rust'>('c');

	const lines = produced(masks, leak);
	const got = lines.filter((m) => !blockedBy(m, masks)).length;
	const run = runTotals(masks, leak);
	const errs = maskErrors(masks);
	const flip = (field: keyof Masks, bit: number) => setMasks((m) => ({ ...m, [field]: m[field] ^ bit }));
	const reset = () => {
		setMasks(COURSE);
		setLeak(false);
	};

	const boxes = (field: keyof Masks, list: typeof SEVERITIES | typeof TYPES) => (
		<fieldset className="mf-set">
			<legend>
				<code>{t[field]}</code> = <code>{hex(masks[field])}</code>
			</legend>
			{list.map((x) => (
				<label key={x.name}>
					<input type="checkbox" checked={(masks[field] & x.bit) !== 0} onChange={() => flip(field, x.bit)} />
					<span>
						{x.name} <span className="mf-bit">{hex(x.bit)}</span>
						{x.name === 'DEVICE_ADDRESS_BINDING_EXT' && <span className="mf-bit"> — {t.dabNote}</span>}
					</span>
				</label>
			))}
		</fieldset>
	);

	return (
		<LabFrame lang={lang} title={t.title} onReset={reset}>
			<div className="mf-controls">
				{boxes('severity', SEVERITIES)}
				{boxes('type', TYPES)}
			</div>
			<div className="lab-row mf-row">
				<button type="button" className="lab-button" onClick={() => setMasks(COURSE)}>
					{t.course}
				</button>
				<button type="button" className="lab-button" onClick={() => setMasks(ALL)}>
					{t.all}
				</button>
				<label className="mf-leak">
					<input type="checkbox" checked={leak} onChange={(e) => setLeak(e.currentTarget.checked)} />
					{t.leak}
				</label>
			</div>

			<div className="lab-stats" aria-live="polite">
				<div className="lab-stat">
					<b>
						{got} / {lines.length}
					</b>
					<span>{fmt(t.sample, { got, of: lines.length })}</span>
				</div>
				<div className="lab-stat">
					<b>
						{run.got} / {run.of}
					</b>
					<span>{fmt(t.whole, run)}</span>
				</div>
			</div>

			{errs.length > 0 && (
				<div className="lab-msg is-err">
					<b>{t.maskTitle}</b>
					<ul>
						{errs.map((v) => (
							<li key={v}>
								<code>{v}</code>: {vuid(v)}
							</li>
						))}
					</ul>
				</div>
			)}

			<div>
				<p className="mf-head">{t.logTitle}</p>
				<ol className="mf-log">
					{lines.map((m, i) => {
						const why = blockedBy(m, masks);
						return (
							<li key={i} className={`mf-line is-${m.severity.toLowerCase()}${why ? ' is-dropped' : ''}`}>
								<div className="mf-meta">
									<span className="mf-sev">{m.severity}</span>
									<span className="mf-type">{m.type}</span>
									<code className="mf-id">{m.id}</code>
									<span className="mf-verdict">{why === 'severity' ? t.bySeverity : why === 'type' ? t.byType : t.delivered}</span>
								</div>
								<pre className="mf-text">{m.message}</pre>
							</li>
						);
					})}
				</ol>
			</div>

			<div>
				<div className="mf-code-head">
					<p className="mf-head">{t.codeTitle}</p>
					<div className="lab-row" role="group" aria-label={t.codeTitle}>
						{(['c', 'rust'] as const).map((l) => (
							<button key={l} type="button" className={`lab-button${code === l ? ' is-primary' : ''}`} aria-pressed={code === l} onClick={() => setCode(l)}>
								{l === 'c' ? t.langC : t.langRust}
							</button>
						))}
					</div>
				</div>
				<pre className="mf-code">
					<code>{snippet(masks, code)}</code>
				</pre>
			</div>

			<details className="mf-choice">
				<summary>{t.choiceTitle}</summary>
				<p>{t.choice}</p>
			</details>
			<p className="lab-muted mf-source">{fmt(t.source, { n: LOG.messages.length })}</p>
		</LabFrame>
	);
}
