/**
 * Queue-planner lab island: the course GPU's real queue families, the roles an app asks queues for, a family-choice
 * strategy, and the VkDeviceQueueCreateInfo list that results, checked against vkCreateDevice's valid-usage rules.
 * All logic lives in lib/labs/queue-planner.ts.
 */
import { LabFrame } from 'course-kit/components/lab/LabFrame.tsx';
import { fmt } from 'course-kit/i18n.mjs';
import { useState } from 'react';
import {
	DEFAULT_OPTIONS,
	DEFAULT_REQS,
	FAMILIES,
	PRIORITY_LEVELS,
	ROLES,
	caps,
	plan,
	type Options,
	type PlanError,
	type Role,
	type RoleReq,
} from '../../lib/labs/queue-planner';
import { pick } from '../../lib/text';
import { vuid } from '../../lib/vuids';
import { TEXT } from './queue-planner.text';
import './queue-planner.css';

type Texts = (typeof TEXT)['en'];
const roleName = (t: Texts, r: Role) => t[`role_${r}`];

/** One error in the reader's language; VUID texts stay verbatim (the spec's English). */
function errorText(t: Texts, e: PlanError): string {
	switch (e.kind) {
		case 'noFamily':
			return fmt(t.errNoFamily, { role: roleName(t, e.role) });
		case 'duplicate':
			return fmt(t.errDuplicate, { family: e.family, infos: e.infos });
		case 'tooMany':
			return fmt(t.errTooMany, { family: e.family, count: e.count, max: e.max });
		case 'priority':
			return fmt(t.errPriority, { family: e.family, value: e.value });
	}
}

/** `1.0` style, the way priorities are written in C/Rust code. */
const num = (p: number) => (Number.isInteger(p) ? p.toFixed(1) : String(p));

export default function QueuePlannerView({ lang }: { lang: string }) {
	const t = pick(TEXT, lang, 'queue-planner');
	const [reqs, setReqs] = useState<Record<Role, RoleReq>>(DEFAULT_REQS);
	const [opt, setOpt] = useState<Options>(DEFAULT_OPTIONS);
	const p = plan(FAMILIES, reqs, opt);
	const setReq = (r: Role, patch: Partial<RoleReq>) => setReqs({ ...reqs, [r]: { ...reqs[r], ...patch } });
	const reset = () => {
		setReqs(DEFAULT_REQS);
		setOpt(DEFAULT_OPTIONS);
	};
	const rolesOf = (family: number) => p.roles.filter((r) => r.family === family).map((r) => roleName(t, r.role));

	return (
		<LabFrame lang={lang} title={t.title} onReset={reset}>
			<h4>{t.families}</h4>
			<div className="vk-qp-scroll">
				<table className="vk-qp-table">
					<thead>
						<tr>
							<th>{t.colFamily}</th>
							<th>{t.colFlags}</th>
							<th>{t.colCount}</th>
							<th>{t.colTs}</th>
							<th>{t.colGran}</th>
							<th>{t.colRoles}</th>
						</tr>
					</thead>
					<tbody>
						{FAMILIES.map((f) => {
							const used = rolesOf(f.index);
							return (
								<tr key={f.index} className={used.length ? 'is-used' : undefined}>
									<td>{f.index}</td>
									<td>
										{f.flags.join(' | ')}
										{caps(f).length > f.flags.length && <span className="lab-muted"> {t.implicitTransfer}</span>}
									</td>
									<td>{f.queueCount}</td>
									<td>{f.timestampValidBits}</td>
									<td>({f.granularity.join(',')})</td>
									<td>{used.join(', ')}</td>
								</tr>
							);
						})}
					</tbody>
				</table>
			</div>
			<p className="lab-muted">{t.famNote}</p>

			<div className="vk-qp-inputs">
				<fieldset className="vk-qp-box">
					<legend>{t.roles}</legend>
					{ROLES.map((r) => (
						<div key={r} className={`vk-qp-role${reqs[r].on ? '' : ' is-off'}`}>
							<label className="vk-qp-check">
								<input type="checkbox" checked={reqs[r].on} onChange={(e) => setReq(r, { on: e.target.checked })} aria-label={fmt(t.roleOn, { role: roleName(t, r) })} />
								<b>{roleName(t, r)}</b>
								<span className="lab-muted">{t[`roleNeeds_${r}`]}</span>
							</label>
							<div className="vk-qp-nums">
								<label>
									<span>{t.queues}</span>
									<input type="number" min={1} max={32} step={1} value={reqs[r].count} disabled={!reqs[r].on} onChange={(e) => setReq(r, { count: Math.max(1, Number(e.target.value) || 1) })} />
								</label>
								<label>
									<span>{t.priority}</span>
									<input type="number" step={0.1} value={reqs[r].priority} disabled={!reqs[r].on} onChange={(e) => setReq(r, { priority: Number(e.target.value) })} />
								</label>
							</div>
						</div>
					))}
				</fieldset>

				<fieldset className="vk-qp-box">
					<legend>{t.strategy}</legend>
					{(['dedicated', 'first'] as const).map((s) => (
						<label key={s} className="vk-qp-check">
							<input type="radio" name="vk-qp-strategy" checked={opt.strategy === s} onChange={() => setOpt({ ...opt, strategy: s })} />
							{s === 'first' ? t.strFirst : t.strDedicated}
						</label>
					))}
					<label className="vk-qp-check">
						<input type="checkbox" checked={opt.share} onChange={(e) => setOpt({ ...opt, share: e.target.checked })} /> {t.share}
					</label>
					<label className="vk-qp-check">
						<input type="checkbox" checked={opt.merge} onChange={(e) => setOpt({ ...opt, merge: e.target.checked })} /> {t.merge}
					</label>
					<p className="lab-muted">{t.presentNote}</p>
					<p className="lab-muted">{fmt(t.priorityNote, { levels: PRIORITY_LEVELS })}</p>
				</fieldset>
			</div>

			<h4>{t.result}</h4>
			<div className="vk-qp-scroll">
				<table className="vk-qp-table">
					<thead>
						<tr>
							<th>{t.colRole}</th>
							<th>{t.colFamily}</th>
							<th>{t.colQueue}</th>
						</tr>
					</thead>
					<tbody>
						{p.roles.map((r) => (
							<tr key={r.role}>
								<td>{roleName(t, r.role)}</td>
								<td>
									{r.family ?? t.noFamily}
									{r.withGraphics && (r.role === 'compute' || r.role === 'transfer') && <span className="vk-qp-warn"> {t.withGraphics}</span>}
								</td>
								<td>
									<code>{r.family === null ? '-' : r.queues.map((q) => `(${r.family}, ${q})`).join(' ')}</code>
								</td>
							</tr>
						))}
					</tbody>
				</table>
			</div>

			<p className="vk-qp-label">
				<code>{t.infos}</code>
			</p>
			<pre className="lab-output vk-qp-infos">
				{p.infos.length
					? p.infos
							.map((i) => `{ queueFamilyIndex: ${i.family}, queueCount: ${i.count}, pQueuePriorities: [${i.priorities.map(num).join(', ')}] }  // ${i.roles.map((r) => roleName(t, r)).join(' + ')}`)
							.join('\n')
					: '[]'}
			</pre>

			<div aria-live="polite">
				{p.errors.length === 0 ? (
					<p className="lab-msg is-ok">{t.ok}</p>
				) : (
					<ul className="vk-qp-errors">
						{p.errors.map((e, i) => (
							<li key={i}>
								<b>{errorText(t, e)}</b>
								{'vuid' in e && (
									<span className="vk-qp-vuid">
										[{e.vuid}] {vuid(e.vuid)}
									</span>
								)}
							</li>
						))}
					</ul>
				)}
			</div>
		</LabFrame>
	);
}
