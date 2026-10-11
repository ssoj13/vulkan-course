/**
 * Buffer-memory lab (React island) over lib/labs/buffer-memory.ts: the reader picks usage flags, a size and where the
 * buffer should live; the flow (create -> requirements -> choose type -> allocate -> bind -> map or staging upload)
 * is a scrubbable timeline, and the heap/type map of the course GPU shows which types memoryTypeBits allows and which
 * one is picked at the current step.
 */
import { useMemo, useState } from 'react';
import { LabFrame } from 'course-kit/components/lab/LabFrame.tsx';
import { Scrubber } from 'course-kit/components/lab/Scrubber.tsx';
import { useTimeline } from 'course-kit/components/lab/useTimeline.ts';
import { fmt } from 'course-kit/i18n.mjs';
import {
	CONSUMER,
	COPY_FAMILIES,
	LIMITS,
	PLACEMENTS,
	REQUESTS,
	STEP_API,
	USAGES,
	VUIDS,
	WANT,
	bitString,
	bytes,
	flow,
	short,
	typesIn,
	type FlowStep,
	type Placement,
	type Usage,
	type Which,
} from '../../lib/labs/buffer-memory';
import { DL, HC, HEAPS, TYPES } from '../../lib/labs/memory';
import { vuid } from '../../lib/vuids';
import reqJson from '../../data/buffer-requirements.json';
import { pick } from '../../lib/text';
import { TEXT } from './buffer-memory.text';
import './buffer-memory.css';

type T = (typeof TEXT)['en'];

const DMA = COPY_FAMILIES.find((f) => f.dedicated);
const BAR_HEAP = HEAPS.find((h) => TYPES.some((t) => t.heapIndex === h.index && t.flags.includes(DL) && t.flags.length > 1));
const flags = (fs: readonly string[]) => fs.map(short).join(' | ');

/** Caption of a flow step in the reader's language, plus the call it makes. */
export function stepText(t: T, s: FlowStep): string {
	const which = (w: Which) => (w === 'dst' ? t.whichDst : t.whichStaging);
	switch (s.kind) {
		case 'create':
			return fmt(t['step.create'], { which: which(s.which), request: bytes(s.request), usage: flags(s.usage) });
		case 'reqs': {
			const parts = [
				fmt(t['step.reqs'], { which: which(s.which), size: bytes(s.req.size), alignment: s.req.alignment, bits: `0b${bitString(s.req.memoryTypeBits)}`, types: typesIn(s.req.memoryTypeBits).join(', ') }),
				t.reqsRound,
			];
			if (!typesIn(s.req.memoryTypeBits).includes(2)) parts.push(t.reqsType2);
			return parts.join(' ');
		}
		case 'choose': {
			const w = WANT[s.placement];
			const picked = TYPES[s.choice.picked!]!;
			const parts = [fmt(t['step.choose'], { which: which(s.which), flags: flags(s.choice.optimal ? w.optimal : w.required), type: picked.index, heap: picked.heapIndex })];
			if (!s.choice.optimal) parts.push(fmt(t.chooseFallback, { optimal: flags(w.optimal), required: flags(w.required) }));
			parts.push(fmt(t.chooseRule, { vuid: VUIDS.bindType }));
			return parts.join(' ');
		}
		case 'allocate':
			return fmt(t['step.allocate'], { size: bytes(s.size), type: s.type.index, heap: s.type.heapIndex, heapSize: bytes(s.heapSize) });
		case 'bind':
			return fmt(t['step.bind'], { which: which(s.which), alignment: s.alignment, next: bytes(s.size) });
		case 'map': {
			const parts = [fmt(t['step.map'], { which: which(s.which), mapAlign: LIMITS.minMemoryMapAlignment, request: bytes(s.request) })];
			if (s.type.flags.includes(HC)) parts.push(t.mapCoherent);
			parts.push(s.type.flags.includes(DL) ? t.mapVram : t.mapHost);
			return parts.join(' ');
		}
		case 'copy':
			return s.ok
				? fmt(t['step.copy'], { request: bytes(s.request), dma: DMA?.index ?? '-', count: DMA?.count ?? 0 })
				: fmt(t.copyError, { vuid: VUIDS.copyDst, text: vuid(VUIDS.copyDst) });
		case 'sync': {
			const c = s.consumers.map((u) => CONSUMER[u]!);
			const list = c.length ? c.map((x) => `${x.stages.map((st) => st.replace('VK_PIPELINE_STAGE_2_', '')).join(' | ')} / ${x.access.replace('VK_ACCESS_2_', '')}`).join(', ') : t.consumersNone;
			return fmt(t['step.sync'], { consumers: list });
		}
	}
}

/** The Vulkan call a step makes, shown as code. */
function call(s: FlowStep): string {
	const buf = (w: Which) => (w === 'dst' ? 'buffer' : 'staging');
	switch (s.kind) {
		case 'create':
			return `vkCreateBuffer(device, { size = ${s.request}, usage = ${flags(s.usage)}, sharingMode = EXCLUSIVE }) -> ${buf(s.which)}`;
		case 'reqs':
			return `vkGetBufferMemoryRequirements(device, ${buf(s.which)}) -> { size = ${s.req.size}, alignment = ${s.req.alignment}, memoryTypeBits = 0b${bitString(s.req.memoryTypeBits)} }`;
		case 'choose':
			// The spec's sample loop (chapter "Device Memory"), first with the optimal flags, then with the required ones.
			return `findProperties(&memoryProperties, 0b${bitString(s.bits)}, ${flags(s.choice.optimal ? WANT[s.placement].optimal : WANT[s.placement].required)}) -> ${s.choice.picked}`;
		case 'allocate':
			return `vkAllocateMemory(device, { allocationSize = ${s.size}, memoryTypeIndex = ${s.type.index} }) -> ${s.which === 'dst' ? 'memory' : 'stagingMemory'}`;
		case 'bind':
			return `vkBindBufferMemory(device, ${buf(s.which)}, ${s.which === 'dst' ? 'memory' : 'stagingMemory'}, 0)`;
		case 'map':
			return `vkMapMemory(device, ${s.which === 'dst' ? 'memory' : 'stagingMemory'}, 0, VK_WHOLE_SIZE, 0, &ptr); memcpy(ptr, data, ${s.request})`;
		case 'copy':
			return `vkCmdCopyBuffer(cmd, staging, buffer, 1, { srcOffset = 0, dstOffset = 0, size = ${s.request} })`;
		case 'sync':
			return 'vkCmdPipelineBarrier2(cmd, { COPY / TRANSFER_WRITE -> consumer stage / access })';
	}
}

/** The type the map highlights at a step: the picked (or allocated) type of the step's buffer. */
function typeAt(steps: readonly FlowStep[], i: number): { bits: number | null; picked: number | null } {
	let bits: number | null = null;
	let picked: number | null = null;
	for (const s of steps.slice(0, i + 1)) {
		if (s.kind === 'create') {
			bits = null;
			picked = null;
		} else if (s.kind === 'choose') {
			bits = s.bits;
			picked = s.choice.picked;
		} else if (s.kind === 'reqs') bits = s.req.memoryTypeBits;
	}
	return { bits, picked };
}

function HeapMap({ t, bits, picked }: { t: T; bits: number | null; picked: number | null }) {
	return (
		<div className="bm-heaps" role="group" aria-label={t.heapsLabel}>
			{HEAPS.map((h) => (
				<div key={h.index} className={`bm-heap${h.flags.length ? ' is-device' : ''}`}>
					<p className="bm-heap-head">
						<b>{fmt(t.heap, { i: h.index, size: bytes(h.size) })}</b> <span className="lab-muted">{h.flags.length ? flags(h.flags) : t.noFlags}</span>
					</p>
					<ul className="bm-types">
						{TYPES.filter((ty) => ty.heapIndex === h.index).map((ty) => {
							const state = bits === null ? '' : bits & (1 << ty.index) ? ' is-allowed' : ' is-blocked';
							return (
								<li key={ty.index} className={`bm-type${state}${picked === ty.index ? ' is-picked' : ''}`}>
									<span className="bm-type-n">{fmt(t.type, { i: ty.index })}</span>
									<span className="bm-type-flags">{ty.flags.length ? flags(ty.flags) : t.noFlags}</span>
								</li>
							);
						})}
					</ul>
				</div>
			))}
		</div>
	);
}

export default function BufferMemoryView({ lang }: { lang: string }) {
	const t = pick(TEXT, lang, 'buffer-memory');
	const [usage, setUsage] = useState<readonly Usage[]>(['VK_BUFFER_USAGE_VERTEX_BUFFER_BIT', 'VK_BUFFER_USAGE_TRANSFER_DST_BIT']);
	const [request, setRequest] = useState(REQUESTS[3]!);
	const [placement, setPlacement] = useState<Placement>('device');
	const steps = useMemo(() => flow(usage, request, placement), [usage, request, placement]);
	const tl = useTimeline(steps);
	const s = tl.frame;
	const { bits, picked } = typeAt(steps, tl.index);
	const deviceOnly = steps.find((x) => x.kind === 'allocate' && x.which === 'dst');

	const toggle = (u: Usage) => setUsage((cur) => (cur.includes(u) ? (cur.length > 1 ? cur.filter((x) => x !== u) : cur) : USAGES.filter((x) => x === u || cur.includes(x))));
	const reset = () => {
		setUsage(['VK_BUFFER_USAGE_VERTEX_BUFFER_BIT', 'VK_BUFFER_USAGE_TRANSFER_DST_BIT']);
		setRequest(REQUESTS[3]!);
		setPlacement('device');
		tl.reset();
	};
	const isError = s.kind === 'copy' && !s.ok;

	return (
		<LabFrame lang={lang} title={t.title} onReset={reset}>
			<div className="bm-controls">
				<fieldset className="bm-group">
					<legend>{t.usageLabel}</legend>
					{USAGES.map((u) => {
						const on = usage.includes(u);
						const last = on && usage.length === 1;
						return (
							<label key={u} className={`bm-check${on ? ' is-on' : ''}`} title={last ? t.usageHint : undefined}>
								<input type="checkbox" checked={on} aria-disabled={last} onChange={() => toggle(u)} />
								<code>{short(u)}</code>
							</label>
						);
					})}
				</fieldset>
				<div className="bm-group">
					<label className="lab-field">
						<span>{t.sizeLabel}</span>
						<select value={request} onChange={(e) => setRequest(Number(e.currentTarget.value))}>
							{REQUESTS.map((r) => (
								<option key={r} value={r}>
									{bytes(r)}
								</option>
							))}
						</select>
					</label>
					<fieldset className="bm-group">
						<legend>{t.placeLabel}</legend>
						{PLACEMENTS.map((p) => (
							<label key={p} className={`bm-check${p === placement ? ' is-on' : ''}`}>
								<input type="radio" name="bm-place" checked={p === placement} onChange={() => setPlacement(p)} />
								<span>
									{t[`place.${p}`]}
									<span className="bm-want">{fmt(t.want, { optimal: flags(WANT[p].optimal), required: flags(WANT[p].required) })}</span>
								</span>
							</label>
						))}
					</fieldset>
				</div>
			</div>

			<div className="bm-main">
				<div className="bm-flow">
					<ol className="bm-steps" aria-hidden="true">
						{steps.map((x, i) => (
							<li key={i} className={`${i === tl.index ? 'is-current' : i < tl.index ? 'is-done' : ''}${x.kind === 'copy' && !x.ok ? ' is-error' : ''}${'which' in x && x.which === 'staging' ? ' is-staging' : ''}`}>
								{STEP_API[x.kind]}
							</li>
						))}
					</ol>
					<pre className={`bm-call${isError ? ' is-error' : ''}`}>
						<code>{call(s)}</code>
					</pre>
					{s.kind === 'create' && s.which === 'staging' && deviceOnly?.kind === 'allocate' && (
						<p className="lab-msg">{fmt(t.mapDeviceNote, { type: deviceOnly.type.index, vuid: VUIDS.mapHostVisible })}</p>
					)}
				</div>
				<div className="bm-side">
					<h4>{t.heapsTitle}</h4>
					<HeapMap t={t} bits={bits} picked={picked} />
					<ul className="bm-legend">
						<li>
							<span className="bm-key is-allowed" /> {t.legendAllowed}
						</li>
						<li>
							<span className="bm-key is-blocked" /> {t.legendBlocked}
						</li>
						<li>
							<span className="bm-key is-picked" /> {t.legendPicked}
						</li>
					</ul>
					{BAR_HEAP && <p className="lab-muted">{fmt(t.barNote, { size: bytes(BAR_HEAP.size) })}</p>}
				</div>
			</div>
			<Scrubber lang={lang} tl={tl} caption={<span className={isError ? 'bm-err' : undefined}>{stepText(t, s)}</span>} />

			<details className="bm-facts">
				<summary>{t.factsTitle}</summary>
				<ul>
					<li>{fmt(t.factAlign, { ubo: LIMITS.minUniformBufferOffsetAlignment })}</li>
					<li>{fmt(t.factMap, { map: LIMITS.minMemoryMapAlignment })}</li>
					<li>{fmt(t.factAtom, { atom: LIMITS.nonCoherentAtomSize, vuid: VUIDS.flushAtom })}</li>
					<li>{fmt(t.factGran, { gran: LIMITS.bufferImageGranularity })}</li>
				</ul>
			</details>
			<p className="lab-muted">{fmt(t.captureNote, { device: reqJson.meta.device })}</p>
		</LabFrame>
	);
}
