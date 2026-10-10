/**
 * The Vulkan object tree as a node graph (React island, @xyflow/react), in the course's node-editor look.
 *   map:   every object in scope, colored by the course stage that introduces it; click for what it is.
 *   build: objects start as ghosts; the reader creates and destroys them like an app would, and the simulator
 *          (lib/objects.ts) answers with the calls made or with the real validation error.
 * Positions come from the build-time layout in ObjectGraph.astro (roots on top, course order left to right);
 * nodes can be dragged and the layout reset.
 */
import { useMemo, useState } from 'react';
import { Background, Controls, Handle, MarkerType, Position, ReactFlow, type Edge, type Node, type NodeProps } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import 'course-kit/components/graph/graph.css';
import { Frame } from 'course-kit/components/graph/Frame.tsx';
import { heat, useFlowNodes } from 'course-kit/components/graph/flow.ts';
import { FIT_ZOOM, alignTop } from 'course-kit/lib/fit';
import { fmt } from 'course-kit/i18n.mjs';
import './objects.css';
import { EMPTY, STAGES, blockers, create, destroy, labelOf, noteOf, obj, type LogLine, type SimState } from '../lib/objects';
import { HOW_KEY, SIM_KEY, STAGE_KEY, courseText } from '../i18n';

export interface ObjNode {
	id: string;
	x: number;
	y: number;
	w: number;
	h: number;
}

export interface ObjectGraphViewProps {
	/** Page language and the course's default language (objects.json is written in the default one). */
	lang: string;
	defaultLang: string;
	mode: 'map' | 'build';
	nodes: ObjNode[];
	/** Parent -> child pairs among the nodes. */
	edges: { from: string; to: string; optional: boolean }[];
	height: number;
	/** term id -> glossary URL (resolved at build time). */
	termHref: Record<string, string>;
}

/** Instance extension the build exercise offers at vkCreateInstance. */
const DEBUG_UTILS = 'VK_EXT_debug_utils';

type Status = 'map' | 'ghost' | 'alive' | 'dangling' | 'ready';
type Data = { id: string; w: number; h: number; status: Status; note: string; caption: string } & Record<string, unknown>;

function ObjNodeView({ data, selected }: NodeProps<Node<Data>>) {
	const o = obj(data.id);
	return (
		<div
			className={`ck-node vk-stage-${STAGES.indexOf(o.stage)} is-${data.status}${selected ? ' is-selected' : ''}`}
			style={{ width: data.w, height: data.h }}
			title={data.note}
		>
			<Handle type="target" position={Position.Top} isConnectable={false} />
			<div className="ck-node-head is-code">
				<span>{labelOf(o)}</span>
			</div>
			<div className="ck-node-body">{data.caption}</div>
			<Handle type="source" position={Position.Bottom} isConnectable={false} />
		</div>
	);
}
const nodeTypes = { obj: ObjNodeView };

export default function ObjectGraphView({ lang, defaultLang, mode, nodes, edges, height, termHref }: ObjectGraphViewProps) {
	const t = courseText(lang);
	const how = (id: string) => t[HOW_KEY[obj(id).how]];
	const stage = (id: string) => t[STAGE_KEY[obj(id).stage]];
	/** A console line in the reader's language: API calls and validation reports stay verbatim. */
	const line = (l: LogLine) => ('text' in l ? l.text : fmt(t[SIM_KEY[l.msg]], l.vars));
	const [sel, setSel] = useState<string | null>(mode === 'build' ? 'instance' : null);
	const [state, setState] = useState<SimState>(EMPTY);
	const [log, setLog] = useState<LogLine[]>([]);
	const [debugUtils, setDebugUtils] = useState(true);
	const ids = useMemo(() => nodes.map((n) => n.id), [nodes]);

	const initial = useMemo<Node<Data>[]>(() => {
		const status = (id: string): Status => {
			if (mode === 'map') return 'map';
			if (state.dangling.includes(id)) return 'dangling';
			if (state.alive.includes(id)) return 'alive';
			return blockers(state, id).length ? 'ghost' : 'ready';
		};
		return nodes.map((n) => ({
			id: n.id,
			type: 'obj',
			position: { x: n.x, y: n.y },
			width: n.w,
			height: n.h,
			data: { id: n.id, w: n.w, h: n.h, status: status(n.id), note: noteOf(obj(n.id), lang, defaultLang), caption: `${how(n.id)} · ${stage(n.id)}` },
			connectable: false,
		}));
	}, [nodes, mode, state, lang, defaultLang]);
	const { nodes: rfNodes, onNodesChange, reset } = useFlowNodes(initial);

	const rfEdges = useMemo<Edge[]>(
		() =>
			edges.map((e) => ({
				id: `${e.from}-${e.to}`,
				source: e.from,
				target: e.to,
				type: 'default',
				className: `ck-edge${e.optional || (mode === 'build' && !state.alive.includes(e.to)) ? ' is-muted' : ''}`,
				style: e.optional ? { strokeDasharray: '4 5' } : undefined,
				markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
			})),
		[edges, mode, state],
	);

	const apply = (r: { state: SimState; log: LogLine[] }) => {
		setState(r.state);
		setLog((l) => [...l, ...r.log].slice(-40));
	};
	const restart = () => {
		setState(EMPTY);
		setLog([]);
	};

	const o = sel ? obj(sel) : null;
	const created = ids.filter((id) => state.alive.includes(id)).length;
	const errors = log.filter((l) => l.kind === 'error').length;

	return (
		<div className="ck-graph vk-objects">
			<Frame lang={lang} height={height} onReset={reset}>
				<ReactFlow
					nodes={rfNodes}
					edges={heat(rfEdges, sel)}
					nodeTypes={nodeTypes}
					onNodesChange={onNodesChange}
					fitView
					fitViewOptions={{ padding: 0.1, minZoom: FIT_ZOOM.min, maxZoom: FIT_ZOOM.max }}
					onInit={alignTop}
					minZoom={0.3}
					maxZoom={2}
					zoomOnScroll={false}
					preventScrolling={false}
					nodesConnectable={false}
					onNodeClick={(_, n) => setSel(n.id)}
					proOptions={{ hideAttribution: true }}
				>
					<Background gap={22} size={1.2} />
					<Controls showInteractive={false} />
				</ReactFlow>
			</Frame>

			<div className="ck-graph-info">
				{o ? (
					<>
						<p className="ck-graph-title">
							<b className="is-code">{labelOf(o)}</b> <span className="ck-badge">{how(o.id)}</span>
						</p>
						<p>{noteOf(o, lang, defaultLang)}</p>
						<p className="ck-muted">
							<code>{o.call}</code>
							{o.destroy ? <> · {fmt(t.releasedWith, { call: o.destroy })}</> : null}
							{o.parents.length > 0 && (
								<> · {fmt(t.madeFrom, { parents: o.parents.map((p) => labelOf(obj(p.id)) + (p.optional ? ` ${t.optional}` : '')).join(', ') })}</>
							)}
							{o.term && termHref[o.term] ? (
								<>
									{' '}
									· <a href={termHref[o.term]}>{t.glossary}</a>
								</>
							) : null}
						</p>
						{mode === 'build' && (
							<div className="lab-row vk-actions">
								{o.id === 'instance' && !state.alive.includes('instance') && (
									<label className="vk-check">
										<input type="checkbox" checked={debugUtils} onChange={(e) => setDebugUtils(e.target.checked)} /> {fmt(t.enableExt, { ext: DEBUG_UTILS })}
									</label>
								)}
								<button type="button" className="lab-button is-primary" onClick={() => apply(create(state, o.id, debugUtils ? [DEBUG_UTILS] : []))}>
									{t[o.how]}
								</button>
								<button type="button" className="lab-button" onClick={() => apply(destroy(state, o.id))}>
									{o.destroy ? (o.how === 'allocate' ? t.free : t.destroy) : t.destroyQ}
								</button>
							</div>
						)}
					</>
				) : (
					<p className="ck-muted">{t.hint}</p>
				)}
			</div>

			{mode === 'build' && (
				<div className="vk-console">
					<div className="lab-row vk-console-head">
						<span>{fmt(t.progress, { n: created, total: ids.length, errors })}</span>
						<button type="button" className="lab-button" onClick={restart}>
							{t.startOver}
						</button>
					</div>
					<ol className="vk-log" aria-live="polite">
						{log.length === 0 && <li className="is-info">{t.startHint}</li>}
						{log.map((l, i) => (
							<li key={i} className={`is-${l.kind}`}>
								{line(l)}
							</li>
						))}
					</ol>
				</div>
			)}

			<div className="ck-legend" aria-label="Legend">
				<span className="ck-legend-note">{t.legendNote}</span>
				<ul>
					{STAGES.filter((s) => nodes.some((n) => obj(n.id).stage === s)).map((s) => (
						<li key={s}>
							<span className={`ck-swatch vk-stage-${STAGES.indexOf(s)}`} aria-hidden="true" />
							{t[STAGE_KEY[s]]}
						</li>
					))}
				</ul>
			</div>
		</div>
	);
}
