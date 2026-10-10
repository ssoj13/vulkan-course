/**
 * The Vulkan object tree as a node graph (React island, @xyflow/react).
 *   map:   every object in scope, colored by the course stage that introduces it; click for what it is.
 *   build: objects start as ghosts; the reader creates and destroys them like an app would, and the simulator
 *          (lib/objects.ts) answers with the calls made or with the real validation error.
 * Positions come from the build-time layout in ObjectGraph.astro.
 */
import { useMemo, useState } from 'react';
import { Background, Controls, Handle, MarkerType, Position, ReactFlow, type Edge, type Node, type NodeProps } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import 'course-kit/components/graph/graph.css';
import { Frame } from 'course-kit/components/graph/Frame.tsx';
import { FIT_ZOOM, alignTop } from 'course-kit/lib/fit';
import './objects.css';
import { EMPTY, STAGES, blockers, create, destroy, labelOf, obj, type LogLine, type SimState } from '../lib/objects';

export interface ObjNode {
	id: string;
	x: number;
	y: number;
	w: number;
	h: number;
}

export interface ObjectGraphViewProps {
	mode: 'map' | 'build';
	nodes: ObjNode[];
	/** Parent -> child pairs among the nodes. */
	edges: { from: string; to: string; optional: boolean }[];
	height: number;
	/** term id -> glossary URL (resolved at build time). */
	termHref: Record<string, string>;
}

/** Instance extensions the build exercise offers at vkCreateInstance. */
const DEBUG_UTILS = 'VK_EXT_debug_utils';

type Data = { id: string; w: number; h: number; status: 'map' | 'ghost' | 'alive' | 'dangling' | 'ready' } & Record<string, unknown>;

function ObjNodeView({ data, selected }: NodeProps<Node<Data>>) {
	const o = obj(data.id);
	return (
		<div
			className={`vk-node vk-stage-${STAGES.indexOf(o.stage)} is-${data.status}${selected ? ' is-selected' : ''}`}
			style={{ width: data.w, height: data.h }}
			title={o.note}
		>
			<Handle type="target" position={Position.Top} isConnectable={false} />
			<span>{labelOf(o)}</span>
			<Handle type="source" position={Position.Bottom} isConnectable={false} />
		</div>
	);
}
const nodeTypes = { obj: ObjNodeView };

export default function ObjectGraphView({ mode, nodes, edges, height, termHref }: ObjectGraphViewProps) {
	const [sel, setSel] = useState<string | null>(mode === 'build' ? 'instance' : null);
	const [state, setState] = useState<SimState>(EMPTY);
	const [log, setLog] = useState<LogLine[]>([]);
	const [debugUtils, setDebugUtils] = useState(true);
	const ids = useMemo(() => nodes.map((n) => n.id), [nodes]);

	const status = (id: string): Data['status'] => {
		if (mode === 'map') return 'map';
		if (state.dangling.includes(id)) return 'dangling';
		if (state.alive.includes(id)) return 'alive';
		return blockers(state, id).length ? 'ghost' : 'ready';
	};

	const rfNodes: Node<Data>[] = nodes.map((n) => ({
		id: n.id,
		type: 'obj',
		position: { x: n.x, y: n.y },
		width: n.w,
		height: n.h,
		data: { id: n.id, w: n.w, h: n.h, status: status(n.id) },
		draggable: false,
		connectable: false,
	}));
	const rfEdges: Edge[] = edges.map((e) => ({
		id: `${e.from}-${e.to}`,
		source: e.from,
		target: e.to,
		type: 'smoothstep',
		className: `ck-edge${e.optional ? ' is-muted' : ''}${mode === 'build' && !state.alive.includes(e.to) ? ' is-muted' : ''}`,
		style: e.optional ? { strokeDasharray: '4 4' } : undefined,
		markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
	}));

	const apply = (r: { state: SimState; log: LogLine[] }) => {
		setState(r.state);
		setLog((l) => [...l, ...r.log].slice(-40));
	};
	const reset = () => {
		setState(EMPTY);
		setLog([]);
	};

	const o = sel ? obj(sel) : null;
	const created = ids.filter((id) => state.alive.includes(id)).length;
	const errors = log.filter((l) => l.kind === 'error').length;

	return (
		<div className="ck-graph vk-objects">
			<Frame height={height}>
				<ReactFlow
					nodes={rfNodes}
					edges={rfEdges}
					nodeTypes={nodeTypes}
					fitView
					fitViewOptions={{ padding: 0.1, minZoom: FIT_ZOOM.min, maxZoom: FIT_ZOOM.max }}
					onInit={alignTop}
					minZoom={0.3}
					maxZoom={2}
					zoomOnScroll={false}
					preventScrolling={false}
					nodesDraggable={false}
					nodesConnectable={false}
					onNodeClick={(_, n) => setSel(n.id)}
					proOptions={{ hideAttribution: true }}
				>
					<Background gap={24} size={1} />
					<Controls showInteractive={false} />
				</ReactFlow>
			</Frame>

			<div className="ck-graph-info">
				{o ? (
					<>
						<p className="ck-graph-title">
							<b>{labelOf(o)}</b> <span className="ck-badge">{o.how === 'create' ? 'created' : o.how === 'allocate' ? 'allocated' : o.how === 'enumerate' ? 'enumerated' : 'retrieved'}</span>
						</p>
						<p>{o.note}</p>
						<p className="ck-muted">
							<code>{o.call}</code>
							{o.destroy ? (
								<>
									{' '}
									· released with <code>{o.destroy}</code>
								</>
							) : null}
							{o.parents.length > 0 && <> · made from {o.parents.map((p) => labelOf(obj(p.id)) + (p.optional ? ' (optional)' : '')).join(', ')}</>}
							{o.term && termHref[o.term] ? (
								<>
									{' '}
									· <a href={termHref[o.term]}>glossary</a>
								</>
							) : null}
						</p>
						{mode === 'build' && (
							<div className="lab-row vk-actions">
								{o.id === 'instance' && !state.alive.includes('instance') && (
									<label className="vk-check">
										<input type="checkbox" checked={debugUtils} onChange={(e) => setDebugUtils(e.target.checked)} /> enable {DEBUG_UTILS} + validation layer
									</label>
								)}
								<button type="button" className="lab-button is-primary" onClick={() => apply(create(state, o.id, debugUtils ? [DEBUG_UTILS] : []))}>
									{o.how === 'enumerate' ? 'Enumerate' : o.how === 'get' ? 'Get' : o.how === 'allocate' ? 'Allocate' : 'Create'}
								</button>
								<button type="button" className="lab-button" onClick={() => apply(destroy(state, o.id))}>
									{o.destroy ? (o.how === 'allocate' ? 'Free' : 'Destroy') : 'Destroy?'}
								</button>
							</div>
						)}
					</>
				) : (
					<p className="ck-muted">Click an object to see what it is, how it is made and what it is made from.</p>
				)}
			</div>

			{mode === 'build' && (
				<div className="vk-console">
					<div className="lab-row vk-console-head">
						<span>
							Created {created}/{ids.length} · validation errors: {errors}
						</span>
						<button type="button" className="lab-button" onClick={reset}>
							Reset
						</button>
					</div>
					<ol className="vk-log" aria-live="polite">
						{log.length === 0 && <li className="is-info">Pick an object and press Create. Start where the arrows start.</li>}
						{log.map((l, i) => (
							<li key={i} className={`is-${l.kind}`}>
								{l.text}
							</li>
						))}
					</ol>
				</div>
			)}

			{mode === 'map' && (
				<ul className="ck-legend" aria-label="Legend">
					{STAGES.filter((s) => nodes.some((n) => obj(n.id).stage === s)).map((s) => (
						<li key={s}>
							<span className={`ck-swatch vk-stage-${STAGES.indexOf(s)}`} aria-hidden="true" />
							{s}
						</li>
					))}
				</ul>
			)}
		</div>
	);
}
