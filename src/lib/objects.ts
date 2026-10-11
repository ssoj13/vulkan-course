/**
 * Vulkan object model for the ObjectGraph widget: which objects exist, what each is created from, how it is
 * obtained and destroyed (src/data/objects.json), plus a pure simulator of creating/destroying them under the
 * spec's rules. Validation messages quote real valid-usage texts (src/data/vuids.json, synced from the SDK by
 * scripts/sync-vuids.ts). Pure module: no DOM, no React, tested with `bun test`.
 */
import objectsJson from '../data/objects.json';
import objectsRu from '../data/objects.ru.json';
import { hasVuid, vuid } from './vuids';

/** Curriculum stages in course order; the map colors objects by the stage that introduces them. */
export const STAGES = ['instance', 'devices', 'presentation', 'pipeline', 'commands', 'sync', 'memory', 'descriptors', 'images'] as const;
export type Stage = (typeof STAGES)[number];

/** How a handle is obtained: only `create` and `allocate` make objects the application must destroy/free. */
export type How = 'create' | 'allocate' | 'enumerate' | 'get';

export interface ParentRef {
	/** Object id this one is made from. */
	id: string;
	/** The call parameter that carries it (`device`, `pCreateInfo->image`). */
	param: string;
	/** VUID quoted when the parent is missing. */
	vuid: string;
	/** True when the API accepts none (e.g. a pipeline layout with zero descriptor set layouts). */
	optional?: boolean;
	/**
	 * True when the child keeps no reference to this parent after creation, so destroying the parent later leaves the
	 * child valid (the spec: "A shader module can be destroyed while pipelines created using its shaders are still in
	 * use").
	 */
	createOnly?: boolean;
}

export interface VkObject {
	id: string;
	handle: string;
	/** Display label when several objects share a handle type (`VkImage (swapchain)`). */
	label?: string;
	/** terms.json id, for the glossary link. */
	term?: string;
	stage: Stage;
	how: How;
	call: string;
	destroy?: string;
	parents: ParentRef[];
	/** Destroying with live children: `block` = invalid (destroyVuid), `free` = children are released with it. */
	onDestroy?: 'block' | 'free';
	destroyVuid?: string;
	/** This object also satisfies parents of that id (a swapchain image is an image). */
	isA?: string;
	/** Instance extension that must be enabled before this object can be created. */
	requires?: string;
	note: string;
}

export const OBJECTS: readonly VkObject[] = (objectsJson as { objects: VkObject[] }).objects;
const BY_ID = new Map(OBJECTS.map((o) => [o.id, o]));

export function obj(id: string): VkObject {
	const o = BY_ID.get(id);
	if (!o) throw new Error(`objects.json has no object "${id}"`);
	return o;
}

export const labelOf = (o: VkObject): string => o.label ?? o.handle;

/**
 * Translations of the object notes per language (objects.<lang>.json, keyed by object id). objects.json itself is
 * the course's default language. Handles, calls and VUIDs are API names and stay as they are.
 */
export const NOTE_OVERLAYS: Readonly<Record<string, Record<string, { note: string }>>> = { ru: objectsRu };

/** An object's note in a language: the default language reads objects.json, the others their overlay. */
export function noteOf(o: VkObject, lang: string, defaultLang: string): string {
	if (lang === defaultLang) return o.note;
	const n = NOTE_OVERLAYS[lang]?.[o.id]?.note;
	if (!n) throw new Error(`objects.${lang}.json has no note for "${o.id}"`);
	return n;
}

/** Every overlay must translate exactly the objects of objects.json (empty list = complete). */
export function overlayErrors(): string[] {
	const errs: string[] = [];
	const ids = new Set(OBJECTS.map((o) => o.id));
	for (const [lang, notes] of Object.entries(NOTE_OVERLAYS)) {
		for (const id of ids) if (!notes[id]?.note?.trim()) errs.push(`objects.${lang}.json: missing note for "${id}"`);
		for (const id of Object.keys(notes)) if (!ids.has(id)) errs.push(`objects.${lang}.json: unknown object "${id}"`);
	}
	return errs;
}

/** Structural problems of objects.json (empty = valid); `termIds` are the ids of terms.json. */
export function objectErrors(termIds: ReadonlySet<string>): string[] {
	const errs: string[] = [];
	const ids = new Set<string>();
	for (const o of OBJECTS) {
		if (ids.has(o.id)) errs.push(`${o.id}: duplicate id`);
		ids.add(o.id);
	}
	for (const o of OBJECTS) {
		if (!STAGES.includes(o.stage)) errs.push(`${o.id}: unknown stage "${o.stage}"`);
		if (o.term && !termIds.has(o.term)) errs.push(`${o.id}: unknown term "${o.term}"`);
		if (o.isA && !ids.has(o.isA)) errs.push(`${o.id}: isA "${o.isA}" is not an object`);
		// Created objects are always destroyed by the app; allocated ones may instead go back with their pool
		// (descriptor sets); enumerated/retrieved handles are never destroyed by the app.
		if (o.how === 'create' && !o.destroy) errs.push(`${o.id}: created objects need \`destroy\``);
		if ((o.how === 'enumerate' || o.how === 'get') && o.destroy) errs.push(`${o.id}: ${o.how} handles must not have \`destroy\``);
		if (o.onDestroy === 'block' && !o.destroyVuid) errs.push(`${o.id}: onDestroy "block" needs destroyVuid`);
		for (const v of [o.destroyVuid, ...o.parents.map((p) => p.vuid)]) if (v && !hasVuid(v)) errs.push(`${o.id}: ${v} missing from vuids.json (run scripts/sync-vuids.ts)`);
		for (const p of o.parents) if (!ids.has(p.id)) errs.push(`${o.id}: unknown parent "${p.id}"`);
	}
	return errs;
}

/**
 * Map order of the stages (legend, spine sections, siblings): what an application creates, roughly in dependency
 * order - resources before the pipelines that read them, commands and sync last. Same set as STAGES.
 */
export const OBJ_GROUPS: readonly Stage[] = ['instance', 'devices', 'presentation', 'memory', 'images', 'pipeline', 'descriptors', 'commands', 'sync'];

/**
 * Parents of `id` as drawn in a map of `scope`, in objects.json order (the first is the primary parent). A parent
 * outside the scope is replaced by the in-scope objects that are a kind of it (`isA`), the rule the simulator applies:
 * in a presentation exercise without `image`, the swapchain images are what the image views are made from.
 */
export function scopedParents(id: string, scope: readonly string[]): ParentRef[] {
	const out: ParentRef[] = [];
	for (const p of obj(id).parents) {
		const ids = scope.includes(p.id) ? [p.id] : scope.filter((s) => obj(s).isA === p.id);
		for (const s of ids) if (!out.some((x) => x.id === s)) out.push({ ...p, id: s });
	}
	return out;
}

/** Objects whose (non-optional or optional) parent list names `id`. */
export const childrenOf = (id: string): VkObject[] => OBJECTS.filter((o) => o.parents.some((p) => p.id === id));

// ---------------------------------------------------------------------------------------------------------------
// Simulator: one live object per kind (enough to learn the creation order), instance extensions as a flag set.

export interface SimState {
	/** Live object ids in creation order. */
	alive: readonly string[];
	/** Live objects whose parent was destroyed: their handles are dangling, any use is invalid. */
	dangling: readonly string[];
	/** Instance extensions enabled at vkCreateInstance. */
	extensions: readonly string[];
}

export type LogKind = 'call' | 'error' | 'warn' | 'info';

/** Simulator remarks; the view words them in the reader's language (src/i18n.ts `sim*` texts). */
export type SimMsg = 'alreadyExists' | 'notLoaded' | 'doesNotExist' | 'backToPool' | 'notDestroyed' | 'stillAlive' | 'released' | 'dangling';

/**
 * One console line. API calls and validation-layer reports are verbatim `text` (always English, exactly like the
 * real tools print them); the simulator's own remarks are a `msg` with values, worded by the view.
 */
export type LogLine = { kind: LogKind; text: string; vuid?: string } | { kind: LogKind; msg: SimMsg; vars: Record<string, string> };

export interface Step {
	state: SimState;
	log: LogLine[];
}

export const EMPTY: SimState = { alive: [], dangling: [], extensions: [] };

/** The live object that satisfies a parent reference (`isA` counts), if any. */
function liveFor(state: SimState, parentId: string): string | undefined {
	return state.alive.find((a) => a === parentId || BY_ID.get(a)?.isA === parentId);
}

/** Validation-layer style message quoting the spec's valid-usage text. */
function vuError(id: string, call: string): LogLine {
	return { kind: 'error', vuid: id, text: `Validation Error: [ ${id} ] ${call}(): The Vulkan spec states: ${vuid(id)}` };
}

/** Why `id` cannot be created now (missing parents / extension), or null when it can. */
export function blockers(state: SimState, id: string): LogLine[] {
	const o = obj(id);
	const out: LogLine[] = [];
	if (state.alive.includes(id)) out.push({ kind: 'info', msg: 'alreadyExists', vars: { obj: labelOf(o) } });
	for (const p of o.parents) if (!p.optional && !liveFor(state, p.id)) out.push(vuError(p.vuid, o.call));
	if (o.requires && state.alive.includes('instance') && !state.extensions.includes(o.requires)) {
		out.push({ kind: 'error', msg: 'notLoaded', vars: { call: o.call, ext: o.requires } });
	}
	return out;
}

/** Creates (or enumerates / gets) object `id`. `extensions` apply to the instance only. */
export function create(state: SimState, id: string, extensions: readonly string[] = []): Step {
	const o = obj(id);
	const errs = blockers(state, id);
	if (errs.length) return { state, log: errs };
	const args = o.parents.map((p) => `${p.param}=${liveFor(state, p.id) ? labelOf(obj(liveFor(state, p.id)!)) : 'none'}`).join(', ');
	const result = o.how === 'create' || o.how === 'allocate' ? ' -> VK_SUCCESS' : '';
	return {
		state: { ...state, alive: [...state.alive, id], extensions: id === 'instance' ? [...extensions] : state.extensions },
		log: [{ kind: 'call', text: `${o.call}(${args})${result}` }],
	};
}

/** All live objects that (transitively) hang off `id`; a `createOnly` parent does not hold its children. */
function liveDescendants(state: SimState, id: string): string[] {
	const out: string[] = [];
	const stack = [id];
	while (stack.length) {
		const cur = stack.pop()!;
		for (const a of state.alive) {
			if (out.includes(a) || a === id) continue;
			if (obj(a).parents.some((p) => !p.createOnly && (p.id === cur || obj(cur).isA === p.id))) {
				out.push(a);
				stack.push(a);
			}
		}
	}
	return out;
}

/** Destroys / frees object `id` under the spec's rules. */
export function destroy(state: SimState, id: string): Step {
	const o = obj(id);
	if (!state.alive.includes(id)) return { state, log: [{ kind: 'info', msg: 'doesNotExist', vars: { obj: labelOf(o) } }] };
	if (!o.destroy) {
		const owner = labelOf(obj(o.parents[0]!.id));
		const msg: SimMsg = o.how === 'allocate' ? 'backToPool' : 'notDestroyed';
		return { state, log: [{ kind: 'info', msg, vars: { obj: labelOf(o), owner, call: o.call } }] };
	}
	const deps = liveDescendants(state, id);
	// Only direct children made by the application count for "block": enumerated/retrieved handles go away with the parent.
	const owned = deps.filter((d) => obj(d).destroy);
	if (o.onDestroy === 'block' && owned.length) {
		return {
			state,
			log: [vuError(o.destroyVuid!, o.destroy), { kind: 'info', msg: 'stillAlive', vars: { objs: owned.map((d) => labelOf(obj(d))).join(', ') } }],
		};
	}
	const log: LogLine[] = [{ kind: 'call', text: `${o.destroy}(${labelOf(o)})` }];
	let alive = state.alive.filter((a) => a !== id);
	let dangling = state.dangling.filter((d) => d !== id);
	// Released with the parent: handles the app never destroys itself, and for `free` parents (pools, swapchain)
	// their direct children. Deeper objects with their own destroy call (a view of a swapchain image) stay alive
	// but dangling.
	const direct = (d: string) => obj(d).parents.some((p) => p.id === id);
	const released = deps.filter((d) => !obj(d).destroy || (o.onDestroy === 'free' && direct(d)));
	if (released.length) {
		alive = alive.filter((a) => !released.includes(a));
		dangling = dangling.filter((d) => !released.includes(d));
		log.push({ kind: 'info', msg: 'released', vars: { objs: released.map((d) => labelOf(obj(d))).join(', ') } });
	}
	const orphans = deps.filter((d) => !released.includes(d));
	if (orphans.length) {
		dangling = [...dangling, ...orphans.filter((d) => !dangling.includes(d))];
		log.push({ kind: 'warn', msg: 'dangling', vars: { objs: orphans.map((d) => labelOf(obj(d))).join(', ') } });
	}
	return { state: { ...state, alive, dangling, extensions: id === 'instance' ? [] : state.extensions }, log };
}
