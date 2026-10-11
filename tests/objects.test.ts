import { describe, expect, test } from 'bun:test';
import termsJson from '../src/data/terms.json';
import { EMPTY, OBJECTS, OBJ_GROUPS, STAGES, create, destroy, objectErrors, overlayErrors, scopedParents, type LogLine, type SimState, type Step } from '../src/lib/objects';

/** Readable form of a console line for failure messages: verbatim text, or the remark key with its values. */
const show = (l: LogLine) => ('text' in l ? l.text : `${l.msg} ${JSON.stringify(l.vars)}`);

/** Applies steps in order, failing the test if an intermediate one is rejected. */
function run(...steps: ((s: SimState) => Step)[]): SimState {
	return steps.reduce((s, f) => {
		const r = f(s);
		const err = r.log.find((l) => l.kind === 'error');
		if (err) throw new Error(show(err));
		return r.state;
	}, EMPTY);
}
const mk = (id: string, ext: string[] = []) => (s: SimState) => create(s, id, ext);
const rm = (id: string) => (s: SimState) => destroy(s, id);

test('objects.json is consistent with terms.json and vuids.json, and every translation is complete', () => {
	expect(objectErrors(new Set(termsJson.terms.map((t) => t.id)))).toEqual([]);
	expect(overlayErrors()).toEqual([]);
	expect(OBJECTS.length).toBeGreaterThan(20);
});

describe('creation order', () => {
	test('a device needs a physical device: the parameter VUID is quoted', () => {
		const r = create(run(mk('instance')), 'device');
		const l = r.log[0]!;
		expect('text' in l && l.vuid).toBe('VUID-vkCreateDevice-physicalDevice-parameter');
		expect(show(l)).toContain('physicalDevice must be a valid VkPhysicalDevice handle');
	});
	test('the debug messenger needs VK_EXT_debug_utils enabled on the instance', () => {
		const r = create(run(mk('instance')), 'debug-messenger');
		expect(r.log[0]).toEqual({ kind: 'error', msg: 'notLoaded', vars: { call: 'vkCreateDebugUtilsMessengerEXT', ext: 'VK_EXT_debug_utils' } });
		expect(() => run(mk('instance', ['VK_EXT_debug_utils']), mk('debug-messenger'))).not.toThrow();
	});
	test('the full first section chain works', () => {
		const s = run(mk('instance', ['VK_EXT_debug_utils']), mk('debug-messenger'), mk('physical-device'), mk('device'), mk('queue'));
		expect(s.alive).toEqual(['instance', 'debug-messenger', 'physical-device', 'device', 'queue']);
	});
	test('a swapchain image satisfies an image-view parent (isA)', () => {
		const s = run(mk('instance'), mk('surface'), mk('physical-device'), mk('device'), mk('swapchain'), mk('swapchain-image'), mk('image-view'));
		expect(s.alive).toContain('image-view');
	});
});

describe('destruction order', () => {
	const full = () => run(mk('instance', ['VK_EXT_debug_utils']), mk('debug-messenger'), mk('physical-device'), mk('device'), mk('queue'));
	test('destroying the instance with live children is the VUID-00629 error', () => {
		const r = destroy(full(), 'instance');
		const l = r.log[0]!;
		expect('text' in l && l.vuid).toBe('VUID-vkDestroyInstance-instance-00629');
		expect(r.state.alive).toContain('instance');
	});
	test('queues and physical devices are not destroyed by the app', () => {
		expect(destroy(full(), 'queue').log[0]).toMatchObject({ msg: 'notDestroyed', vars: { obj: 'VkQueue', owner: 'VkDevice' } });
	});
	test('correct reverse order releases retrieved handles with their parent', () => {
		const s = run(() => ({ state: full(), log: [] }), rm('device'), rm('debug-messenger'), rm('instance'));
		expect(s.alive).toEqual([]);
	});
	test('destroying a swapchain releases its images but leaves their views dangling', () => {
		const s = run(mk('instance'), mk('surface'), mk('physical-device'), mk('device'), mk('swapchain'), mk('swapchain-image'), mk('image-view'), rm('swapchain'));
		expect(s.alive).not.toContain('swapchain-image');
		expect(s.dangling).toEqual(['image-view']);
	});
	test('destroying a command pool frees its command buffers', () => {
		const s = run(mk('instance'), mk('physical-device'), mk('device'), mk('command-pool'), mk('command-buffer'), rm('command-pool'));
		expect(s.alive).not.toContain('command-buffer');
		expect(s.dangling).toEqual([]);
	});
	test('a shader module may go as soon as its pipeline exists (createOnly parent)', () => {
		const s = run(mk('instance'), mk('physical-device'), mk('device'), mk('shader-module'), mk('pipeline-layout'), mk('pipeline'), rm('shader-module'));
		expect(s.alive).toContain('pipeline');
		expect(s.dangling).toEqual([]);
	});
	test('a surface cannot go before its swapchain', () => {
		const s = run(mk('instance'), mk('surface'), mk('physical-device'), mk('device'), mk('swapchain'));
		const l = destroy(s, 'surface').log[0]!;
		expect('text' in l && l.vuid).toBe('VUID-vkDestroySurfaceKHR-surface-01266');
	});
});

describe('map parents', () => {
	const all = OBJECTS.map((o) => o.id);
	const ids = (id: string, scope = all) => scopedParents(id, scope).map((p) => p.id);
	test('the map groups are exactly the stages', () => {
		expect([...OBJ_GROUPS].sort()).toEqual([...STAGES].sort());
	});
	test('second parents stay, in objects.json order (first = primary)', () => {
		expect(ids('descriptor-set')).toEqual(['descriptor-pool', 'descriptor-set-layout']);
		expect(ids('pipeline')).toEqual(['device', 'pipeline-layout', 'shader-module']);
		expect(ids('swapchain')).toEqual(['device', 'surface']);
		expect(ids('image-view')).toEqual(['device', 'image']);
	});
	test('a parent outside the scope is replaced by what is a kind of it (isA)', () => {
		const scope = ['instance', 'surface', 'physical-device', 'device', 'swapchain', 'swapchain-image', 'image-view'];
		expect(ids('image-view', scope)).toEqual(['device', 'swapchain-image']);
		expect(ids('swapchain', ['device', 'swapchain'])).toEqual(['device']);
	});
	test('createOnly and optional flags travel with the parent', () => {
		const p = scopedParents('pipeline', all).find((x) => x.id === 'shader-module');
		expect(p?.createOnly).toBe(true);
		expect(scopedParents('pipeline-layout', all).find((x) => x.id === 'descriptor-set-layout')?.optional).toBe(true);
	});
});