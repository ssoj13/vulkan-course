import { describe, expect, test } from 'bun:test';
import { StaleAct, act, begin, complete, consistency, counts, decode, encode, replay, undo, why, type Act, type Run } from 'course-kit/lib/trainer.ts';
import { OBJECTS, type SimState } from '../src/lib/objects';
import { DEBUG_UTILS, objectScenario, ruleVersion } from '../src/lib/object-trainer';

/** The exercise of lessons devices/logical-device: the first section's chain. */
const SCOPE = ['instance', 'debug-messenger', 'physical-device', 'device', 'queue'];
const V = 'test';
const sc = objectScenario(SCOPE, 'en', V);
const play = (...acts: Act[]): Run<SimState> => acts.reduce((r, a) => act(sc, r, a), begin(sc));
const mk = (item: string, debugUtils?: boolean): Act => {
	const verb = OBJECTS.find((o) => o.id === item)!.how;
	return debugUtils === undefined ? { item, verb } : { item, verb, opts: { debugUtils } };
};
const rm = (item: string): Act => ({ item, verb: 'destroy' });

describe('ObjectGraph build exercise on the engine trainer', () => {
	test('the instance offers VK_EXT_debug_utils (on by default) only while it does not exist', () => {
		const opt = (r: Run<SimState>) => sc.verbs(r.state, 'instance')[0]!.options;
		expect(opt(begin(sc))).toEqual([{ kind: 'flag', id: 'debugUtils', label: `enable ${DEBUG_UTILS} + validation layer`, initial: true }]);
		const r = play(mk('instance'));
		expect(r.state.extensions).toEqual([DEBUG_UTILS]);
		expect(opt(r)).toBeUndefined();
	});
	test('a refused create quotes the VUID verbatim and counts as a validation error', () => {
		const r = play(mk('instance'), mk('device'));
		expect(r.last?.ok).toBe(false);
		expect(r.last?.log[0]).toMatchObject({ kind: 'error', ref: 'VUID-vkCreateDevice-physicalDevice-parameter' });
		expect(r.last?.log[0]!.text).toContain('Validation Error: [ VUID-vkCreateDevice-physicalDevice-parameter ] vkCreateDevice(): The Vulkan spec states:');
		expect(counts(sc, r).errors).toBe(1);
	});
	test('without the extension the messenger is refused with the worded remark', () => {
		const r = play(mk('instance', false), mk('debug-messenger'));
		expect(r.last?.log).toEqual([{ kind: 'error', text: 'vkCreateDebugUtilsMessengerEXT is not available: its function pointer is only loaded when VK_EXT_debug_utils was enabled in vkCreateInstance.' }]);
		const ru = act(objectScenario(SCOPE, 'ru', V), begin(objectScenario(SCOPE, 'ru', V)), mk('instance', false));
		expect(act(objectScenario(SCOPE, 'ru', V), ru, mk('debug-messenger')).last?.log[0]!.text).toStartWith('vkCreateDebugUtilsMessengerEXT недоступна');
	});
	test('statuses follow the simulator: locked, ready, done, and error for a dangling view', () => {
		const st = (r: Run<SimState>) => Object.fromEntries(SCOPE.map((i) => [i, sc.status(r.state, i)]));
		expect(st(play(mk('instance')))).toEqual({ instance: 'done', 'debug-messenger': 'ready', 'physical-device': 'ready', device: 'locked', queue: 'locked' });
		const pres = ['instance', 'surface', 'physical-device', 'device', 'swapchain', 'swapchain-image', 'image-view'];
		const ps = objectScenario(pres, 'en', V);
		const r = [...pres.map((id) => mk(id)), rm('swapchain')].reduce((x, a) => act(ps, x, a), begin(ps));
		expect(ps.status(r.state, 'image-view')).toBe('error');
		expect(ps.status(r.state, 'swapchain-image')).toBe('locked');
	});
	test('why a locked object is locked: the missing parents as validation errors', () => {
		expect(why(sc, play(mk('instance')).state, 'queue').map((f) => f.ref)).toEqual(['VUID-vkGetDeviceQueue-device-parameter']);
	});
	test('destroying the instance with live children is refused (VUID-00629), queues are never destroyed by the app', () => {
		const full = play(...SCOPE.map((id) => mk(id)));
		expect(complete(sc, full.state)).toBe(true);
		expect(act(sc, full, rm('instance')).last?.log[0]).toMatchObject({ kind: 'error', ref: 'VUID-vkDestroyInstance-instance-00629' });
		expect(act(sc, full, rm('queue')).last?.log).toEqual([{ kind: 'info', text: 'VkQueue is not destroyed by the application: it was obtained with vkGetDeviceQueue and lives as long as VkDevice.' }]);
		const down = [rm('device'), rm('debug-messenger'), rm('instance')].reduce((r, a) => act(sc, r, a), full);
		expect(down.state.alive).toEqual([]);
		expect(complete(sc, down.state)).toBe(false);
	});
	test('undo and the stored run reproduce the simulator state', () => {
		const r = play(mk('instance', true), mk('physical-device'), mk('queue'), mk('device'));
		expect(undo(sc, r).state.alive).toEqual(['instance', 'physical-device']);
		expect(replay(sc, decode(JSON.parse(JSON.stringify(encode(sc, r))), V)!)).toEqual(r);
	});
	test('every object of objects.json has a primary verb and a destroy verb', () => {
		const all = objectScenario(OBJECTS.map((o) => o.id), 'en', V);
		for (const o of OBJECTS) expect(all.verbs(all.init, o.id).map((v) => v.id)).toEqual([o.how, 'destroy']);
	});
	test('a stored run of another rules version is not replayed (the trainer starts fresh)', () => {
		const stored = JSON.parse(JSON.stringify(encode(sc, play(mk('instance'), mk('physical-device')))));
		expect(decode(stored, V)).toHaveLength(2);
		expect(decode(stored, objectScenario(SCOPE, 'en', 'other').version)).toBeNull();
		// v0.6.0 runs (format 1, no version) are dropped too.
		expect(decode({ v: 1, acts: stored.acts }, V)).toBeNull();
	});
	test('replay stops at the first act the rules no longer accept and keeps the ones before it', () => {
		const acts = play(...SCOPE.map((id) => mk(id))).acts;
		// Rule change: the exercise no longer has a debug messenger; the device was created with "create".
		const narrower = objectScenario(['instance', 'physical-device', 'device', 'queue'], 'en', V);
		expect(replay(narrower, acts).acts).toEqual(acts.slice(0, 1));
		expect(() => act(narrower, begin(narrower), mk('debug-messenger'))).toThrow(StaleAct);
		// A verb id the object does not have (e.g. renamed) and an option value of the wrong type are stale as well.
		expect(replay(sc, [mk('instance'), { item: 'physical-device', verb: 'create' }]).acts).toHaveLength(1);
		expect(replay(sc, [{ item: 'instance', verb: 'create', opts: { debugUtils: 'yes' } }]).acts).toEqual([]);
	});
	test('the rules version follows the data and the rule code', () => {
		expect(ruleVersion(['{"a":1}', 'code'])).toMatch(/^[0-9a-f]{8}$/);
		expect(ruleVersion(['{"a":1}', 'code'])).toBe(ruleVersion(['{"a":1}', 'code']));
		expect(ruleVersion(['{"a":2}', 'code'])).not.toBe(ruleVersion(['{"a":1}', 'code']));
		expect(ruleVersion(['ab', 'c'])).not.toBe(ruleVersion(['a', 'bc']));
	});
	test('card statuses agree with the create rules in every state of a full build and tear-down', () => {
		const all = objectScenario(OBJECTS.map((o) => o.id), 'en', V);
		let r = begin(all);
		const order = ['instance', 'debug-messenger', 'surface', 'physical-device', 'device', 'queue', 'swapchain', 'swapchain-image', 'image-view'];
		for (const a of [...order.map((id) => mk(id)), rm('swapchain'), rm('image-view'), rm('device')]) {
			expect(consistency(all, r.state)).toEqual([]);
			r = act(all, r, a);
		}
		expect(consistency(all, r.state)).toEqual([]);
	});});
