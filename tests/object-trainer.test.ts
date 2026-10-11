import { describe, expect, test } from 'bun:test';
import { act, begin, complete, counts, decode, encode, replay, undo, why, type Act, type Run } from 'course-kit/lib/trainer.ts';
import { OBJECTS, type SimState } from '../src/lib/objects';
import { DEBUG_UTILS, objectScenario } from '../src/lib/object-trainer';

/** The exercise of lessons devices/logical-device: the first section's chain. */
const SCOPE = ['instance', 'debug-messenger', 'physical-device', 'device', 'queue'];
const sc = objectScenario(SCOPE, 'en');
const play = (...acts: Act[]): Run<SimState> => acts.reduce((r, a) => act(sc, r, a), begin(sc));
const mk = (item: string, debugUtils?: boolean): Act => {
	const verb = OBJECTS.find((o) => o.id === item)!.how;
	return debugUtils === undefined ? { item, verb } : { item, verb, opts: { debugUtils } };
};
const rm = (item: string): Act => ({ item, verb: 'destroy' });

describe('ObjectGraph build exercise on the engine trainer', () => {
	test('the instance offers VK_EXT_debug_utils (on by default) only while it does not exist', () => {
		const opt = (r: Run<SimState>) => sc.verbs(r.state, 'instance')[0]!.options;
		expect(opt(begin(sc))).toEqual([{ id: 'debugUtils', label: `enable ${DEBUG_UTILS} + validation layer`, initial: true }]);
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
		const ru = act(objectScenario(SCOPE, 'ru'), begin(objectScenario(SCOPE, 'ru')), mk('instance', false));
		expect(act(objectScenario(SCOPE, 'ru'), ru, mk('debug-messenger')).last?.log[0]!.text).toStartWith('vkCreateDebugUtilsMessengerEXT недоступна');
	});
	test('statuses follow the simulator: locked, ready, done, and error for a dangling view', () => {
		const st = (r: Run<SimState>) => Object.fromEntries(sc.items.map((i) => [i, sc.status(r.state, i)]));
		expect(st(play(mk('instance')))).toEqual({ instance: 'done', 'debug-messenger': 'ready', 'physical-device': 'ready', device: 'locked', queue: 'locked' });
		const pres = ['instance', 'surface', 'physical-device', 'device', 'swapchain', 'swapchain-image', 'image-view'];
		const ps = objectScenario(pres, 'en');
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
		expect(replay(sc, decode(JSON.parse(JSON.stringify(encode(r))))!)).toEqual(r);
	});
	test('every object of objects.json has a primary verb and a destroy verb', () => {
		const all = objectScenario(OBJECTS.map((o) => o.id), 'en');
		for (const o of OBJECTS) expect(all.verbs(all.init, o.id).map((v) => v.id)).toEqual([o.how, 'destroy']);
	});
});
