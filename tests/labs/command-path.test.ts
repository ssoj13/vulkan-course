import { describe, expect, test } from 'bun:test';
import { FRAMES, TICKS, busy, percent, simulate, type Schedule } from '../../src/lib/labs/command-path';
import { textErrors } from '../../src/lib/text';
import { TEXT } from '../../src/components/labs/command-path.text';

const one = simulate(1);
const two = simulate(2);
const seg = (s: Schedule, f: number, kind: string) => s.segments.find((x) => x.f === f && x.kind === kind)!;

test('texts: every language has the English keys and placeholders', () => {
	expect(textErrors(TEXT, 'command-path')).toEqual([]);
});

describe('both modes', () => {
	for (const s of [one, two]) {
		test(`${s.inFlight} in flight: events are in time order, from start to done`, () => {
			const ts = s.events.map((e) => e.t);
			expect(ts).toEqual([...ts].sort((a, b) => a - b));
			expect(s.events[0]!.kind).toBe('start');
			expect(s.events.at(-1)!.kind).toBe('done');
			expect(s.events.at(-1)!.t).toBe(s.end);
		});
		test(`${s.inFlight} in flight: a submit returns at once, the GPU starts only later`, () => {
			for (let f = 0; f < FRAMES; f++) {
				const sub = seg(s, f, 'submit');
				expect(sub.t1 - sub.t0).toBe(TICKS.submit);
				expect(seg(s, f, 'exec').t0).toBeGreaterThan(sub.t1);
				expect(seg(s, f, 'queued').t0).toBe(sub.t1);
			}
		});
		test(`${s.inFlight} in flight: no lane does two things at once, and the GPU lane keeps submission order`, () => {
			for (const lane of ['cpu', 'gpu'] as const) {
				const bars = s.segments.filter((x) => x.lane === lane).sort((a, b) => a.t0 - b.t0);
				for (let i = 1; i < bars.length; i++) expect(bars[i]!.t0).toBeGreaterThanOrEqual(bars[i - 1]!.t1);
			}
			const gpu = s.segments.filter((x) => x.lane === 'gpu').sort((a, b) => a.t0 - b.t0);
			expect(gpu.map((x) => x.f)).toEqual([0, 1, 2]);
		});
		test(`${s.inFlight} in flight: a command buffer is never re-recorded while the GPU still executes it`, () => {
			// VUID-vkBeginCommandBuffer-commandBuffer-00049: not in the recording or pending state.
			for (let f = s.inFlight; f < FRAMES; f++) expect(seg(s, f, 'record').t0).toBeGreaterThanOrEqual(seg(s, f - s.inFlight, 'exec').t1);
		});
		test(`${s.inFlight} in flight: every frame records into slot f % inFlight`, () => {
			for (const e of s.events) expect(e.cb).toBe(e.f % s.inFlight);
		});
	}
});

describe('waiting right after the submit vs a frame later', () => {
	test('1 in flight: the CPU waits for the whole GPU execution of every frame', () => {
		for (let f = 0; f < FRAMES; f++) expect(seg(one, f, 'wait').t1).toBe(seg(one, f, 'exec').t1);
		expect(one.events.some((e) => e.kind === 'waitFree')).toBe(false);
	});
	test('1 in flight: CPU and GPU never work at the same time', () => {
		const cpuWork = one.segments.filter((x) => x.lane === 'cpu' && x.kind !== 'wait');
		for (const g of one.segments.filter((x) => x.lane === 'gpu')) for (const c of cpuWork) expect(c.t1 <= g.t0 || c.t0 >= g.t1).toBe(true);
	});
	test('2 in flight: the first waits return at once (fences created signaled)', () => {
		const free = two.events.filter((e) => e.kind === 'waitFree' && e.g < 0).map((e) => e.f);
		expect(free).toEqual([0, 1]);
	});
	test('2 in flight overlaps CPU and GPU: shorter run, busier GPU, less waiting', () => {
		expect(two.end).toBeLessThan(one.end);
		expect(percent(two, ['exec'])).toBeGreaterThan(percent(one, ['exec']));
		expect(busy(two, ['wait'])).toBeLessThan(busy(one, ['wait']));
	});
	test('the illustrative numbers the page quotes', () => {
		expect([one.end, percent(one, ['exec']), percent(one, ['wait'])]).toEqual([39, 46, 54]);
		expect([two.end, percent(two, ['exec']), percent(two, ['wait'])]).toEqual([26, 69, 4]);
	});
});
