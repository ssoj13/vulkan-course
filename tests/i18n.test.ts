import { expect, test } from 'bun:test';
import { COURSE_TEXT, HOW_KEY, SIM_KEY, STAGE_KEY } from '../src/i18n';
import { STAGES } from '../src/lib/objects';

const names = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();

test('every language has exactly the keys and placeholders of English', () => {
	const en = COURSE_TEXT.en!;
	for (const [lang, t] of Object.entries(COURSE_TEXT)) {
		expect(Object.keys(t).sort(), lang).toEqual(Object.keys(en).sort());
		for (const [k, v] of Object.entries(t)) expect(names(v), `${lang}.${k}`).toEqual(names(en[k as keyof typeof en]));
	}
});

test('every stage, way of obtaining a handle and simulator remark has a text', () => {
	const keys = new Set(Object.keys(COURSE_TEXT.en!));
	for (const s of STAGES) expect(keys.has(STAGE_KEY[s]), s).toBe(true);
	for (const k of [...Object.values(HOW_KEY), ...Object.values(SIM_KEY)]) expect(keys.has(k), k).toBe(true);
});
