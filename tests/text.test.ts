import { expect, test } from 'bun:test';
import { plural, textErrors, tfmt } from '../src/lib/text';

const RU = { one: 'очередь', few: 'очереди', many: 'очередей' };

test('plural picks the Russian form by the last digits', () => {
	const got = [0, 1, 2, 4, 5, 11, 12, 14, 21, 22, 25, 101, 111, 1024].map((n) => `${n} ${plural('ru', n, RU)}`);
	expect(got).toEqual([
		'0 очередей', '1 очередь', '2 очереди', '4 очереди', '5 очередей', '11 очередей', '12 очередей',
		'14 очередей', '21 очередь', '22 очереди', '25 очередей', '101 очередь', '111 очередей', '1024 очереди',
	]);
});

test('Russian fractions take the few form; English has one and other', () => {
	expect(plural('ru', 2.5, RU)).toBe('очереди');
	expect([0, 1, 2].map((n) => plural('en', n, { one: 'queue', other: 'queues' }))).toEqual(['queues', 'queue', 'queues']);
});

test('plural throws when the form of the category is missing', () => {
	expect(() => plural('ru', 5, { one: 'очередь', few: 'очереди' })).toThrow();
});

test('tfmt fills plain and plural placeholders', () => {
	expect(tfmt('ru', 'просим {count|# очередь|# очереди|# очередей}, а в нём {max}', { count: 3, max: 2 })).toBe('просим 3 очереди, а в нём 2');
	expect(tfmt('en', '{n|# frame|# frames} in {t} ticks', { n: 1, t: 9 })).toBe('1 frame in 9 ticks');
	expect(tfmt('ru', '{got} из {of|# строки|# строк|# строк}', { got: 3, of: 21 })).toBe('3 из 21 строки');
	expect(tfmt('en', 'keeps {missing}', {})).toBe('keeps {missing}');
	expect(() => tfmt('en', '{n|a|b}', { n: '3' })).toThrow();
});

test('textErrors ignores plural forms in the parity check but counts them per language', () => {
	const ok = { en: { a: '{n|layer|layers}, {x}' }, ru: { a: '{x}: {n|слой|слоя|слоёв}' } };
	expect(textErrors(ok, 'ok')).toEqual([]);
	const bad = { en: { a: '{n|layer}' }, ru: { a: '{n|слой|слоёв}', b: '{y}' } };
	expect(textErrors(bad, 'bad')).toEqual([
		'bad.en.a: {n} has 1 plural forms, en needs 2',
		'bad.ru.a: {n} has 2 plural forms, ru needs 3',
		'bad.ru: unknown "b"',
	]);
});
