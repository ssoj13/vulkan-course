/**
 * Per-language text tables of this course's widgets (ObjectGraph, labs). Every widget keeps its own table
 * `{ en: {...}, ru: {...} }` next to its code; these helpers pick a language and check that every language has the
 * same keys and the same `{placeholders}` as English (tests call `textErrors` for every table).
 */

export type TextTable<T extends Record<string, string>> = Readonly<Record<string, T>>;

/** The texts of one language; throws for a language the table does not have (a missing translation is a bug). */
export function pick<T extends Record<string, string>>(table: TextTable<T>, lang: string, what: string): T {
	const t = table[lang];
	if (!t) throw new Error(`${what}: no texts for "${lang}"`);
	return t;
}

const names = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort().join(',');

/** Differences from English in keys or placeholders (empty = consistent). */
export function textErrors(table: TextTable<Record<string, string>>, what: string): string[] {
	const en = table.en;
	if (!en) return [`${what}: no English texts`];
	const errs: string[] = [];
	for (const [lang, t] of Object.entries(table)) {
		for (const k of Object.keys(en)) if (!(k in t)) errs.push(`${what}.${lang}: missing "${k}"`);
		for (const k of Object.keys(t)) {
			if (!(k in en)) errs.push(`${what}.${lang}: unknown "${k}"`);
			else if (names(t[k]!) !== names(en[k]!)) errs.push(`${what}.${lang}.${k}: placeholders differ from English`);
		}
	}
	return errs;
}
