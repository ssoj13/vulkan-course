/**
 * Per-language text tables of this course's widgets (ObjectGraph, labs). Every widget keeps its own table
 * `{ en: {...}, ru: {...} }` next to its code; these helpers pick a language, fill placeholders (`tfmt`, with plural
 * forms) and check that every language has the same keys and the same `{placeholders}` as English (tests call
 * `textErrors` for every table).
 *
 * Placeholders: `{name}` is replaced by the value; `{name|form|form...}` by the plural form that Intl.PluralRules
 * picks for the number, forms listed in the order of `PLURAL_FORMS[lang]`, `#` in a form standing for the number
 * (as in ICU MessageFormat): en `{n|# queue|# queues}`, ru `{n|# очередь|# очереди|# очередей}` (1 очередь,
 * 3 очереди, 5 очередей); a form without `#` is the word alone (a unit after a value the page shows itself).
 */

export type TextTable<T extends Record<string, string>> = Readonly<Record<string, T>>;

/** The texts of one language; throws for a language the table does not have (a missing translation is a bug). */
export function pick<T extends Record<string, string>>(table: TextTable<T>, lang: string, what: string): T {
	const t = table[lang];
	if (!t) throw new Error(`${what}: no texts for "${lang}"`);
	return t;
}

/**
 * Plural categories a text of each language lists, in the order its `{n|...}` forms are written. Russian integers
 * only ever fall into one/few/many; its `other` (fractions: "2,5 очереди") takes the `few` form, see `plural`.
 */
export const PLURAL_FORMS: Readonly<Record<string, readonly Intl.LDMLPluralRule[]>> = {
	en: ['one', 'other'],
	ru: ['one', 'few', 'many'],
};

/** Categories a language shares with another of its own forms (Russian fractions read like `few`). */
const SAME_AS: Readonly<Record<string, Partial<Record<Intl.LDMLPluralRule, Intl.LDMLPluralRule>>>> = {
	ru: { other: 'few' },
};

const rules = new Map<string, Intl.PluralRules>();

/** The form of `forms` (keyed by plural category) that `n` takes in `lang`; throws when that form is missing. */
export function plural(lang: string, n: number, forms: Partial<Record<Intl.LDMLPluralRule, string>>): string {
	let r = rules.get(lang);
	if (!r) rules.set(lang, (r = new Intl.PluralRules(lang)));
	const cat = r.select(n);
	const form = forms[cat] ?? forms[SAME_AS[lang]?.[cat] ?? cat];
	if (form === undefined) throw new Error(`plural: no "${cat}" form for ${n} in ${lang}`);
	return form;
}

const PLACEHOLDER = /\{(\w+)((?:\|[^|{}]*)*)\}/g;

/**
 * Fills the placeholders of a text in `lang`: `{name}` with `String(vars.name)`, `{name|forms...}` with the plural
 * form of the number (its `#` replaced by the number). A placeholder without a value stays as written; a plural one
 * needs a number.
 */
export function tfmt(lang: string, text: string, vars: Readonly<Record<string, string | number>>): string {
	return text.replace(PLACEHOLDER, (m, k: string, list: string) => {
		if (!(k in vars)) return m;
		const v = vars[k]!;
		if (!list) return String(v);
		if (typeof v !== 'number') throw new Error(`tfmt: plural placeholder {${k}} needs a number, got "${v}"`);
		const cats = PLURAL_FORMS[lang];
		if (!cats) throw new Error(`tfmt: no plural forms defined for "${lang}"`);
		const given = list.slice(1).split('|');
		return plural(lang, v, Object.fromEntries(cats.map((c, i) => [c, given[i]]))).replaceAll('#', String(v));
	});
}

/** Placeholder names of a text, plural forms ignored (they differ by language). */
const names = (s: string) => [...s.matchAll(PLACEHOLDER)].map((m) => m[1]).sort().join(',');

/** Differences from English in keys or placeholders, and plural placeholders with a wrong number of forms. */
export function textErrors(table: TextTable<Record<string, string>>, what: string): string[] {
	const en = table.en;
	if (!en) return [`${what}: no English texts`];
	const errs: string[] = [];
	for (const [lang, t] of Object.entries(table)) {
		for (const k of Object.keys(en)) if (!(k in t)) errs.push(`${what}.${lang}: missing "${k}"`);
		for (const k of Object.keys(t)) {
			if (!(k in en)) errs.push(`${what}.${lang}: unknown "${k}"`);
			else if (names(t[k]!) !== names(en[k]!)) errs.push(`${what}.${lang}.${k}: placeholders differ from English`);
			for (const m of t[k]!.matchAll(PLACEHOLDER)) {
				if (!m[2]) continue;
				const want = PLURAL_FORMS[lang]?.length;
				const have = m[2].slice(1).split('|').length;
				if (want === undefined) errs.push(`${what}.${lang}.${k}: no plural forms defined for "${lang}"`);
				else if (have !== want) errs.push(`${what}.${lang}.${k}: {${m[1]}} has ${have} plural forms, ${lang} needs ${want}`);
			}
		}
	}
	return errs;
}
