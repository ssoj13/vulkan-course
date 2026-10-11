/**
 * Checks the lessons of every language before a full build: each <Quiz> must pass the engine's rules (course-kit
 * lib/quiz.ts `quizErrors`: structure, and no answer given away by option length), and its `#anchor` refs must be
 * headings of the same page (slugged like Starlight does, with github-slugger). Russian prose (pages under ru/ and the
 * `ru` widget texts) is checked for two style rules: interactive widgets are «интерактив», never «лаба» or
 * «лаборатория» (the engine's nav says «Интерактивы»), and a dash between words is «—», not a hyphen. Prints every
 * problem and exits non-zero if there is any.
 *
 *   bun scripts/lint-content.ts
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import GithubSlugger from 'github-slugger';
import { quizErrors, type Question } from 'course-kit/lib/quiz.ts';
import { COURSE_TEXT } from '../src/i18n';

const docs = join(import.meta.dir, '..', 'src', 'content', 'docs');

function mdxFiles(dir: string): string[] {
	return readdirSync(dir).flatMap((n) => {
		const p = join(dir, n);
		return statSync(p).isDirectory() ? mdxFiles(p) : p.endsWith('.mdx') ? [p] : [];
	});
}

/** Heading anchors of a page, as Starlight generates them (fenced code blocks are skipped). */
function anchors(src: string): Set<string> {
	const slugger = new GithubSlugger();
	const out = new Set<string>();
	let fenced = false;
	for (const line of src.split(/\r?\n/)) {
		if (/^\s*```/.test(line)) fenced = !fenced;
		const m = !fenced && /^#{2,6}\s+(.+?)\s*$/.exec(line);
		if (m) out.add(slugger.slug(m[1]!.replace(/`/g, '')));
	}
	return out;
}

/** «лаба», «лаб», «лабой»…, «лаборатория» in any form: the Russian word for a lab is «интерактив». */
const LAB_WORD = /(?<![а-яё])лаб(?:а|ы|е|у|ой|ою|ам|ами|ах)?(?![а-яё])|лаборатори/iu;
/** A hyphen standing alone between words is a dash written wrong (list markers start a line, so they never match). */
const HYPHEN_DASH = /\S - \S/u;

/** Russian style problems of one piece of prose; inline code is left out (`a - b` is code, not a dash). */
function ruStyle(text: string): string[] {
	const prose = text.replace(/`[^`]*`/g, '``');
	const errs: string[] = [];
	if (LAB_WORD.test(prose)) errs.push('«лаба»/«лаборатория»: call it «интерактив»');
	if (HYPHEN_DASH.test(prose)) errs.push('hyphen used as a dash: write « — »');
	return errs;
}

const problems: string[] = [];

for (const file of mdxFiles(join(docs, 'ru'))) {
	let fenced = false;
	readFileSync(file, 'utf8')
		.split(/\r?\n/)
		.forEach((line, i) => {
			if (/^\s*```/.test(line)) fenced = !fenced;
			else if (!fenced) for (const e of ruStyle(line)) problems.push(`${relative(docs, file)}:${i + 1}: ${e}`);
		});
}

const labs = join(import.meta.dir, '..', 'src', 'components', 'labs');
const tables: [string, Record<string, Record<string, string>>][] = [['src/i18n.ts', COURSE_TEXT]];
for (const n of readdirSync(labs).filter((n) => n.endsWith('.text.ts'))) {
	const { TEXT } = (await import(join(labs, n))) as { TEXT: Record<string, Record<string, string>> };
	tables.push([n, TEXT]);
}
for (const [where, t] of tables) {
	for (const [k, v] of Object.entries(t.ru ?? {})) for (const e of ruStyle(v)) problems.push(`${where} ru.${k}: ${e}`);
}
// Components write some Russian inline (RebarCheck.astro); only the word rule applies there, code is full of " - ".
const components = join(import.meta.dir, '..', 'src', 'components');
const astroFiles = (dir: string): string[] =>
	readdirSync(dir).flatMap((n) => {
		const p = join(dir, n);
		return statSync(p).isDirectory() ? astroFiles(p) : p.endsWith('.astro') ? [p] : [];
	});
for (const file of astroFiles(components)) {
	readFileSync(file, 'utf8')
		.split(/\r?\n/)
		.forEach((line, i) => {
			if (LAB_WORD.test(line)) problems.push(`${relative(components, file)}:${i + 1}: «лаба»/«лаборатория»: call it «интерактив»`);
		});
}

let quizzes = 0;
for (const file of mdxFiles(docs)) {
	const where = relative(docs, file);
	const src = readFileSync(file, 'utf8');
	const m = /<Quiz\s+questions=\{(\[[\s\S]*?\])\}\s*\/>/.exec(src);
	if (!m) continue;
	quizzes++;
	let qs: Question[];
	try {
		qs = new Function(`return ${m[1]}`)() as Question[];
	} catch (e) {
		problems.push(`${where}: quiz is not a JS literal (${e instanceof Error ? e.message : e})`);
		continue;
	}
	for (const err of quizErrors(qs)) problems.push(`${where}: ${err}`);
	const heads = anchors(src);
	qs.forEach((q, i) => {
		if (typeof q?.ref === 'string' && q.ref.startsWith('#') && !heads.has(q.ref.slice(1))) {
			problems.push(`${where} Q${i + 1}: ref "${q.ref}" is not a heading (have: ${[...heads].join(', ')})`);
		}
	});
}

if (problems.length) {
	console.error(problems.join('\n'));
	console.error(`\n${problems.length} problem(s) in ${quizzes} quizzes`);
	process.exit(1);
}
console.log(`content ok: ${quizzes} quizzes`);
