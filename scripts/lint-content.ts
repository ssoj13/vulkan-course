/**
 * Checks the lessons of every language before a full build: each <Quiz> must follow the engine's quiz model
 * (course-kit lib/quiz.ts: four options with a `why` each, a valid answer, a `ref`), its `#anchor` refs must be headings
 * of the same page (slugged like Starlight does, with github-slugger), and at most a third of a quiz may have the
 * correct option as the longest. Prints every problem and exits non-zero if there is any.
 *
 *   bun scripts/lint-content.ts
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import GithubSlugger from 'github-slugger';
import { correctIsLongest, giveawayLimit, questionErrors, type Question } from 'course-kit/lib/quiz.ts';

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

const problems: string[] = [];
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
	const heads = anchors(src);
	qs.forEach((q, i) => {
		for (const err of questionErrors(q)) problems.push(`${where} Q${i + 1}: ${err}`);
		if (typeof q.ref === 'string' && q.ref.startsWith('#') && !heads.has(q.ref.slice(1))) {
			problems.push(`${where} Q${i + 1}: ref "${q.ref}" is not a heading (have: ${[...heads].join(', ')})`);
		}
	});
	const give = qs.filter((q) => Array.isArray(q.options) && q.options.every((o) => typeof o?.text === 'string') && correctIsLongest(q));
	if (give.length > giveawayLimit(qs.length)) {
		problems.push(`${where}: ${give.length}/${qs.length} questions have the correct option as the longest (limit ${giveawayLimit(qs.length)})`);
	}
}

if (problems.length) {
	console.error(problems.join('\n'));
	console.error(`\n${problems.length} problem(s) in ${quizzes} quizzes`);
	process.exit(1);
}
console.log(`content ok: ${quizzes} quizzes`);
