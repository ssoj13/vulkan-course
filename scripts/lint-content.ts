/**
 * Checks the lessons of every language before a full build: each <Quiz> must pass the engine's rules (course-kit
 * lib/quiz.ts `quizErrors`: structure, and no answer given away by option length), and its `#anchor` refs must be
 * headings of the same page (slugged like Starlight does, with github-slugger). Prints every problem and exits
 * non-zero if there is any.
 *
 *   bun scripts/lint-content.ts
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import GithubSlugger from 'github-slugger';
import { quizErrors, type Question } from 'course-kit/lib/quiz.ts';

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
