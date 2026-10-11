/**
 * Finds every valid-usage id (`VUID-...`) the course mentions anywhere in src/: lab models, views, data, lessons.
 * One scanner for scripts/sync-vuids.ts (which copies the texts of exactly these ids into src/data/vuids.json) and
 * tests/vuids.test.ts (which fails when an id is mentioned but its text was never synced), so a VUID needs no
 * per-lab list: writing its id is enough.
 */
import { join } from 'node:path';

export const SRC_DIR = join(import.meta.dir, '..', 'src');
export const VUIDS_FILE = join(SRC_DIR, 'data', 'vuids.json');

/** A VUID: `VUID-` and dash-separated name parts (never ends with a dash, so prose like "VUID-..." is not matched). */
const VUID_RE = /VUID-[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)+/g;
const FILES = new Bun.Glob('**/*.{ts,tsx,json,mdx,astro}');

/** Referenced VUIDs, sorted, each with the src-relative files that mention it. vuids.json itself is not scanned. */
export async function vuidRefs(root = SRC_DIR): Promise<Map<string, string[]>> {
	const refs = new Map<string, string[]>();
	for await (const rel of FILES.scan({ cwd: root })) {
		const file = join(root, rel);
		if (file === VUIDS_FILE) continue;
		for (const [id] of (await Bun.file(file).text()).matchAll(VUID_RE)) {
			const files = refs.get(id) ?? [];
			if (!files.includes(rel)) files.push(rel);
			refs.set(id, files);
		}
	}
	return new Map([...refs].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
}
