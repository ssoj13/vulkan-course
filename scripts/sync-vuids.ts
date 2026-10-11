/**
 * Copies the valid-usage texts the course quotes from the Vulkan SDK's registry into src/data/vuids.json, so no
 * validation message in the course is written by hand. vuids.json is the only place with VUID texts: code reads them
 * through `vuid(id)` (src/lib/vuids.ts).
 *
 *   bun scripts/sync-vuids.ts [path/to/validusage.json]   (default: $VULKAN_SDK/share/vulkan/registry/validusage.json)
 *
 * Collects every `VUID-...` id mentioned anywhere in src/ (scripts/vuid-refs.ts), looks each one up, and fails if any
 * is missing from the registry. Also checks that every Vulkan command named in objects.json (`call`, `destroy`)
 * exists in the registry. Rerun after an SDK update or after mentioning a new VUID.
 */
import { join } from 'node:path';
import { SRC_DIR, VUIDS_FILE, vuidRefs } from './vuid-refs';

type Vu = { vuid: string; text: string };
type Db = { validation: Record<string, Record<string, Vu[]>> };

const sdk = process.env.VULKAN_SDK;
const regPath = Bun.argv[2] ?? (sdk ? join(sdk, 'share', 'vulkan', 'registry', 'validusage.json') : '');
if (!regPath) throw new Error('pass the path to validusage.json or set VULKAN_SDK');

const db = (await Bun.file(regPath).json()) as Db;
const index = new Map<string, string>();
for (const byExt of Object.values(db.validation)) for (const list of Object.values(byExt)) for (const v of list) index.set(v.vuid, v.text);

/** The registry text carries asciidoc-style markup (`<code>`, `<a>`); the course shows plain text. */
const plain = (t: string) =>
	t
		.replace(/<[^>]+>/g, '')
		.replace(/&lt;/g, '<')
		.replace(/&gt;/g, '>')
		.replace(/&amp;/g, '&')
		.replace(/\s+/g, ' ')
		.trim();

const refs = await vuidRefs();
const missing = [...refs].filter(([v]) => !index.has(v)).map(([v, files]) => `${v} (${files.join(', ')})`);
const objects = (await Bun.file(join(SRC_DIR, 'data', 'objects.json')).json()) as { objects: { id: string; call: string; destroy?: string }[] };
const badCalls = objects.objects.flatMap((o) => [o.call, o.destroy].filter((c): c is string => !!c && !(c in db.validation)).map((c) => `${o.id}: ${c}`));
if (missing.length || badCalls.length) {
	throw new Error(
		[missing.length && `VUIDs not in ${regPath}:\n  ${missing.join('\n  ')}`, badCalls.length && `unknown commands:\n  ${badCalls.join('\n  ')}`]
			.filter(Boolean)
			.join('\n'),
	);
}

const out = Object.fromEntries([...refs.keys()].map((v) => [v, plain(index.get(v)!)]));
await Bun.write(VUIDS_FILE, `${JSON.stringify(out, null, 2)}\n`);
console.log(`vuids.json: ${refs.size} VUIDs from ${regPath}`);
