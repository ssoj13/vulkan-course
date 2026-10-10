/**
 * Copies the valid-usage texts the simulators quote from the Vulkan SDK's registry into src/data/vuids.json,
 * so no validation message in the course is written by hand.
 *
 *   bun scripts/sync-vuids.ts [path/to/validusage.json]   (default: $VULKAN_SDK/share/vulkan/registry/validusage.json)
 *
 * Collects every `VUID-...` string in src/data/*.json (except vuids.json itself), looks each one up, and fails if
 * any is missing from the registry. Also checks that every Vulkan command named in objects.json (`call`,
 * `destroy`) exists in the registry. Rerun after an SDK update or after editing objects.json.
 */
import { readdirSync } from 'node:fs';
import { join } from 'node:path';

type Vu = { vuid: string; text: string };
type Db = { validation: Record<string, Record<string, Vu[]>> };

const dataDir = join(import.meta.dir, '..', 'src', 'data');
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

const wanted = new Set<string>();
const files = readdirSync(dataDir).filter((f) => f.endsWith('.json') && f !== 'vuids.json');
for (const f of files) for (const m of (await Bun.file(join(dataDir, f)).text()).matchAll(/"(VUID-[\w-]+)"/g)) wanted.add(m[1]!);

const missing = [...wanted].filter((v) => !index.has(v));
const objects = (await Bun.file(join(dataDir, 'objects.json')).json()) as { objects: { id: string; call: string; destroy?: string }[] };
const badCalls = objects.objects.flatMap((o) => [o.call, o.destroy].filter((c): c is string => !!c && !(c in db.validation)).map((c) => `${o.id}: ${c}`));
if (missing.length || badCalls.length) {
	throw new Error(
		[missing.length && `VUIDs not in ${regPath}:\n  ${missing.join('\n  ')}`, badCalls.length && `unknown commands:\n  ${badCalls.join('\n  ')}`]
			.filter(Boolean)
			.join('\n'),
	);
}

const out = Object.fromEntries([...wanted].sort().map((v) => [v, plain(index.get(v)!)]));
await Bun.write(join(dataDir, 'vuids.json'), `${JSON.stringify(out, null, 2)}\n`);
console.log(`vuids.json: ${wanted.size} VUIDs from ${regPath}`);
