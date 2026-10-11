/**
 * Extracts the core VkDynamicState values from the Vulkan registry into src/data/dynamic-states.json, so the labs
 * that show "what can be dynamic" (graphics-pipeline, gl-vs-vulkan) never list a state by hand.
 *
 *   bun scripts/sync-dynamic-states.ts [path/to/vk.xml]   (default: $VULKAN_SDK/share/vulkan/registry/vk.xml)
 *
 * Core = the 9 values of the original `<enums name="VkDynamicState">` block (Vulkan 1.0) plus every non-alias
 * `extends="VkDynamicState"` value required by a `<feature>` of the "vulkan" API (1.3: extended dynamic state 1/2,
 * 1.4: line stipple). Extension-only values (VK_EXT_extended_dynamic_state3, NV, ...) are left out.
 * Output: `[{ name, version }]`, sorted by version, then in registry order. Rerun after an SDK update.
 */
import { join } from 'node:path';

/** One core dynamic state and the core version that introduced it. */
type DynamicState = { name: string; version: string };

const sdk = process.env.VULKAN_SDK;
const regPath = Bun.argv[2] ?? (sdk ? join(sdk, 'share', 'vulkan', 'registry', 'vk.xml') : '');
if (!regPath) throw new Error('pass the path to vk.xml or set VULKAN_SDK');
const xml = await Bun.file(regPath).text();

const base = xml.match(/<enums name="VkDynamicState" type="enum">([\s\S]*?)<\/enums>/)?.[1];
if (!base) throw new Error(`no VkDynamicState enum block in ${regPath}`);
const out: DynamicState[] = [...base.matchAll(/<enum [^>]*name="(VK_DYNAMIC_STATE_\w+)"/g)].map((m) => ({ name: m[1]!, version: '1.0' }));

for (const f of xml.matchAll(/<feature ([^>]*)>([\s\S]*?)<\/feature>/g)) {
	const attrs = f[1]!;
	const api = attrs.match(/\bapi="([^"]+)"/)?.[1]?.split(',') ?? [];
	const version = attrs.match(/\bnumber="([^"]+)"/)?.[1];
	if (!api.includes('vulkan') || !version) continue;
	for (const e of f[2]!.matchAll(/<enum ([^>]*)\/>/g)) {
		const a = e[1]!;
		if (!a.includes('extends="VkDynamicState"') || a.includes('alias=')) continue;
		const name = a.match(/\bname="(\w+)"/)?.[1];
		if (name && !out.some((s) => s.name === name)) out.push({ name, version });
	}
}

// Stable sort keeps registry order inside one version.
out.sort((a, b) => Number(a.version) - Number(b.version));
if (out.length < 9) throw new Error(`only ${out.length} dynamic states found in ${regPath}`);
await Bun.write(join(import.meta.dir, '..', 'src', 'data', 'dynamic-states.json'), `${JSON.stringify(out, null, 2)}\n`);
console.log(`dynamic-states.json: ${out.length} core dynamic states from ${regPath}`);
