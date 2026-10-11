import { expect, test } from 'bun:test';
import { readdirSync } from 'node:fs';
import { join } from 'node:path';
import { type Manifest, MANIFEST, SHADER_DIR, SPV_DIR, bytesHash, slangSources, sourceHash } from '../scripts/sync-shaders';

// The committed SPIR-V must match the current Slang sources. CI has no slangc, so instead of recompiling we compare
// the hashes `bun run sync:shaders` recorded; any failure here means: rerun it and commit shaders/spv/.
const manifest = (await Bun.file(MANIFEST).json()) as Manifest;

test('every .slang has a manifest entry with at least one output, and nothing else is listed', () => {
	const sources = slangSources();
	expect(sources.length).toBeGreaterThan(0);
	expect(manifest.shaders.map((s) => s.source).sort()).toEqual(sources);
	for (const s of manifest.shaders) expect(s.outputs.length).toBeGreaterThan(0);
});

test('manifest source hashes match the current .slang files (else run bun run sync:shaders)', async () => {
	for (const s of manifest.shaders) {
		const text = await Bun.file(join(SHADER_DIR, s.source)).text();
		expect({ source: s.source, sha256: sourceHash(text) }).toEqual({ source: s.source, sha256: s.sourceSha256 });
	}
});

test('every output exists, is SPIR-V, and is the exact file the manifest recorded', async () => {
	for (const o of manifest.shaders.flatMap((s) => s.outputs)) {
		const data = new Uint8Array(await Bun.file(join(SPV_DIR, o.file)).arrayBuffer());
		// SPIR-V magic number 0x07230203, little-endian (the byte order slangc writes).
		expect([...data.subarray(0, 4)]).toEqual([0x03, 0x02, 0x23, 0x07]);
		expect(data.length % 4).toBe(0);
		expect({ file: o.file, bytes: data.length, sha256: bytesHash(data) }).toEqual(o);
	}
});

test('no orphan .spv without a source', () => {
	const listed = new Set(manifest.shaders.flatMap((s) => s.outputs.map((o) => o.file)));
	expect(readdirSync(SPV_DIR).filter((f) => f.endsWith('.spv') && !listed.has(f))).toEqual([]);
});
