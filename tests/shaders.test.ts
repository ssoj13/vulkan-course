import { expect, test } from 'bun:test';
import { readdirSync } from 'node:fs';
import { join } from 'node:path';
import {
	type Manifest,
	MANIFEST,
	SHADER_DIR,
	SPV_DIR,
	bytesHash,
	capabilities,
	missingNonUniform,
	slangSources,
	sourceHash,
} from '../scripts/sync-shaders';

/** SPIR-V Capability enumerants (unified spec) that the shaders below use. */
const CAP = {
	Shader: 1,
	ShaderNonUniform: 5301,
	RuntimeDescriptorArray: 5302,
	StorageBufferArrayNonUniformIndexing: 5308,
	PhysicalStorageBufferAddresses: 5347,
} as const;
type Cap = keyof typeof CAP;

/**
 * The exact capabilities each .spv must declare. Every Vulkan capability needs a device feature or version, so a
 * module that silently gains one fails on devices without it, and one that silently loses one is undefined behavior:
 * bindless.spv indexes its storage-buffer array non-uniformly, which without StorageBufferArrayNonUniformIndexing
 * violates VUID-RuntimeSpirv-StorageBufferArrayNonUniformIndexing-10136. A new shader must be added here.
 */
const REQUIRED: Record<string, Cap[]> = {
	'bindless.spv': ['Shader', 'RuntimeDescriptorArray', 'ShaderNonUniform', 'StorageBufferArrayNonUniformIndexing'],
	'cube.spv': ['Shader'],
	'debug_printf.spv': ['Shader'],
	'device_address.spv': ['Shader', 'PhysicalStorageBufferAddresses'],
	'mesh.spv': ['Shader'],
	'particles.spv': ['Shader'],
	'reduce.spv': ['Shader'],
	'saxpy.spv': ['Shader'],
	'textured.spv': ['Shader'],
	'transform.spv': ['Shader'],
	'triangle.spv': ['Shader'],
};

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

/** The words of one compiled output (SPIR-V is little-endian as written by slangc, like the hosts that run this). */
async function spvWords(file: string): Promise<Uint32Array> {
	return new Uint32Array(await Bun.file(join(SPV_DIR, file)).arrayBuffer());
}

test('every output declares exactly its pinned capabilities', async () => {
	const outputs = manifest.shaders.flatMap((s) => s.outputs.map((o) => o.file)).sort();
	expect(Object.keys(REQUIRED).sort()).toEqual(outputs);
	for (const file of outputs) {
		const declared = capabilities(await spvWords(file)).sort((a, b) => a - b);
		const pinned = REQUIRED[file]!.map((c) => CAP[c]).sort((a, b) => a - b);
		expect({ file, capabilities: declared }).toEqual({ file, capabilities: pinned });
	}
});

test('every pointer derived from a NonUniform one is decorated NonUniform too', async () => {
	for (const o of manifest.shaders.flatMap((s) => s.outputs)) {
		expect({ file: o.file, missing: missingNonUniform(await spvWords(o.file)) }).toEqual({ file: o.file, missing: [] });
	}
});

test('no orphan .spv without a source', () => {
	const listed = new Set(manifest.shaders.flatMap((s) => s.outputs.map((o) => o.file)));
	expect(readdirSync(SPV_DIR).filter((f) => f.endsWith('.spv') && !listed.has(f))).toEqual([]);
});
