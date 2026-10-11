/**
 * Compiles every `shaders/*.slang` to SPIR-V in `shaders/spv/` and records what was built in
 * `shaders/spv/manifest.json`. The .spv files are committed, so the examples (Rust: `include_bytes!`, C++: read at
 * run time) and CI need no shader compiler; `tests/shaders.test.ts` compares the manifest with the current sources,
 * so an edited .slang without a rerun fails the tests.
 *
 *   bun run sync:shaders            (needs VULKAN_SDK: slangc and spirv-val come from the SDK's Bin folder)
 *
 * One .slang = one .spv holding all of its `[shader(...)]` entry points under their own names
 * (`-fvk-use-entrypoint-name`; without it Slang renames a lone entry point to `main`). Target: SPIR-V 1.6, the version
 * Vulkan 1.3+ accepts, validated with `spirv-val --target-env vulkan1.4`. Hashes are over the source text with CRLF
 * folded to LF, so a Windows checkout with autocrlf hashes the same as the LF file in git.
 */
import { createHash } from 'node:crypto';
import { mkdirSync, readdirSync, rmSync } from 'node:fs';
import { basename, join } from 'node:path';

export const SHADER_DIR = join(import.meta.dir, '..', 'shaders');
export const SPV_DIR = join(SHADER_DIR, 'spv');
export const MANIFEST = join(SPV_DIR, 'manifest.json');

/** Flags passed to slangc for every shader (also stored in the manifest). */
const FLAGS = ['-target', 'spirv', '-profile', 'spirv_1_6', '-fvk-use-entrypoint-name'];

/** One compiled output file of a shader source. */
export type Output = { file: string; bytes: number; sha256: string };
/** One `.slang` source and the files compiled from it. */
export type Entry = { source: string; sourceSha256: string; outputs: Output[] };
export type Manifest = { compiler: string; flags: string[]; shaders: Entry[] };

/** SHA-256 (hex) of a shader source, line endings normalized to LF. */
export function sourceHash(text: string): string {
	return createHash('sha256').update(text.replace(/\r\n/g, '\n')).digest('hex');
}

/** SHA-256 (hex) of binary data. */
export function bytesHash(data: Uint8Array): string {
	return createHash('sha256').update(data).digest('hex');
}

/** Sorted file names of all `.slang` sources in shaders/. */
export function slangSources(): string[] {
	return readdirSync(SHADER_DIR)
		.filter((f) => f.endsWith('.slang'))
		.sort();
}

/** Runs a tool and throws with its output if it fails; returns stdout + stderr. */
function run(cmd: string[]): string {
	const r = Bun.spawnSync(cmd, { stdout: 'pipe', stderr: 'pipe' });
	const out = `${r.stdout.toString()}${r.stderr.toString()}`.trim();
	if (r.exitCode !== 0) throw new Error(`${basename(cmd[0]!)} failed (exit ${r.exitCode}):\n${out}`);
	return out;
}

async function main(): Promise<void> {
	const sdk = process.env.VULKAN_SDK;
	if (!sdk) throw new Error('VULKAN_SDK is not set: install the Vulkan SDK (it ships slangc and spirv-val)');
	const exe = process.platform === 'win32' ? '.exe' : '';
	const bin = join(sdk, process.platform === 'win32' ? 'Bin' : 'bin');
	const slangc = join(bin, `slangc${exe}`);
	const spirvVal = join(bin, `spirv-val${exe}`);
	mkdirSync(SPV_DIR, { recursive: true });

	const shaders: Entry[] = [];
	for (const source of slangSources()) {
		const text = await Bun.file(join(SHADER_DIR, source)).text();
		const file = source.replace(/\.slang$/, '.spv');
		const out = join(SPV_DIR, file);
		run([slangc, join(SHADER_DIR, source), ...FLAGS, '-o', out]);
		run([spirvVal, '--target-env', 'vulkan1.4', out]);
		const data = new Uint8Array(await Bun.file(out).arrayBuffer());
		shaders.push({ source, sourceSha256: sourceHash(text), outputs: [{ file, bytes: data.length, sha256: bytesHash(data) }] });
		console.log(`${source} -> spv/${file} (${data.length} bytes, spirv-val OK)`);
	}

	// A .spv whose .slang was deleted or renamed would otherwise linger and be loaded by nobody.
	const expected = new Set(shaders.flatMap((s) => s.outputs.map((o) => o.file)));
	for (const f of readdirSync(SPV_DIR).filter((f) => f.endsWith('.spv') && !expected.has(f))) {
		rmSync(join(SPV_DIR, f));
		console.log(`removed stale spv/${f}`);
	}

	const compiler = `slangc ${run([slangc, '-version'])}`;
	const manifest: Manifest = { compiler, flags: FLAGS, shaders };
	await Bun.write(MANIFEST, `${JSON.stringify(manifest, null, '\t')}\n`);
	console.log(`wrote ${MANIFEST} (${compiler})`);
}

if (import.meta.main) await main();
