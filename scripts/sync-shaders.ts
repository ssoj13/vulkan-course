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
 * Vulkan 1.3+ accepts, validated with `spirv-val --target-env vulkan1.4`. Between the two, `decorateNonUniform` adds
 * the NonUniform decorations slangc 2026.1 leaves out (see there). Hashes are over the source text with CRLF folded to
 * LF, so a Windows checkout with autocrlf hashes the same as the LF file in git.
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

/** SPIR-V opcodes and enumerants read or written below (SPIR-V 1.6 unified spec). */
export const SPV = {
	capability: 17,
	load: 61,
	accessChain: 65,
	inBoundsAccessChain: 66,
	ptrAccessChain: 67,
	inBoundsPtrAccessChain: 70,
	decorate: 71,
	copyObject: 83,
	/** Decoration NonUniform (alias NonUniformEXT). */
	nonUniform: 5300,
} as const;

/** Opcodes of the annotation section (OpDecorate ... OpMemberDecorateString), where new decorations go. */
const ANNOTATIONS = new Set([71, 72, 73, 74, 75, 332, 5632, 5633]);

/**
 * Instructions whose result is derived from the operand at word 3 (result id at word 2): access chains and copies
 * produce a pointer into the same resource, OpLoad reads it (a loaded image or sampler is itself the operand the next
 * image instruction uses).
 */
const DERIVED = new Set<number>([SPV.load, SPV.accessChain, SPV.inBoundsAccessChain, SPV.ptrAccessChain, SPV.inBoundsPtrAccessChain, SPV.copyObject]);

/** Splits a SPIR-V module into its instructions (header excluded); each word 0 is (word count << 16) | opcode. */
export function instructions(words: Uint32Array): Uint32Array[] {
	const out: Uint32Array[] = [];
	for (let pos = 5; pos < words.length; ) {
		const count = words[pos]! >>> 16;
		if (count === 0 || pos + count > words.length) throw new Error(`malformed SPIR-V instruction at word ${pos}`);
		out.push(words.subarray(pos, pos + count));
		pos += count;
	}
	return out;
}

/** The capabilities a module declares (OpCapability operands), in module order. */
export function capabilities(words: Uint32Array): number[] {
	return instructions(words)
		.filter((i) => (i[0]! & 0xffff) === SPV.capability)
		.map((i) => i[1]!);
}

/**
 * Ids derived from a NonUniform-decorated id that lack the decoration themselves.
 *
 * Vulkan requires NonUniform on the exact operand an instruction accesses a non-uniformly selected resource through,
 * "the pointer or sampled image operand" (VUID-RuntimeSpirv-subgroupSize-10149); drivers are not required to chase
 * the decoration back through access chains (Mesa does not). Slang 2026.1 decorates only the access chain that picks
 * the descriptor, while the load goes through a further chain derived from it. One forward pass suffices: within a
 * function, definitions precede uses in module order, and pointers through OpPhi (VariablePointers) are not used here.
 */
export function missingNonUniform(words: Uint32Array): number[] {
	const ins = instructions(words);
	const decorated = new Set(ins.filter((i) => (i[0]! & 0xffff) === SPV.decorate && i[2] === SPV.nonUniform).map((i) => i[1]!));
	const missing: number[] = [];
	for (const i of ins) {
		if (!DERIVED.has(i[0]! & 0xffff) ||!decorated.has(i[3]!) || decorated.has(i[2]!)) continue;
		decorated.add(i[2]!);
		missing.push(i[2]!);
	}
	return missing;
}

/**
 * Returns the module with OpDecorate NonUniform added for every id `missingNonUniform` reports, inserted after the last
 * annotation (the section where SPIR-V requires decorations). A workaround for slangc, not a Vulkan rule: Slang fixed
 * part of it upstream (shader-slang/slang#10525, after 2026.1), and once slangc decorates derived pointers itself this
 * returns the module unchanged.
 */
export function decorateNonUniform(words: Uint32Array): Uint32Array {
	const missing = missingNonUniform(words);
	if (missing.length === 0) return words;
	const ins = instructions(words);
	const last = ins.findLastIndex((i) => ANNOTATIONS.has(i[0]! & 0xffff));
	const split = ins[last]!.byteOffset / 4 + ins[last]!.length - words.byteOffset / 4;
	const added = missing.flatMap((id) => [(3 << 16) | SPV.decorate, id, SPV.nonUniform]);
	return Uint32Array.from([...words.subarray(0, split), ...added, ...words.subarray(split)]);
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
		const compiled = new Uint32Array(await Bun.file(out).arrayBuffer());
		const fixed = decorateNonUniform(compiled);
		if (fixed !== compiled) {
			await Bun.write(out, fixed);
			console.log(`${source}: decorated ${(fixed.length - compiled.length) / 3} derived id(s) NonUniform`);
		}
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
