/**
 * Refreshes `src/data/pipeline-stats.json`: compiles tools/vk-probe/shaders/pressure.slang (the register-pressure
 * series) with the flags the course shaders were built with (read from shaders/spv/manifest.json, so both stay in
 * step), validates it, then runs the vk-probe binary `pipeline-stats`, which embeds the .spv with include_bytes! and
 * queries VK_KHR_pipeline_executable_properties on GPU 0.
 *
 *   bun run sync:pipeline-stats     (needs VULKAN_SDK for slangc/spirv-val and a GPU with the extension)
 *
 * pressure.spv is committed so the crate builds (and CI lints it) without slangc. Text dumps of the driver's internal
 * representations land in tools/vk-probe/out/ (gitignored).
 */
import { join } from 'node:path';
import { MANIFEST, type Manifest } from './sync-shaders.ts';

const ROOT = join(import.meta.dir, '..');
const PROBE = join(ROOT, 'tools', 'vk-probe');

/** Runs a tool with inherited output and throws if it fails. */
function run(cmd: string[]): void {
	const r = Bun.spawnSync(cmd, { cwd: ROOT, stdout: 'inherit', stderr: 'inherit' });
	if (r.exitCode !== 0) throw new Error(`${cmd.join(' ')} failed (exit ${r.exitCode})`);
}

const sdk = process.env.VULKAN_SDK;
if (!sdk) throw new Error('VULKAN_SDK is not set: install the Vulkan SDK (it ships slangc and spirv-val)');
const exe = process.platform === 'win32' ? '.exe' : '';
const bin = join(sdk, process.platform === 'win32' ? 'Bin' : 'bin');
const { flags } = (await Bun.file(MANIFEST).json()) as Manifest;

const spv = join(PROBE, 'shaders', 'pressure.spv');
run([join(bin, `slangc${exe}`), join(PROBE, 'shaders', 'pressure.slang'), ...flags, '-o', spv]);
run([join(bin, `spirv-val${exe}`), '--target-env', 'vulkan1.4', spv]);
run(['cargo', 'run', '--release', '--locked', '--manifest-path', join(PROBE, 'Cargo.toml'), '--bin', 'pipeline-stats', '--', join(ROOT, 'src', 'data', 'pipeline-stats.json')]);
