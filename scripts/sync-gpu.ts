/**
 * Snapshots the course dev machine's GPU into src/data/gpu.json, so the device, queue and memory labs show real
 * numbers instead of invented ones.
 *
 *   bun scripts/sync-gpu.ts [path/to/vulkaninfo]   (default: $VULKAN_SDK/Bin/vulkaninfoSDK.exe)
 *
 * Two vulkaninfo runs, both for GPU 0:
 * - `--json=0` (Vulkan Profiles format): device properties, limits, features, queue families.
 * - `--text`: memory heaps and types, which the profile format does not carry, plus the driver version string
 *   (vendor-specific encoding that vulkaninfo decodes) and the instance version. Heap budget/usage are left out:
 *   they change from run to run.
 * Fails if the two runs disagree on the device name or the text cannot be parsed. Rerun after a driver or SDK update.
 */
import { mkdtempSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';

type Flat = Record<string, unknown>;
type QueueFamily = {
	VkQueueFamilyProperties: {
		queueFlags: string[];
		queueCount: number;
		timestampValidBits: number;
		minImageTransferGranularity: { width: number; height: number; depth: number };
	};
};
type Profile = {
	capabilities: {
		device: {
			features: Record<string, Flat>;
			properties: { VkPhysicalDeviceProperties: Flat & { limits: Flat } };
			queueFamiliesProperties: QueueFamily[];
		};
	};
};

const sdk = process.env.VULKAN_SDK;
const exe = Bun.argv[2] ?? (sdk ? join(sdk, 'Bin', 'vulkaninfoSDK.exe') : '');
if (!exe) throw new Error('pass the path to vulkaninfo or set VULKAN_SDK');
const outPath = join(import.meta.dir, '..', 'src', 'data', 'gpu.json');
const FEATURE_STRUCTS = [
	'VkPhysicalDeviceFeatures',
	'VkPhysicalDeviceVulkan11Features',
	'VkPhysicalDeviceVulkan12Features',
	'VkPhysicalDeviceVulkan13Features',
	'VkPhysicalDeviceVulkan14Features',
];

/** Runs vulkaninfo in `cwd` (the JSON mode writes its file there) and returns stdout; stderr carries layer warnings. */
function run(args: string[], cwd: string): string {
	const r = Bun.spawnSync([exe, ...args], { cwd, stdout: 'pipe', stderr: 'pipe' });
	if (r.exitCode !== 0) throw new Error(`${exe} ${args.join(' ')} exited with ${r.exitCode}:\n${r.stderr.toString()}`);
	return r.stdout.toString();
}

/** Same object with keys sorted, so the file diffs cleanly between runs. */
const sorted = (o: Flat): Flat => Object.fromEntries(Object.entries(o).sort(([a], [b]) => a.localeCompare(b)));

/** VK_API_VERSION_MAJOR/MINOR/PATCH of a packed version (variant bits ignored). */
const apiString = (v: number) => `${(v >>> 22) & 0x7f}.${(v >>> 12) & 0x3ff}.${v & 0xfff}`;

function must(m: RegExpMatchArray | null, what: string): string {
	if (!m?.[1]) throw new Error(`vulkaninfo --text: ${what} not found`);
	return m[1];
}

/** Splits `text` into blocks that start at `name[i]:` lines, in index order. */
function blocks(text: string, name: string): string[] {
	const parts = text.split(new RegExp(`^\\s*${name}\\[\\d+\\]:\\s*$`, 'm'));
	return parts.slice(1);
}

/** The text names flags without the `VK_` prefix (`MEMORY_HEAP_DEVICE_LOCAL_BIT`); the JSON names keep it. */
const flagsOf = (block: string, prefix: string) => [...block.matchAll(new RegExp(`\\b(${prefix}_\\w+)`, 'g'))].map((m) => `VK_${m[1]}`);

const BIT: Record<string, number> = {
	VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: 0x1,
	VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: 0x2,
	VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: 0x4,
	VK_MEMORY_PROPERTY_HOST_CACHED_BIT: 0x8,
	VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT: 0x10,
	VK_MEMORY_PROPERTY_PROTECTED_BIT: 0x20,
	VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD: 0x40,
	VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD: 0x80,
	VK_MEMORY_PROPERTY_RDMA_CAPABLE_BIT_NV: 0x100,
};

/** Heaps and types from the `VkPhysicalDeviceMemoryProperties` section of one GPU's text output. */
function memory(gpu: string) {
	// The section runs from its `===` underline to the first blank line (the next section header follows it).
	const sec = must(gpu.match(/^VkPhysicalDeviceMemoryProperties:\n=+\n([\s\S]*?)\n[ \t]*\n/m), 'VkPhysicalDeviceMemoryProperties');
	const [heapText, typeText] = sec.split(/^memoryTypes:/m);
	if (!typeText) throw new Error('vulkaninfo --text: memoryTypes not found');
	const heaps = blocks(heapText!, 'memoryHeaps').map((b) => ({
		size: Number(must(b.match(/^\s*size\s*=\s*(\d+)/m), 'heap size')),
		flags: flagsOf(b, 'MEMORY_HEAP'),
	}));
	const types = blocks(typeText, 'memoryTypes').map((b) => {
		const propertyFlags = flagsOf(b, 'MEMORY_PROPERTY');
		// Cross-check the parsed names against the hex mask vulkaninfo prints next to them.
		const mask = parseInt(must(b.match(/propertyFlags\s*=\s*(0x[0-9a-f]+)/i), 'propertyFlags'), 16);
		const sum = propertyFlags.reduce((s, f) => {
			const bit = BIT[f];
			if (bit === undefined) throw new Error(`unknown memory property flag ${f}: add it to BIT`);
			return s | bit;
		}, 0);
		if (sum !== mask) throw new Error(`memory type flags ${propertyFlags.join('|')} do not match mask 0x${mask.toString(16)}`);
		return { heapIndex: Number(must(b.match(/heapIndex\s*=\s*(\d+)/), 'heapIndex')), propertyFlags };
	});
	const counts = sec.match(/memoryHeaps: count = (\d+)[\s\S]*memoryTypes: count = (\d+)/);
	if (Number(counts?.[1]) !== heaps.length || Number(counts?.[2]) !== types.length) throw new Error('memory heap/type counts do not match');
	return { heaps, types };
}

const tmp = mkdtempSync(join(tmpdir(), 'sync-gpu-'));
try {
	run(['--json=0'], tmp);
	const jsonFile = readdirSync(tmp).find((f) => f.endsWith('.json'));
	if (!jsonFile) throw new Error(`vulkaninfo --json=0 wrote no .json file into ${tmp}`);
	const dev = ((await Bun.file(join(tmp, jsonFile)).json()) as Profile).capabilities.device;
	const text = run(['--text'], tmp).replace(/\r\n/g, '\n');

	const props = dev.properties.VkPhysicalDeviceProperties;
	const gpu = must(text.match(/^GPU0:\n([\s\S]*?)(?=^GPU\d+:|(?![\s\S]))/m), 'GPU0 section');
	const textName = must(gpu.match(/^\s*deviceName\s*=\s*(.+)$/m), 'deviceName').trim();
	if (textName !== props.deviceName) throw new Error(`--json says "${props.deviceName}", --text says "${textName}"`);

	const missing = FEATURE_STRUCTS.filter((s) => !dev.features[s]);
	if (missing.length) throw new Error(`feature structs missing from the profile: ${missing.join(', ')}`);

	const out = {
		meta: {
			generator: 'scripts/sync-gpu.ts',
			source: 'vulkaninfo --json on the course dev machine',
			vulkaninfo: `${basename(exe)} --json=0 (device, features, limits, queue families) + --text (memory, driverVersion)`,
			sdk: sdk ? basename(sdk) : null,
			instanceVersion: must(text.match(/^Vulkan Instance Version:\s*(\S+)/m), 'Vulkan Instance Version'),
			// Local calendar date (the 'sv' locale formats as YYYY-MM-DD); toISOString would give the UTC date.
			date: new Date().toLocaleDateString('sv'),
		},
		device: {
			name: props.deviceName,
			type: props.deviceType,
			apiVersion: apiString(props.apiVersion as number),
			driverVersion: must(gpu.match(/^\s*driverVersion\s*=\s*(\S+)/m), 'driverVersion'),
			driverVersionRaw: props.driverVersion,
			vendorID: props.vendorID,
			deviceID: props.deviceID,
		},
		features: Object.fromEntries(FEATURE_STRUCTS.map((s) => [s, sorted(dev.features[s]!)])),
		limits: sorted(props.limits),
		queueFamilies: dev.queueFamiliesProperties.map(({ VkQueueFamilyProperties: q }, index) => ({
			index,
			queueFlags: q.queueFlags,
			queueCount: q.queueCount,
			timestampValidBits: q.timestampValidBits,
			minImageTransferGranularity: q.minImageTransferGranularity,
		})),
		memory: memory(gpu),
	};
	await Bun.write(outPath, `${JSON.stringify(out, null, 2)}\n`);
	console.log(
		`gpu.json: ${out.device.name}, Vulkan ${out.device.apiVersion}, ${out.queueFamilies.length} queue families, ` +
			`${out.memory.heaps.length} heaps / ${out.memory.types.length} memory types`,
	);
} finally {
	rmSync(tmp, { recursive: true, force: true });
}
