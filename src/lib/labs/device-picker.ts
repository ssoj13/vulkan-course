/**
 * Model of the device-picker lab: the real properties, features and limits of the course GPU (src/data/gpu.json, from
 * vulkaninfo) and the device-picking rule of examples step 03 (`#region pick` in 03_physical_devices.rs), with the
 * rule's inputs made editable and evaluated over a list of candidate GPUs.
 * Pure module: no DOM, no React, tested with `bun test`.
 */
import gpu from '../../data/gpu.json';

export const DEVICE_TYPES = [
	'VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU',
	'VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU',
	'VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU',
	'VK_PHYSICAL_DEVICE_TYPE_CPU',
	'VK_PHYSICAL_DEVICE_TYPE_OTHER',
] as const;
export type DeviceType = (typeof DEVICE_TYPES)[number];

/** Step 03's preference: discrete over integrated over virtual over everything else (CPU, other). */
export const SCORE: Readonly<Record<DeviceType, number>> = {
	VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: 1000,
	VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: 100,
	VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: 10,
	VK_PHYSICAL_DEVICE_TYPE_CPU: 1,
	VK_PHYSICAL_DEVICE_TYPE_OTHER: 1,
};

/** `DISCRETE_GPU` from `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`, the way step 03 prints it. */
export const shortType = (t: string) => t.replace(/^VK_PHYSICAL_DEVICE_TYPE_/, '');

/** Feature structs of gpu.json and the core version that introduced each one. */
export const FEATURE_STRUCTS = [
	{ struct: 'VkPhysicalDeviceFeatures', minor: 0 },
	{ struct: 'VkPhysicalDeviceVulkan11Features', minor: 1 },
	{ struct: 'VkPhysicalDeviceVulkan12Features', minor: 2 },
	{ struct: 'VkPhysicalDeviceVulkan13Features', minor: 3 },
	{ struct: 'VkPhysicalDeviceVulkan14Features', minor: 4 },
] as const;
export type FeatureStruct = (typeof FEATURE_STRUCTS)[number]['struct'];

/** The five features step 03 prints, with the Vulkan 1.x version of the struct they come from. */
export const STEP03_FEATURES = [
	{ name: 'samplerAnisotropy', minor: 0 },
	{ name: 'dynamicRendering', minor: 3 },
	{ name: 'synchronization2', minor: 3 },
	{ name: 'pushDescriptor', minor: 4 },
	{ name: 'maintenance5', minor: 4 },
] as const;
export type Step03Feature = (typeof STEP03_FEATURES)[number]['name'];
export const featureMinor = (f: Step03Feature) => STEP03_FEATURES.find((x) => x.name === f)!.minor;

/** `{ major, minor, patch }` of a version string such as "1.4.351". */
export function parseVersion(v: string): { major: number; minor: number; patch: number } {
	const m = /^(\d+)\.(\d+)(?:\.(\d+))?$/.exec(v);
	if (!m) throw new Error(`bad version "${v}"`);
	return { major: Number(m[1]), minor: Number(m[2]), patch: Number(m[3] ?? 0) };
}

/** True when version `v` is at least 1.`minor`.0 (the `props.api_version >= vk::API_VERSION_1_x` test of step 03). */
export function reaches(v: string, minor: number): boolean {
	const p = parseVersion(v);
	return p.major > 1 || (p.major === 1 && p.minor >= minor);
}

/** PCI vendor ids named in the lesson. */
export const VENDORS: Readonly<Record<number, string>> = { 0x10de: 'NVIDIA', 0x1002: 'AMD', 0x8086: 'Intel' };

/**
 * NVIDIA's packing of `driverVersion` (the one vulkaninfo decodes for vendor 0x10DE): 10 bits major, 8 minor,
 * 8 secondary, 6 tertiary. Other vendors pack it differently, which is why step 03 prints the raw number.
 */
export function nvidiaDriver(raw: number): string {
	return [(raw >>> 22) & 0x3ff, (raw >>> 14) & 0xff, (raw >>> 6) & 0xff, raw & 0x3f].join('.');
}

export const hex = (n: number, digits = 4) => `0x${n.toString(16).padStart(digits, '0')}`;

/** A GPU as the picking rule sees it. Values of hypothetical candidates are illustrative, not measured. */
export interface Candidate {
	id: string;
	/** deviceName as the driver reports it; hypothetical devices have none (the lab names them per language). */
	name?: string;
	type: DeviceType;
	apiVersion: string;
	/** Values of the step-03 features (only read when the device's version reaches the feature's struct). */
	features: Readonly<Record<Step03Feature, boolean>>;
	hypothetical: boolean;
}

const real = gpu.device;
const featuresOf = (f: typeof gpu.features) => ({
	samplerAnisotropy: f.VkPhysicalDeviceFeatures.samplerAnisotropy,
	dynamicRendering: f.VkPhysicalDeviceVulkan13Features.dynamicRendering,
	synchronization2: f.VkPhysicalDeviceVulkan13Features.synchronization2,
	pushDescriptor: f.VkPhysicalDeviceVulkan14Features.pushDescriptor,
	maintenance5: f.VkPhysicalDeviceVulkan14Features.maintenance5,
});

/**
 * Enumeration order of the lab's machine: a hypothetical integrated GPU first (laptops often list it first), the real
 * course GPU, then a hypothetical software rasterizer. The integrated GPU claims Vulkan 1.4, so all five features are
 * true: the spec makes them mandatory on 1.4. The software rasterizer claims 1.3, where dynamicRendering and
 * synchronization2 are mandatory and the 1.4 struct cannot be chained.
 */
export const CANDIDATES: readonly Candidate[] = [
	{
		id: 'igpu',
		type: 'VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU',
		apiVersion: '1.4.0',
		features: { samplerAnisotropy: true, dynamicRendering: true, synchronization2: true, pushDescriptor: true, maintenance5: true },
		hypothetical: true,
	},
	{
		id: 'real',
		name: real.name,
		type: real.type as DeviceType,
		apiVersion: real.apiVersion,
		features: featuresOf(gpu.features),
		hypothetical: false,
	},
	{
		id: 'cpu',
		type: 'VK_PHYSICAL_DEVICE_TYPE_CPU',
		apiVersion: '1.3.0',
		features: { samplerAnisotropy: false, dynamicRendering: true, synchronization2: true, pushDescriptor: false, maintenance5: false },
		hypothetical: true,
	},
];

/** The editable picking rule. */
export interface Rule {
	/** Required Vulkan 1.x minor version. */
	minMinor: number;
	required: readonly Step03Feature[];
	/** Score by device type (step 03); off = the first eligible device wins. */
	preferDiscrete: boolean;
}

/** Exactly the rule of step 03. */
export const STEP03_RULE: Rule = { minMinor: 4, required: ['dynamicRendering', 'synchronization2'], preferDiscrete: true };

/** One line of the rule trace of a candidate. */
export type Check =
	| { kind: 'version'; ok: boolean; have: string; minor: number }
	/** `na`: the device's version is below the feature's struct, so the struct may not be chained and the feature is not asked. */
	| { kind: 'feature'; ok: boolean; name: Step03Feature; value: boolean | 'na' }
	| { kind: 'score'; ok: true; score: number };

export interface Verdict {
	id: string;
	checks: Check[];
	eligible: boolean;
	score: number;
}

/**
 * Applies the rule to one candidate in step 03's order: version first (asking about a newer struct on an older device
 * is invalid), then each required feature; the first failed check rejects it.
 */
export function judge(c: Candidate, rule: Rule): Verdict {
	const checks: Check[] = [];
	const fail = () => ({ id: c.id, checks, eligible: false, score: 0 });
	const vOk = reaches(c.apiVersion, rule.minMinor);
	checks.push({ kind: 'version', ok: vOk, have: c.apiVersion, minor: rule.minMinor });
	if (!vOk) return fail();
	for (const name of rule.required) {
		const value = reaches(c.apiVersion, featureMinor(name)) ? c.features[name] : 'na';
		checks.push({ kind: 'feature', ok: value === true, name, value });
		if (value !== true) return fail();
	}
	const score = rule.preferDiscrete ? SCORE[c.type] : 0;
	checks.push({ kind: 'score', ok: true, score });
	return { id: c.id, checks, eligible: true, score };
}

export interface Pick {
	verdicts: Verdict[];
	/** Id of the chosen device, or null when no candidate is eligible (step 03 then exits with an error). */
	picked: string | null;
	/** Eligible devices that lost only because they came later with an equal score. */
	tiedLater: string[];
}

/** Step 03's loop: highest score wins, strictly greater, so on a tie the first device found stays. */
export function pick(cands: readonly Candidate[], rule: Rule): Pick {
	const verdicts = cands.map((c) => judge(c, rule));
	let best: Verdict | null = null;
	for (const v of verdicts) if (v.eligible && (!best || v.score > best.score)) best = v;
	const tiedLater = best ? verdicts.filter((v) => v !== best && v.eligible && v.score === best!.score).map((v) => v.id) : [];
	return { verdicts, picked: best?.id ?? null, tiedLater };
}

/** The Vulkan version an app may use with a device: the lower of the instance's request (1.4 here) and the device's. */
export function usable(instanceMinor: number, device: string): string {
	const d = parseVersion(device);
	return d.major > 1 || d.minor >= instanceMinor ? `1.${instanceMinor}` : `1.${d.minor}`;
}

// ---- Features tab ----

export interface FeatureRow {
	name: string;
	value: boolean;
	/** Printed by step 03. */
	step03: boolean;
	/** Required by step 03's picking rule. */
	pickRule: boolean;
}

export interface FeatureGroup {
	struct: FeatureStruct;
	minor: number;
	rows: FeatureRow[];
	/** Supported / total in this struct before filtering. */
	supported: number;
	total: number;
}

const step03 = new Set<string>(STEP03_FEATURES.map((f) => f.name));
const ruleSet = new Set<string>(STEP03_RULE.required);

/** Feature structs of the real GPU, rows filtered by a case-insensitive substring of the name (empty = all). */
export function featureGroups(query: string, onlyUnsupported = false): FeatureGroup[] {
	const q = query.trim().toLowerCase();
	return FEATURE_STRUCTS.map(({ struct, minor }) => {
		const all = Object.entries(gpu.features[struct] as Record<string, boolean>);
		const rows = all
			.filter(([n, v]) => (!q || n.toLowerCase().includes(q)) && (!onlyUnsupported || !v))
			.map(([name, value]) => ({ name, value, step03: step03.has(name) && struct === structOf(name), pickRule: ruleSet.has(name) }));
		return { struct, minor, rows, supported: all.filter(([, v]) => v).length, total: all.length };
	});
}

/** The struct a step-03 feature lives in. */
function structOf(name: string): FeatureStruct | undefined {
	const f = STEP03_FEATURES.find((x) => x.name === name);
	return f && FEATURE_STRUCTS.find((s) => s.minor === f.minor)!.struct;
}

// ---- Limits tab ----

export type Unit = 'texels' | 'layers' | 'bytes' | 'count' | 'ns' | 'invocations' | 'groups' | 'levels' | 'degree' | 'sets';

/**
 * The limits the lab explains (texts in device-picker.text.ts, written from the VkPhysicalDeviceLimits member
 * descriptions of the spec), in reading order. Every other limit is listed with its raw value.
 */
export const EXPLAINED: readonly { name: string; unit: Unit }[] = [
	{ name: 'maxImageDimension2D', unit: 'texels' },
	{ name: 'maxImageDimension3D', unit: 'texels' },
	{ name: 'maxImageArrayLayers', unit: 'layers' },
	{ name: 'maxPushConstantsSize', unit: 'bytes' },
	{ name: 'maxUniformBufferRange', unit: 'bytes' },
	{ name: 'maxStorageBufferRange', unit: 'bytes' },
	{ name: 'minUniformBufferOffsetAlignment', unit: 'bytes' },
	{ name: 'minStorageBufferOffsetAlignment', unit: 'bytes' },
	{ name: 'maxBoundDescriptorSets', unit: 'sets' },
	{ name: 'maxMemoryAllocationCount', unit: 'count' },
	{ name: 'maxSamplerAllocationCount', unit: 'count' },
	{ name: 'bufferImageGranularity', unit: 'bytes' },
	{ name: 'nonCoherentAtomSize', unit: 'bytes' },
	{ name: 'maxComputeWorkGroupInvocations', unit: 'invocations' },
	{ name: 'maxComputeWorkGroupSize', unit: 'invocations' },
	{ name: 'maxComputeWorkGroupCount', unit: 'groups' },
	{ name: 'maxComputeSharedMemorySize', unit: 'bytes' },
	{ name: 'maxSamplerAnisotropy', unit: 'degree' },
	{ name: 'maxColorAttachments', unit: 'count' },
	{ name: 'timestampPeriod', unit: 'ns' },
	{ name: 'discreteQueuePriorities', unit: 'levels' },
];

type LimitValue = number | boolean | (number | string)[];
const LIMITS = gpu.limits as Record<string, LimitValue>;

/** `1, 2, 4, 8` for sample-count masks, `x, y, z` for arrays, `true`/`false`, plain numbers. */
export function showValue(v: LimitValue): string {
	if (Array.isArray(v)) return v.map((x) => (typeof x === 'string' ? x.replace(/^VK_SAMPLE_COUNT_(\d+)_BIT$/, '$1') : String(x))).join(', ');
	return String(v);
}

/** "64 KiB" for an exact multiple of a power of 1024 at or above 1 KiB, else null. */
export function binarySize(bytes: number): string | null {
	for (const [u, n] of [['GiB', 1024 ** 3], ['MiB', 1024 ** 2], ['KiB', 1024]] as const) if (bytes >= n && bytes % n === 0) return `${bytes / n} ${u}`;
	return null;
}

export interface LimitRow {
	name: string;
	value: string;
	/** Unit of an explained limit, else undefined. */
	unit?: Unit;
	/** Byte size in binary units when it reads better (`49152` -> `48 KiB`). */
	size?: string;
}

const explainedUnit = new Map(EXPLAINED.map((e) => [e.name, e.unit]));

/** Explained limits first (in EXPLAINED order), then the rest alphabetically; filtered by substring. */
export function limitRows(query: string, onlyExplained = false): LimitRow[] {
	const q = query.trim().toLowerCase();
	const rest = Object.keys(LIMITS)
		.filter((n) => !explainedUnit.has(n))
		.sort((a, b) => a.localeCompare(b));
	const names = [...EXPLAINED.map((e) => e.name), ...(onlyExplained ? [] : rest)];
	return names
		.filter((n) => !q || n.toLowerCase().includes(q))
		.map((name) => {
			const v = LIMITS[name]!;
			const unit = explainedUnit.get(name);
			const size = unit === 'bytes' && typeof v === 'number' ? binarySize(v) : null;
			return { name, value: showValue(v), ...(unit ? { unit } : {}), ...(size ? { size } : {}) };
		});
}

/** Data problems the lab cannot show correctly (empty = fine); the Astro wrapper fails the build on any. */
export function pickerErrors(): string[] {
	const errs: string[] = [];
	for (const e of EXPLAINED) if (!(e.name in LIMITS)) errs.push(`gpu.json limits: no ${e.name}`);
	for (const f of STEP03_FEATURES) {
		const s = structOf(f.name)!;
		if (typeof (gpu.features[s] as Record<string, boolean>)[f.name] !== 'boolean') errs.push(`gpu.json ${s}: no ${f.name}`);
	}
	if (!(DEVICE_TYPES as readonly string[]).includes(real.type)) errs.push(`gpu.json: unknown device type ${real.type}`);
	return errs;
}
