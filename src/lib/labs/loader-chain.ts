/**
 * Loader-chain lab model: the path of one Vulkan call through the loader on the course's development machine.
 *
 * Sources (verified 2026-10-10):
 * - Vulkan-Loader docs (KhronosGroup/Vulkan-Loader, docs/LoaderInterfaceArchitecture.md "Dispatch Tables and Call
 *   Chains", LoaderApplicationInterface.md "Best Application Performance Setup" and "Overall Layer Ordering",
 *   LoaderLayerInterface.md manifest fields `enable_environment` / `disable_environment`, and loader.c
 *   `loader_implicit_layer_is_enabled`: the enable variable must equal the manifest's value (strcmp), the disable
 *   variable counts when set to anything): instance calls go
 *   trampoline -> layers -> terminator -> every driver; physical-device calls end in a terminator that calls only the
 *   owning driver; device calls have no loader terminator (the driver ends the chain), and a pointer from
 *   vkGetDeviceProcAddr skips the trampoline too; with no layers such a pointer points directly to the driver
 *   (LoaderApplicationInterface.md). Implicit layers sit closest to the application, layers the app enables come after
 *   them.
 * - The machine itself: the 8 implicit-layer manifests registered there (HKLM\SOFTWARE\Khronos\Vulkan\ImplicitLayers
 *   plus the NVIDIA driver's nv-vk64.json) and the loader's own "vkCreateInstance layer callstack" printout from a run
 *   of step 02: NV_optimus, NV_present, bandicam_helper, then KHRONOS_validation. The other five layers declare an
 *   `enable_environment` variable and stay out of the chain unless it is set.
 * Pure module: no DOM, no React; tested by tests/labs/loader-chain.test.ts.
 */
import gpu from '../../data/gpu.json';

/** An implicit layer registered on the dev machine, in the order the loader discovered it. */
export interface ImplicitLayer {
	name: string;
	/** Who installed it (shown to the reader). */
	by: string;
	/** Set => the layer loads only when this variable holds exactly this value (manifest `enable_environment`). */
	enableEnv?: { name: string; value: string };
	/** Set => the layer is skipped (manifest `disable_environment`, required for implicit layers). */
	disableEnv: string;
}

/** Real manifests of the dev machine, discovery order (the driver's own manifest is found first, through PnP). */
export const IMPLICIT_LAYERS: readonly ImplicitLayer[] = [
	{ name: 'VK_LAYER_NV_optimus', by: 'NVIDIA driver', disableEnv: 'DISABLE_LAYER_NV_OPTIMUS_1' },
	{ name: 'VK_LAYER_NV_present', by: 'NVIDIA driver', disableEnv: 'DISABLE_LAYER_NV_PRESENT_1' },
	{ name: 'VK_LAYER_RENDERDOC_Capture', by: 'RenderDoc', enableEnv: { name: 'ENABLE_VULKAN_RENDERDOC_CAPTURE', value: '1' }, disableEnv: 'DISABLE_VULKAN_RENDERDOC_CAPTURE_1_39' },
	{ name: 'VK_LAYER_bandicam_helper', by: 'Bandicam', disableEnv: 'VK_LAYER_bandicam_helper_DEBUG_1' },
	{ name: 'VK_LAYER_EOS_Overlay', by: 'Epic Games overlay', enableEnv: { name: 'EOS_OVERLAY_ENABLE_VULKAN_WIN64', value: '1' }, disableEnv: 'EOS_OVERLAY_DISABLE_VULKAN_WIN64' },
	{ name: 'VK_LAYER_VALVE_steam_overlay', by: 'Steam overlay', enableEnv: { name: 'ENABLE_VK_LAYER_VALVE_steam_overlay_1', value: '1' }, disableEnv: 'DISABLE_VK_LAYER_VALVE_steam_overlay_1' },
	{ name: 'VK_LAYER_VALVE_steam_fossilize', by: 'Steam Fossilize', enableEnv: { name: 'ENABLE_VK_LAYER_VALVE_steam_fossilize_1', value: '1' }, disableEnv: 'DISABLE_VK_LAYER_VALVE_steam_fossilize_1' },
	{ name: 'VK_LAYER_NV_nsight-sys', by: 'Nsight Systems', enableEnv: { name: 'ENABLE_VK_LAYER_NV_nsight_sys', value: '1' }, disableEnv: 'DISABLE_VK_LAYER_NV_nsight_sys' },
];

/** The explicit layer the course's step 02 enables by name. */
export const VALIDATION = 'VK_LAYER_KHRONOS_validation';

/** The real driver, and a hypothetical second one (a CPU driver such as Mesa's lavapipe) to show the fan-out. */
export const ICDS = ['nvidia', 'lavapipe'] as const;
export type IcdId = (typeof ICDS)[number];
export const ICD_INFO: Readonly<Record<IcdId, { dll: string; gpu: string }>> = {
	nvidia: { dll: 'nvoglv64.dll', gpu: gpu.device.name },
	lavapipe: { dll: 'vulkan_lvp.dll', gpu: 'llvmpipe' },
};

export const CALLS = ['vkCreateInstance', 'vkEnumeratePhysicalDevices', 'vkGetPhysicalDeviceProperties', 'vkQueueSubmit'] as const;
export type CallId = (typeof CALLS)[number];
/** The level of a call decides its chain: instance (broadcast), physical device (one driver), device (no terminator). */
export const LEVEL: Readonly<Record<CallId, 'instance' | 'physical' | 'device'>> = {
	vkCreateInstance: 'instance',
	vkEnumeratePhysicalDevices: 'instance',
	vkGetPhysicalDeviceProperties: 'physical',
	vkQueueSubmit: 'device',
};
/** The dispatchable handle the call is made on (vkCreateInstance has none yet). */
const HANDLE: Readonly<Record<CallId, string>> = {
	vkCreateInstance: '',
	vkEnumeratePhysicalDevices: 'VkInstance',
	vkGetPhysicalDeviceProperties: 'VkPhysicalDevice',
	vkQueueSubmit: 'VkQueue',
};

/** How the program got the function pointer of a device call: the loader's export, or vkGetDeviceProcAddr. */
export type Via = 'export' | 'gdpa';

export interface Setup {
	call: CallId;
	/** Environment variables that are set, with their values (an empty value counts as unset, as on Windows). */
	env: ReadonlyMap<string, string>;
	/** The app names VK_LAYER_KHRONOS_validation in ppEnabledLayerNames. */
	validation: boolean;
	/** The hypothetical second driver is installed. */
	second: boolean;
	/** The GPU the physical-device and device calls are made on. */
	gpu: IcdId;
	via: Via;
}

/**
 * Loader rule for implicit layers: an enable variable, if declared, must hold the manifest's value exactly; a disable
 * variable set to any value wins.
 */
export function implicitActive(l: ImplicitLayer, env: ReadonlyMap<string, string>): boolean {
	return (!l.enableEnv || env.get(l.enableEnv.name) === l.enableEnv.value) && !env.has(l.disableEnv);
}

export interface ChainLayer {
	name: string;
	kind: 'implicit' | 'explicit';
}

/** Layers in call order: active implicit ones first (closest to the app), then the ones the app enabled. */
export function activeLayers(s: Pick<Setup, 'env' | 'validation'>): ChainLayer[] {
	const implicit = IMPLICIT_LAYERS.filter((l) => implicitActive(l, s.env)).map((l): ChainLayer => ({ name: l.name, kind: 'implicit' }));
	return s.validation ? [...implicit, { name: VALIDATION, kind: 'explicit' }] : implicit;
}

export const installedIcds = (second: boolean): IcdId[] => (second ? ['nvidia', 'lavapipe'] : ['nvidia']);

/** A box of the picture. */
export type NodeId = 'app' | 'trampoline' | 'terminator' | `layer:${string}` | `icd:${IcdId}`;
export const layerNode = (name: string): NodeId => `layer:${name}`;
export const icdNode = (id: IcdId): NodeId => `icd:${id}`;

/** Caption keys; the texts live in components/labs/loader-chain.text.ts. */
export type CaptionKey =
	| 'capAppCreate'
	| 'capAppInstance'
	| 'capAppSubmitExport'
	| 'capAppSubmitGdpa'
	| 'capAppSubmitGdpaDirect'
	| 'capTrampCreate'
	| 'capTramp'
	| 'capTrampDevice'
	| 'capTrampDeviceDirect'
	| 'capLayerCreate'
	| 'capLayer'
	| 'capValidationCreate'
	| 'capValidation'
	| 'capTermCreate'
	| 'capTermEnum'
	| 'capTermPhysical'
	| 'capIcdCreate'
	| 'capIcdEnum'
	| 'capIcdPhysical'
	| 'capIcdSubmit'
	| 'capIcdSubmitTramp'
	| 'capIcdSubmitDirect'
	| 'capReturnCreate'
	| 'capReturnEnum'
	| 'capReturn';

/** One step of the animation: the boxes lit at that moment and the caption. */
export interface Hop {
	at: NodeId[];
	key: CaptionKey;
	vars: Record<string, string | number>;
}

export interface Trace {
	layers: ChainLayer[];
	icds: IcdId[];
	/** Boxes the call passes through at all; the others are drawn as "not on this path". */
	path: ReadonlySet<NodeId>;
	hops: Hop[];
}

/** The GPU a physical-device or device call is made on; falls back to NVIDIA when the second driver is absent. */
export const targetIcd = (s: Pick<Setup, 'second' | 'gpu'>): IcdId => (s.second ? s.gpu : 'nvidia');

/** The hop-by-hop path of the chosen call under the chosen setup. */
export function trace(s: Setup): Trace {
	const layers = activeLayers(s);
	const icds = installedIcds(s.second);
	const level = LEVEL[s.call];
	const owner = targetIcd(s);
	const handle = HANDLE[s.call];
	const call = s.call;
	const hops: Hop[] = [];
	const create = call === 'vkCreateInstance';
	// Without layers a device call has nothing between the loader (or the program) and the driver.
	const bare = layers.length === 0;

	// The program's own call.
	if (create) hops.push({ at: ['app'], key: 'capAppCreate', vars: { call, n: s.validation ? 1 : 0 } });
	else if (level === 'device') hops.push({ at: ['app'], key: s.via === 'export' ? 'capAppSubmitExport' : bare ? 'capAppSubmitGdpaDirect' : 'capAppSubmitGdpa', vars: { call, gpu: ICD_INFO[owner].gpu } });
	else hops.push({ at: ['app'], key: 'capAppInstance', vars: { call, handle } });

	// The loader's trampoline, unless the pointer came from vkGetDeviceProcAddr (it then points into the first layer,
	// or into the driver when there are no layers).
	const tramp = !(level === 'device' && s.via === 'gdpa');
	if (tramp) {
		if (create) hops.push({ at: ['trampoline'], key: 'capTrampCreate', vars: { n: layers.length } });
		else hops.push({ at: ['trampoline'], key: level !== 'device' ? 'capTramp' : bare ? 'capTrampDeviceDirect' : 'capTrampDevice', vars: { call, handle } });
	}

	for (const l of layers) {
		const v = l.name === VALIDATION;
		const key: CaptionKey = create ? (v ? 'capValidationCreate' : 'capLayerCreate') : v ? 'capValidation' : 'capLayer';
		hops.push({ at: [layerNode(l.name)], key, vars: { layer: l.name, call } });
	}

	// Instance and physical-device chains end in the loader's terminator; device chains end in the driver itself.
	const names = (ids: IcdId[]) => ids.map((i) => ICD_INFO[i].dll).join(', ');
	if (level === 'instance') {
		hops.push({ at: ['terminator'], key: create ? 'capTermCreate' : 'capTermEnum', vars: { call, n: icds.length } });
		const gpus = icds.map((i) => ICD_INFO[i].gpu).join(', ');
		hops.push({ at: icds.map(icdNode), key: create ? 'capIcdCreate' : 'capIcdEnum', vars: { call, icds: names(icds), gpus } });
		hops.push(create ? { at: ['app'], key: 'capReturnCreate', vars: { call } } : { at: ['app'], key: 'capReturnEnum', vars: { call, n: icds.length } });
	} else if (level === 'physical') {
		hops.push({ at: ['terminator'], key: 'capTermPhysical', vars: { call, icd: ICD_INFO[owner].dll, gpu: ICD_INFO[owner].gpu } });
		hops.push({ at: [icdNode(owner)], key: 'capIcdPhysical', vars: { call, icd: ICD_INFO[owner].dll, gpu: ICD_INFO[owner].gpu } });
		hops.push({ at: ['app'], key: 'capReturn', vars: { call } });
	} else {
		const key: CaptionKey = !bare ? 'capIcdSubmit' : s.via === 'export' ? 'capIcdSubmitTramp' : 'capIcdSubmitDirect';
		hops.push({ at: [icdNode(owner)], key, vars: { call, icd: ICD_INFO[owner].dll, gpu: ICD_INFO[owner].gpu } });
		hops.push({ at: ['app'], key: 'capReturn', vars: { call } });
	}

	return { layers, icds, path: new Set(hops.flatMap((h) => h.at)), hops };
}
