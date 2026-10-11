/**
 * Model of the queue-planner lab: the real queue families of the course GPU (src/data/gpu.json, from vulkaninfo), a
 * family chosen per role by one of two strategies, and the VkDeviceQueueCreateInfo list that follows, checked against
 * the valid-usage rules of vkCreateDevice. The "dedicated" strategy generalizes examples step 04 (`#region
 * find-graphics` in 04_queue_families.rs) and picks the same families on this GPU.
 * Pure module: no DOM, no React, tested with `bun test`.
 */
import gpu from '../../data/gpu.json';

/** Queue capability names without `VK_QUEUE_` and `_BIT[_KHR|_NV]`, the way step 04 prints them. */
export const shortFlag = (f: string) => f.replace(/^VK_QUEUE_/, '').replace(/_BIT(_\w+)?$/, '');

export interface Family {
	index: number;
	/** Reported flags, short names (`GRAPHICS`, `VIDEO_DECODE`...). */
	flags: string[];
	queueCount: number;
	timestampValidBits: number;
	granularity: [number, number, number];
}

export const FAMILIES: readonly Family[] = gpu.queueFamilies.map((q) => ({
	index: q.index,
	flags: q.queueFlags.map(shortFlag),
	queueCount: q.queueCount,
	timestampValidBits: q.timestampValidBits,
	granularity: [q.minImageTransferGranularity.width, q.minImageTransferGranularity.height, q.minImageTransferGranularity.depth],
}));

/** How many discrete priority levels the GPU maps pQueuePriorities to (spec: at least 2, one at 0.0, one at 1.0). */
export const PRIORITY_LEVELS: number = gpu.limits.discreteQueuePriorities;

export const ROLES = ['graphics', 'present', 'compute', 'transfer', 'decode'] as const;
export type Role = (typeof ROLES)[number];

/** The capability a role needs; present has none: it is a property of a family and a window surface. */
export const ROLE_FLAG: Readonly<Record<Role, string | null>> = {
	graphics: 'GRAPHICS',
	present: null,
	compute: 'COMPUTE',
	transfer: 'TRANSFER',
	decode: 'VIDEO_DECODE',
};

/**
 * What a family can really do: its flags plus TRANSFER on graphics and compute families, which the spec makes
 * implicit ("reporting the VK_QUEUE_TRANSFER_BIT capability separately for that queue family is optional").
 */
export function caps(f: Family): string[] {
	const implicit = !f.flags.includes('TRANSFER') && (f.flags.includes('GRAPHICS') || f.flags.includes('COMPUTE'));
	return implicit ? [...f.flags, 'TRANSFER'] : f.flags;
}

/**
 * Present support is asked per family for one window surface (vkGetPhysicalDeviceSurfaceSupportKHR, a later lesson).
 * The lab has no surface, so it assumes what is true for desktop GPUs: family 0 can present.
 */
export const PRESENT_FAMILY = 0;

export function supports(f: Family, role: Role): boolean {
	const flag = ROLE_FLAG[role];
	return flag === null ? f.index === PRESENT_FAMILY : caps(f).includes(flag);
}

export type Strategy = 'first' | 'dedicated';

/** Capabilities of a family other than the one the role needs; fewer = more dedicated to that role. */
export const extras = (f: Family, role: Role) => caps(f).filter((c) => c !== ROLE_FLAG[role]).length;

/**
 * Family for a role. `first`: the lowest index that supports it. `dedicated`: the one with the fewest other
 * capabilities, lowest index on a tie. Step 04 asks for TRANSFER without GRAPHICS/COMPUTE and for COMPUTE without
 * GRAPHICS, first match; on the course GPU both rules give families 1 and 2, and `dedicated` still falls back to a
 * shared family where step 04 would print "none".
 */
export function familyFor(fams: readonly Family[], role: Role, strategy: Strategy): number | null {
	const ok = fams.filter((f) => supports(f, role));
	if (!ok.length) return null;
	if (strategy === 'first') return ok[0]!.index;
	return ok.reduce((best, f) => (extras(f, role) < extras(best, role) ? f : best)).index;
}

/** What the reader asks for one role. */
export interface RoleReq {
	on: boolean;
	/** Queues wanted (at least 1). */
	count: number;
	/** pQueuePriorities value for each of them. */
	priority: number;
}

export interface Options {
	strategy: Strategy;
	/** Roles that land in one family use the same queues (graphics + present on one queue is the common case). */
	share: boolean;
	/** One VkDeviceQueueCreateInfo per family, as vkCreateDevice requires; off = the naive one per role. */
	merge: boolean;
}

export const DEFAULT_REQS: Readonly<Record<Role, RoleReq>> = {
	graphics: { on: true, count: 1, priority: 1 },
	present: { on: true, count: 1, priority: 1 },
	compute: { on: true, count: 1, priority: 0.5 },
	transfer: { on: true, count: 1, priority: 0.5 },
	decode: { on: false, count: 1, priority: 0.5 },
};
export const DEFAULT_OPTIONS: Options = { strategy: 'dedicated', share: true, merge: true };

export interface CreateInfo {
	family: number;
	count: number;
	priorities: number[];
	roles: Role[];
}

export interface RolePlan {
	role: Role;
	family: number | null;
	/** Queue indices for vkGetDeviceQueue(device, family, index). Empty when the role has no family. */
	queues: number[];
	/** Lands in the graphics role's family (so it is not async compute / a dedicated copy family). */
	withGraphics: boolean;
}

/** Valid-usage rules a plan can break; their texts come from `vuid()` (src/lib/vuids.ts). */
export type Vuid =
	| 'VUID-VkDeviceCreateInfo-queueFamilyIndex-02802'
	| 'VUID-VkDeviceQueueCreateInfo-queueCount-00382'
	| 'VUID-VkDeviceQueueCreateInfo-pQueuePriorities-00383';

export type PlanError =
	| { kind: 'noFamily'; role: Role }
	| { kind: 'duplicate'; vuid: Vuid; family: number; infos: number }
	| { kind: 'tooMany'; vuid: Vuid; family: number; count: number; max: number }
	| { kind: 'priority'; vuid: Vuid; family: number; value: number };

export interface Plan {
	roles: RolePlan[];
	infos: CreateInfo[];
	errors: PlanError[];
}

const range = (from: number, n: number) => Array.from({ length: n }, (_, i) => from + i);

/** Families, queue indices and create infos for the enabled roles, with every rule they break. */
export function plan(fams: readonly Family[], reqs: Readonly<Record<Role, RoleReq>>, opt: Options): Plan {
	const active = ROLES.filter((r) => reqs[r].on);
	const fam = new Map(active.map((r) => [r, familyFor(fams, r, opt.strategy)]));
	const count = (r: Role) => Math.max(1, Math.floor(reqs[r].count));
	const errors: PlanError[] = active.filter((r) => fam.get(r) === null).map((role) => ({ kind: 'noFamily' as const, role }));
	const placed = active.filter((r) => fam.get(r) !== null);
	const queues = new Map<Role, number[]>();
	const infos: CreateInfo[] = [];

	if (opt.merge) {
		const families = [...new Set(placed.map((r) => fam.get(r)!))].sort((a, b) => a - b);
		for (const family of families) {
			const roles = placed.filter((r) => fam.get(r) === family);
			const priorities: number[] = [];
			let next = 0;
			for (const r of roles) {
				const n = count(r);
				// Shared: every role starts at queue 0; a queue used by several roles gets the highest priority asked.
				const first = opt.share ? 0 : next;
				queues.set(r, range(first, n));
				for (const q of range(first, n)) priorities[q] = Math.max(priorities[q] ?? -Infinity, reqs[r].priority);
				next = first + n;
			}
			infos.push({ family, count: priorities.length, priorities, roles });
		}
	} else {
		for (const r of placed) {
			queues.set(r, range(0, count(r)));
			infos.push({ family: fam.get(r)!, count: count(r), priorities: Array(count(r)).fill(reqs[r].priority), roles: [r] });
		}
		const seen = new Map<number, number>();
		for (const i of infos) seen.set(i.family, (seen.get(i.family) ?? 0) + 1);
		for (const [family, n] of seen)
			if (n > 1) errors.push({ kind: 'duplicate', vuid: 'VUID-VkDeviceCreateInfo-queueFamilyIndex-02802', family, infos: n });
	}

	for (const i of infos) {
		const max = fams.find((f) => f.index === i.family)!.queueCount;
		if (i.count > max) errors.push({ kind: 'tooMany', vuid: 'VUID-VkDeviceQueueCreateInfo-queueCount-00382', family: i.family, count: i.count, max });
		const bad = i.priorities.find((p) => !(p >= 0 && p <= 1));
		if (bad !== undefined) errors.push({ kind: 'priority', vuid: 'VUID-VkDeviceQueueCreateInfo-pQueuePriorities-00383', family: i.family, value: bad });
	}

	const gfx = fam.get('graphics') ?? null;
	const roles = active.map((role) => {
		const family = fam.get(role) ?? null;
		return { role, family, queues: queues.get(role) ?? [], withGraphics: role !== 'graphics' && family !== null && family === gfx };
	});
	return { roles, infos, errors };
}
