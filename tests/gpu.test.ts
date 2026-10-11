import { describe, expect, test } from 'bun:test';
import gpu from '../src/data/gpu.json';

const GiB = 1024 ** 3;
const isBoolMap = (o: object) => Object.values(o).every((v) => typeof v === 'boolean');

describe('gpu.json (scripts/sync-gpu.ts snapshot of the dev machine)', () => {
	test('shape', () => {
		expect(typeof gpu.meta.date).toBe('string');
		expect(gpu.meta.date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
		for (const k of ['name', 'type', 'apiVersion', 'driverVersion'] as const) expect(typeof gpu.device[k]).toBe('string');
		for (const k of ['driverVersionRaw', 'vendorID', 'deviceID'] as const) expect(Number.isInteger(gpu.device[k])).toBe(true);

		expect(Object.keys(gpu.features)).toEqual([
			'VkPhysicalDeviceFeatures',
			'VkPhysicalDeviceVulkan11Features',
			'VkPhysicalDeviceVulkan12Features',
			'VkPhysicalDeviceVulkan13Features',
			'VkPhysicalDeviceVulkan14Features',
		]);
		for (const map of Object.values(gpu.features)) expect(isBoolMap(map)).toBe(true);
		// VkPhysicalDeviceFeatures has 55 members in every Vulkan version.
		expect(Object.keys(gpu.features.VkPhysicalDeviceFeatures)).toHaveLength(55);

		const limits = Object.values(gpu.limits);
		expect(limits.length).toBeGreaterThan(100);
		for (const v of limits) expect(typeof v === 'number' || typeof v === 'boolean' || Array.isArray(v)).toBe(true);

		gpu.queueFamilies.forEach((q, i) => {
			expect(q.index).toBe(i);
			expect(q.queueCount).toBeGreaterThan(0);
			expect(q.queueFlags.every((f) => f.startsWith('VK_QUEUE_'))).toBe(true);
			expect(Object.keys(q.minImageTransferGranularity)).toEqual(['width', 'height', 'depth']);
		});

		for (const h of gpu.memory.heaps) {
			expect(Number.isSafeInteger(h.size)).toBe(true);
			expect(h.flags.every((f) => f.startsWith('VK_MEMORY_HEAP_'))).toBe(true);
		}
		for (const t of gpu.memory.types) {
			expect(t.heapIndex).toBeLessThan(gpu.memory.heaps.length);
			expect(t.propertyFlags.every((f) => f.startsWith('VK_MEMORY_PROPERTY_'))).toBe(true);
		}
	});

	test('matches what the lessons quote (RTX 3080 Ti, step 03 / step 04 output)', () => {
		expect(gpu.device.name).toBe('NVIDIA GeForce RTX 3080 Ti');
		expect(gpu.device.type).toBe('VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU');
		expect(gpu.device.apiVersion.startsWith('1.4')).toBe(true);
		expect(gpu.device.vendorID).toBe(0x10de);

		// physical-devices.mdx, step 03 output.
		expect(gpu.features.VkPhysicalDeviceFeatures.samplerAnisotropy).toBe(true);
		expect(gpu.features.VkPhysicalDeviceVulkan13Features.dynamicRendering).toBe(true);
		expect(gpu.features.VkPhysicalDeviceVulkan13Features.synchronization2).toBe(true);
		expect(gpu.features.VkPhysicalDeviceVulkan14Features.pushDescriptor).toBe(true);
		expect(gpu.features.VkPhysicalDeviceVulkan14Features.maintenance5).toBe(true);
		expect(gpu.limits.maxImageDimension2D).toBe(32768);
		expect(gpu.limits.maxPushConstantsSize).toBe(256);
		expect(gpu.limits.minUniformBufferOffsetAlignment).toBe(64);
		expect(gpu.limits.timestampPeriod).toBe(1);

		// queue-families.mdx table: family -> queue count and flags.
		const table = gpu.queueFamilies.map((q) => [q.queueCount, q.queueFlags.map((f) => f.replace(/^VK_QUEUE_|_BIT(_\w+)?$/g, ''))]);
		expect(table).toEqual([
			[16, ['GRAPHICS', 'COMPUTE', 'TRANSFER', 'SPARSE_BINDING']],
			[2, ['TRANSFER', 'SPARSE_BINDING']],
			[8, ['COMPUTE', 'TRANSFER', 'SPARSE_BINDING']],
			[1, ['TRANSFER', 'SPARSE_BINDING', 'VIDEO_DECODE']],
			[1, ['TRANSFER', 'SPARSE_BINDING', 'VIDEO_ENCODE']],
			[1, ['TRANSFER', 'SPARSE_BINDING', 'OPTICAL_FLOW']],
		]);

		// gpu.mdx: "12 GB of GDDR6X". The driver reports a bit less than 12 GiB as the main DEVICE_LOCAL heap.
		const vram = Math.max(...gpu.memory.heaps.filter((h) => h.flags.includes('VK_MEMORY_HEAP_DEVICE_LOCAL_BIT')).map((h) => h.size));
		expect(vram).toBeGreaterThan(11 * GiB);
		expect(vram).toBeLessThanOrEqual(12 * GiB);
	});
});
