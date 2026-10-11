import { describe, expect, test } from 'bun:test';
import { TEXT } from '../../src/components/labs/messenger-filter.text';
import { COURSE, DAB, LOG, SEVERITIES, TYPES, blockedBy, maskErrors, passes, produced, runTotals, snippet, vuidText } from '../../src/lib/labs/messenger-filter';
import { textErrors } from '../../src/lib/text';

const bit = (list: readonly { name: string; bit: number }[], ...names: string[]) => list.filter((x) => names.includes(x.name)).reduce((a, x) => a | x.bit, 0);
const ALL = { severity: bit(SEVERITIES, 'VERBOSE', 'INFO', 'WARNING', 'ERROR'), type: bit(TYPES, 'GENERAL', 'VALIDATION', 'PERFORMANCE') };

test('texts: every language has the same keys and placeholders', () => {
	expect(textErrors(TEXT, 'messenger-filter')).toEqual([]);
});

describe('flag bits match vk.xml', () => {
	test('severity: VERBOSE 0x1, INFO 0x10, WARNING 0x100, ERROR 0x1000', () => {
		expect(SEVERITIES.map((s) => s.bit)).toEqual([0x1, 0x10, 0x100, 0x1000]);
	});
	test('type: GENERAL 0x1, VALIDATION 0x2, PERFORMANCE 0x4, DEVICE_ADDRESS_BINDING_EXT 0x8', () => {
		expect(TYPES.map((t) => t.bit)).toEqual([0x1, 0x2, 0x4, 0x8]);
		expect(DAB).toBe(0x8);
	});
	test('the course mask is WARNING | ERROR over the three core types', () => {
		expect(COURSE).toEqual({ severity: 0x1100, type: 0x7 });
	});
});

describe('the real log', () => {
	test('the sample uses only known severities and types and is part of the counted run', () => {
		for (const m of LOG.messages) {
			expect(SEVERITIES.some((s) => s.name === m.severity)).toBe(true);
			expect(TYPES.some((t) => t.name === m.type)).toBe(true);
		}
		expect(LOG.run.counts.reduce((a, c) => a + c.n, 0)).toBe(LOG.run.total);
		expect(LOG.run.extra.leak).toBe(LOG.messages.filter((m) => m.when === 'leak').length);
		expect(LOG.run.extra.dab).toBe(LOG.messages.filter((m) => m.when === 'dab').length);
	});
	test('the lesson’s facts: the Bandicam warning and the 00629 error are in it', () => {
		expect(LOG.messages.some((m) => m.severity === 'WARNING' && m.message.includes('VK_LAYER_bandicam_helper'))).toBe(true);
		expect(LOG.messages.some((m) => m.id === 'VUID-vkDestroyInstance-instance-00629' && m.when === 'leak')).toBe(true);
	});
	test('the numbers quoted in the texts: 108 VERBOSE+INFO of 109, 107 of them from the loader', () => {
		const n = (f: (c: (typeof LOG.run.counts)[number]) => boolean) => LOG.run.counts.filter(f).reduce((a, c) => a + c.n, 0);
		expect(LOG.run.total).toBe(109);
		expect(n((c) => c.severity === 'VERBOSE' || c.severity === 'INFO')).toBe(108);
		expect(n((c) => (c.severity === 'VERBOSE' || c.severity === 'INFO') && c.type === 'GENERAL')).toBe(107);
	});
});

describe('filtering', () => {
	test('a message passes only when both its severity and its type are in the masks', () => {
		const warn = LOG.messages.find((m) => m.severity === 'WARNING')!;
		expect(passes(warn, COURSE)).toBe(true);
		expect(blockedBy(warn, { severity: 0x1000, type: 0x7 })).toBe('severity');
		expect(blockedBy(warn, { severity: 0x1100, type: 0x2 })).toBe('type');
	});
	test('the course mask lets through exactly the warning in a normal run, plus the errors when a rule is broken', () => {
		const got = (leak: boolean) => produced(COURSE, leak).filter((m) => passes(m, COURSE));
		expect(got(false).map((m) => m.severity)).toEqual(['WARNING']);
		expect(got(true).map((m) => m.severity)).toEqual(['WARNING', 'ERROR', 'ERROR']);
		expect(runTotals(COURSE, false)).toEqual({ got: 1, of: 109 });
		expect(runTotals(COURSE, true)).toEqual({ got: 3, of: 111 });
	});
	test('everything enabled delivers every message of the run', () => {
		expect(runTotals(ALL, false)).toEqual({ got: 109, of: 109 });
	});
	test('asking for DEVICE_ADDRESS_BINDING_EXT on the instance produces the real messageType error', () => {
		expect(produced(COURSE, false).some((m) => m.when === 'dab')).toBe(false);
		const dab = { ...COURSE, type: COURSE.type | 0x8 };
		expect(produced(dab, false).filter((m) => m.when === 'dab').map((m) => m.id)).toEqual(['VUID-VkDebugUtilsMessengerCreateInfoEXT-messageType-parameter']);
		expect(maskErrors(dab)).toEqual(['VUID-VkDebugUtilsMessengerCreateInfoEXT-messageType-parameter']);
	});
	test('empty masks break the requiredbitmask rules; every VUID the lab quotes has its registry text', () => {
		const errs = maskErrors({ severity: 0, type: 0 });
		expect(errs).toEqual(['VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-requiredbitmask', 'VUID-VkDebugUtilsMessengerCreateInfoEXT-messageType-requiredbitmask']);
		for (const v of [...errs, ...maskErrors({ severity: 1, type: 0x8 })]) {
			expect(LOG.maskVuids).toContain(v);
			expect(vuidText(v)).toContain('must');
		}
	});
});

describe('create-info code', () => {
	test('C uses the real enumerant names and the mask values', () => {
		const c = snippet(COURSE, 'c');
		expect(c).toContain('VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |');
		expect(c).toContain('VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT, // 0x1100');
		expect(c).toContain('VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT, // 0x0007');
		expect(snippet({ severity: 1, type: 0x8 }, 'c')).toContain('VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT');
		expect(snippet({ severity: 0, type: 1 }, 'c')).toContain('.messageSeverity = 0, // 0x0000');
	});
	test('Rust matches step 02 and ash master’s constant names', () => {
		const r = snippet(COURSE, 'rust');
		expect(r).toContain('vk::DebugUtilsMessageSeverityFlagsEXT::WARNING\n            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR');
		expect(snippet({ severity: 1, type: 0x8 }, 'rust')).toContain('vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING,');
		expect(snippet({ severity: 0, type: 1 }, 'rust')).toContain('vk::DebugUtilsMessageSeverityFlagsEXT::empty()');
	});
});
