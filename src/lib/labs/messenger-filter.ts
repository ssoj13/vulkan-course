/**
 * Messenger-filter lab model: which messages of a real validation log reach a debug messenger callback for given
 * messageSeverity / messageType masks, and the create-info code for those masks.
 *
 * Rules (Vulkan spec, "Debugging" chapter; bits and values from vk.xml of SDK 1.4.341.1):
 * - a message reaches the callback when its severity bit is in messageSeverity AND one of its type bits is in
 *   messageType;
 * - both masks must not be 0 (VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-requiredbitmask,
 *   -messageType-requiredbitmask);
 * - DEVICE_ADDRESS_BINDING_EXT belongs to the device extension VK_EXT_device_address_binding_report; asking for it on a
 *   messenger chained to vkCreateInstance made the validation layer report the real error stored in the log
 *   (`when: "dab"`).
 * The log (src/data/validation-log.json) is a sample of a real run of step 02 with every severity and type enabled;
 * `run.counts` are the totals of that whole run (109 messages; the terminal ran elevated, which adds a few loader lines
 * about ignored environment variables). Pure module: no DOM, no React; tested by tests/labs/messenger-filter.test.ts.
 */
import logJson from '../../data/validation-log.json';
import vuids from '../../data/vuids.json';

/** VkDebugUtilsMessageSeverityFlagBitsEXT, in vk.xml order. */
export const SEVERITIES = [
	{ name: 'VERBOSE', bit: 0x1 },
	{ name: 'INFO', bit: 0x10 },
	{ name: 'WARNING', bit: 0x100 },
	{ name: 'ERROR', bit: 0x1000 },
] as const;
/** VkDebugUtilsMessageTypeFlagBitsEXT; the last one comes from VK_EXT_device_address_binding_report. */
export const TYPES = [
	{ name: 'GENERAL', bit: 0x1 },
	{ name: 'VALIDATION', bit: 0x2 },
	{ name: 'PERFORMANCE', bit: 0x4 },
	{ name: 'DEVICE_ADDRESS_BINDING_EXT', bit: 0x8 },
] as const;
export type Severity = (typeof SEVERITIES)[number]['name'];
export type MsgType = (typeof TYPES)[number]['name'];

/** Extra situation a message belongs to: the lesson's leak experiment, or asking for DEVICE_ADDRESS_BINDING_EXT. */
export type When = 'leak' | 'dab';

export interface LogMessage {
	severity: Severity;
	type: MsgType;
	/** pMessageIdName: "Loader Message" for the loader, the VUID or check name for the validation layer. */
	id: string;
	message: string;
	when?: When;
}

interface LogFile {
	source: string;
	/** VUIDs the lab quotes about the masks; listed in the data so scripts/sync-vuids.ts copies their texts. */
	maskVuids: string[];
	run: { total: number; counts: { severity: Severity; type: MsgType; n: number }[]; extra: Record<When, number> };
	messages: LogMessage[];
}
export const LOG = logJson as LogFile;

const SEV_BIT = Object.fromEntries(SEVERITIES.map((s) => [s.name, s.bit])) as Record<Severity, number>;
const TYPE_BIT = Object.fromEntries(TYPES.map((t) => [t.name, t.bit])) as Record<MsgType, number>;
export const DAB = TYPE_BIT.DEVICE_ADDRESS_BINDING_EXT;

/** The course's choice in step 02: warnings and errors of the three core types. */
export const COURSE = { severity: SEV_BIT.WARNING | SEV_BIT.ERROR, type: TYPE_BIT.GENERAL | TYPE_BIT.VALIDATION | TYPE_BIT.PERFORMANCE };

export interface Masks {
	severity: number;
	type: number;
}

/** The spec's filter: severity bit in the severity mask and type bit in the type mask. */
export const passes = (m: LogMessage, k: Masks): boolean => (SEV_BIT[m.severity] & k.severity) !== 0 && (TYPE_BIT[m.type] & k.type) !== 0;

/** Why a message is filtered out (`undefined` = it reaches the callback). Severity is checked first. */
export function blockedBy(m: LogMessage, k: Masks): 'severity' | 'type' | undefined {
	if ((SEV_BIT[m.severity] & k.severity) === 0) return 'severity';
	if ((TYPE_BIT[m.type] & k.type) === 0) return 'type';
	return undefined;
}

/** The messages the layer and loader produce in this setup (before any filtering). */
export function produced(k: Masks, leak: boolean): LogMessage[] {
	return LOG.messages.filter((m) => (m.when === 'leak' ? leak : m.when === 'dab' ? (k.type & DAB) !== 0 : true));
}

/** Messages of the whole real run (not only the sample) that reach the callback, and how many were produced. */
export function runTotals(k: Masks, leak: boolean): { got: number; of: number } {
	const extra = [...(leak ? LOG.messages.filter((m) => m.when === 'leak') : []), ...((k.type & DAB) !== 0 ? LOG.messages.filter((m) => m.when === 'dab') : [])];
	const base = LOG.run.counts.reduce((a, c) => a + ((SEV_BIT[c.severity] & k.severity) !== 0 && (TYPE_BIT[c.type] & k.type) !== 0 ? c.n : 0), 0);
	return { got: base + extra.filter((m) => passes(m, k)).length, of: LOG.run.total + extra.length };
}

/** Valid-usage problems of the masks themselves, as VUIDs (texts in vuids.json). */
export function maskErrors(k: Masks): string[] {
	const errs: string[] = [];
	if (k.severity === 0) errs.push('VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-requiredbitmask');
	if (k.type === 0) errs.push('VUID-VkDebugUtilsMessengerCreateInfoEXT-messageType-requiredbitmask');
	if ((k.type & DAB) !== 0) errs.push('VUID-VkDebugUtilsMessengerCreateInfoEXT-messageType-parameter');
	return errs;
}

/** The registry text of a VUID (plain text, synced by scripts/sync-vuids.ts). */
export function vuidText(vuid: string): string {
	const t = (vuids as Record<string, string>)[vuid];
	if (!t) throw new Error(`${vuid} missing from vuids.json (run scripts/sync-vuids.ts)`);
	return t;
}

export const hex = (n: number) => `0x${n.toString(16).padStart(4, '0')}`;

/** Create-info code for the masks, C (designated initializers) or Rust (ash master). */
export function snippet(k: Masks, lang: 'c' | 'rust'): string {
	const sev = SEVERITIES.filter((s) => s.bit & k.severity).map((s) => s.name);
	const typ = TYPES.filter((t) => t.bit & k.type).map((t) => t.name);
	if (lang === 'c') {
		const c = (prefix: string, names: string[], pad: string) =>
			names.length ? names.map((n) => `VK_DEBUG_UTILS_MESSAGE_${prefix}_${n.replace(/_EXT$/, '')}_BIT_EXT`).join(` |\n${pad}`) : '0';
		return [
			'VkDebugUtilsMessengerCreateInfoEXT info = {',
			'    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,',
			`    .messageSeverity = ${c('SEVERITY', sev, ' '.repeat(23))}, // ${hex(k.severity)}`,
			`    .messageType = ${c('TYPE', typ, ' '.repeat(19))}, // ${hex(k.type)}`,
			'    .pfnUserCallback = debug_callback,',
			'};',
		].join('\n');
	}
	// ash master names the extension bit without the _EXT suffix (DEVICE_ADDRESS_BINDING).
	const r = (ty: string, names: string[]) =>
		names.length ? names.map((n) => `vk::${ty}::${n.replace(/_EXT$/, '')}`).join(`\n            | `) : `vk::${ty}::empty()`;
	return [
		'let info = vk::DebugUtilsMessengerCreateInfoEXT::default()',
		`    .message_severity( // ${hex(k.severity)}`,
		`        ${r('DebugUtilsMessageSeverityFlagsEXT', sev)},`,
		'    )',
		`    .message_type( // ${hex(k.type)}`,
		`        ${r('DebugUtilsMessageTypeFlagsEXT', typ)},`,
		'    )',
		'    .pfn_user_callback(Some(debug_callback));',
	].join('\n');
}
