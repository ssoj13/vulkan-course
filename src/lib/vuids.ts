/**
 * Valid-usage texts of the Vulkan spec, the one way code reads them. src/data/vuids.json is the only place with VUID
 * texts: scripts/sync-vuids.ts copies them from the SDK's validusage.json for every `VUID-...` id mentioned in src/,
 * so mentioning an id in a model or lesson and rerunning the script is all a new quote needs.
 */
import vuidsJson from '../data/vuids.json';

const TEXTS: Readonly<Record<string, string>> = vuidsJson;

/** The id is in vuids.json. */
export const hasVuid = (id: string): boolean => Object.hasOwn(TEXTS, id);

/** Plain registry text of a VUID. Throws for an id that was never synced (run `bun scripts/sync-vuids.ts`). */
export function vuid(id: string): string {
	if (!hasVuid(id)) throw new Error(`${id} missing from vuids.json (run bun scripts/sync-vuids.ts)`);
	return TEXTS[id] as string;
}

/** A validation-layer style report quoting the registry text (the format the course's simulators print). */
export function vuReport(id: string, call: string): string {
	return `Validation Error: [ ${id} ] ${call}(): The Vulkan spec states: ${vuid(id)}`;
}
