import { expect, test } from 'bun:test';
import { vuidRefs } from '../scripts/vuid-refs';
import { hasVuid, vuid } from '../src/lib/vuids';

test('every VUID mentioned in src/ has its registry text in vuids.json (else run bun scripts/sync-vuids.ts)', async () => {
	const refs = await vuidRefs();
	expect(refs.size).toBeGreaterThan(0);
	const unsynced = [...refs].filter(([id]) => !hasVuid(id) || vuid(id).trim() === '').map(([id, files]) => `${id} (${files.join(', ')})`);
	expect(unsynced).toEqual([]);
});

test('an unknown VUID throws instead of showing an empty quote', () => {
	expect(() => vuid('VUID-vkNotACommand-none-00000')).toThrow('missing from vuids.json');
});
