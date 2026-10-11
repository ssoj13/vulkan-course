import { expect, test } from 'bun:test';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { extractRegion } from 'course-kit/lib/regions.ts';
import { CPP_PRESET, blobUrl, regionLines, stepSources } from '../src/lib/source';

const ROOT = join(import.meta.dir, '..');
const read = (path: string) => readFileSync(join(ROOT, path), 'utf8');
const MARKER = /^\s*(?:\/\/|#|--|;)\s*#(region|endregion)\b/;

/** Every `path#region` the lessons show through <Step> (both languages) and <Snippet src="...#region">. */
function shownRegions(): Set<string> {
	const out = new Set<string>();
	const dir = join(ROOT, 'src/content/docs');
	for (const f of readdirSync(dir, { recursive: true, encoding: 'utf8' })) {
		if (!f.endsWith('.mdx')) continue;
		const text = readFileSync(join(dir, f), 'utf8');
		for (const m of text.matchAll(/<Step\s+file="([^"]+)"\s+region="([^"]+)"/g)) {
			const s = stepSources(m[1]!);
			out.add(`${s.rust.path}#${m[2]}`).add(`${s.cpp.path}#${m[2]}`);
		}
		for (const m of text.matchAll(/<Snippet\s+src="([^"#]+#[^"]+)"/g)) out.add(m[1]!);
	}
	return out;
}

test('the line range of every shown region holds exactly the code the lesson shows', () => {
	const regions = shownRegions();
	expect(regions.size).toBeGreaterThan(100);
	for (const src of regions) {
		const [path, region] = src.split('#') as [string, string];
		const text = read(path);
		const { from, to } = regionLines(text, region, path);
		const lines = text.split(/\r?\n/).slice(from - 1, to);
		// Same text once markers are dropped and indentation removed (what course-kit's extractRegion returns).
		const body = lines.filter((l) => !MARKER.test(l));
		const indent = Math.min(...body.filter((l) => l.trim()).map((l) => l.length - l.trimStart().length));
		expect(body.map((l) => l.slice(Math.min(indent, l.length - l.trimStart().length))).join('\n'), src).toBe(
			extractRegion(text, region, path),
		);
		expect(lines[0]!.trim(), src).not.toBe('');
		expect(lines.at(-1)!.trim(), src).not.toBe('');
	}
});

test('regionLines: nested regions, blank edges, errors', () => {
	const src = ['a', '// #region outer', '', 'x', '  // #region inner', '  y', '  // #endregion', '', '// #endregion', 'z'].join('\r\n');
	expect(regionLines(src, 'outer')).toEqual({ from: 4, to: 7 });
	expect(regionLines(src, 'inner')).toEqual({ from: 6, to: 6 });
	expect(() => regionLines(src, 'nope')).toThrow('no region "nope"');
	expect(() => regionLines('// #region open\nx', 'open')).toThrow('not closed');
});

test('step paths, commands and GitHub links', () => {
	const one = stepSources('22_depth');
	expect(one.rust).toEqual({ path: 'examples/rust/src/bin/22_depth.rs', dir: 'examples/rust', run: 'cargo run --release --bin 22_depth' });
	expect(one.cpp.path).toBe('examples/cpp/src/22_depth.cpp');
	expect(one.cpp.run).toBe('cmake --build --preset msvc-release --target 22_depth && .\\build\\msvc-release\\22_depth.exe');
	// A step with several files: the binary / target is named after its folder.
	const many = stepSources('35_capstone/main');
	expect(many.rust.path).toBe('examples/rust/src/bin/35_capstone/main.rs');
	expect(many.rust.run).toBe('cargo run --release --bin 35_capstone');
	expect(blobUrl('https://github.com/o/r/', 'a/b.rs', { from: 3, to: 9 })).toBe('https://github.com/o/r/blob/main/a/b.rs#L3-L9');
	expect(blobUrl('https://github.com/o/r', 'shaders/x.slang')).toBe('https://github.com/o/r/blob/main/shaders/x.slang');
});

test('the C++ preset of the run command exists and builds into build/<preset>', () => {
	const presets = JSON.parse(read('examples/cpp/CMakePresets.json')) as {
		configurePresets: { name: string; binaryDir?: string }[];
		buildPresets: { name: string }[];
	};
	expect(presets.buildPresets.map((p) => p.name)).toContain(CPP_PRESET);
	expect(presets.configurePresets.find((p) => p.name === 'base')?.binaryDir).toBe('${sourceDir}/build/${presetName}');
});
