/**
 * Where the code a lesson shows lives in the repository: the files of one example step, the 1-based line range of a
 * `#region` (for `#L<a>-L<b>` links on GitHub) and the commands that build and run the step. Pure functions over
 * source text, used at build time by Step.astro and Snippet.astro (tests/source.test.ts checks them against every
 * region the lessons show).
 */

/**
 * A region marker line. Same rule as course-kit's lib/regions.ts (which extracts the region text but does not expose
 * line numbers); tests/source.test.ts proves both agree on every region the lessons use.
 */
const MARKER = /^\s*(?:\/\/|#|--|;)\s*#(region|endregion)\b\s*([\w.-]*)\s*$/;

/** 1-based, inclusive line range in the original file. */
export interface Lines {
	from: number;
	to: number;
}

/**
 * Lines of region `name` in `source`: from its first to its last non-blank line between the markers (nested markers
 * included, as they sit inside the range on GitHub). Throws when the region is missing or not closed.
 */
export function regionLines(source: string, name: string, where = 'source'): Lines {
	const lines = source.split(/\r?\n/);
	let start = -1;
	let depth = 0;
	for (const [i, line] of lines.entries()) {
		const m = MARKER.exec(line);
		if (start < 0) {
			if (m?.[1] === 'region' && m[2] === name) start = i;
			continue;
		}
		if (m?.[1] === 'region') depth++;
		else if (m?.[1] === 'endregion' && depth > 0) depth--;
		else if (m?.[1] === 'endregion') {
			let a = start + 1;
			let b = i - 1;
			while (a <= b && !lines[a]!.trim()) a++;
			while (b >= a && !lines[b]!.trim()) b--;
			if (a > b) throw new Error(`${where}: region "${name}" is empty`);
			return { from: a + 1, to: b + 1 };
		}
	}
	throw new Error(start < 0 ? `${where}: no region "${name}"` : `${where}: region "${name}" is not closed`);
}

/** GitHub link to `path` (from the repository root) on the main branch, optionally to a line range. */
export function blobUrl(repo: string, path: string, lines?: Lines): string {
	const anchor = lines ? `#L${lines.from}-L${lines.to}` : '';
	return `${repo.replace(/\/$/, '')}/blob/main/${path}${anchor}`;
}

/** C++ preset whose build directory the run command uses (examples/cpp/CMakePresets.json). */
export const CPP_PRESET = 'msvc-release';

/** One example step in both languages: source paths, the working directory and the command that runs it. */
export interface StepSource {
	path: string;
	dir: string;
	run: string;
}

/**
 * Sources of step `file` (`22_depth`, or `35_capstone/main` for a step with several files). The binary / CMake target
 * is the first path segment: Cargo names `src/bin/<dir>/main.rs` after `<dir>`, and the C++ project names a target
 * after its step.
 */
export function stepSources(file: string): { rust: StepSource; cpp: StepSource } {
	const target = file.split('/')[0]!;
	return {
		rust: {
			path: `examples/rust/src/bin/${file}.rs`,
			dir: 'examples/rust',
			run: `cargo run --release --bin ${target}`,
		},
		cpp: {
			path: `examples/cpp/src/${file}.cpp`,
			dir: 'examples/cpp',
			// Ninja is single-config: the executable lands directly in the preset's binaryDir, build/<preset>.
			run: `cmake --build --preset ${CPP_PRESET} --target ${target} && .\\build\\${CPP_PRESET}\\${target}.exe`,
		},
	};
}
