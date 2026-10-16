// @ts-check
// The whole site config comes from the course engine; this file only names the course and its manifests.
import { defineCourse } from 'course-kit';

// Published as a GitHub Pages project site: https://ssoj13.github.io/course_vulkan/
export default defineCourse({
	site: 'https://ssoj13.github.io',
	base: '/course_vulkan',
	repo: 'https://github.com/ssoj13/course_vulkan',
	course: new URL('./src/data/course.json', import.meta.url),
	terms: new URL('./src/data/terms.json', import.meta.url),
	// English at the base path, Russian under /ru/. Russian lessons are authored for the Russian reader, not translated
	// sentence by sentence (see AGENTS.md "Writing").
	languages: [
		{ code: 'en', label: 'English' },
		{ code: 'ru', label: 'Русский' },
	],
});
