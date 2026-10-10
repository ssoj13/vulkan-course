// @ts-check
// The whole site config comes from the course engine; this file only names the course and its manifests.
import { defineCourse } from 'course-kit';

// Published as a GitHub Pages project site: https://ssoj13.github.io/vulkan-course/
export default defineCourse({
	site: 'https://ssoj13.github.io',
	base: '/vulkan-course',
	repo: 'https://github.com/ssoj13/vulkan-course',
	course: new URL('./src/data/course.json', import.meta.url),
	terms: new URL('./src/data/terms.json', import.meta.url),
});
