/**
 * Interface texts of this course's own widgets (ObjectGraph and its simulator, the source line under Step/Snippet code),
 * per language. The engine's texts live in course-kit (i18n.mjs); this table only adds what the Vulkan widgets show. Placeholders `{name}` are filled with
 * the engine's `fmt`. tests/i18n.test.ts checks that every language has the same keys and placeholders.
 */
import type { SimMsg } from './lib/objects';

const EN = {
	hint: 'Click an object to see what it is, how it is made and what it is made from. Drag boxes to untangle.',
	legendNote: 'Roots on top; wires run down to the objects made from them.',
	asText: 'These objects as text',
	madeFrom: 'made from {parents}',
	releasedWith: 'released with {call}',
	glossary: 'glossary',
	optional: '(optional)',
	enableExt: 'enable {ext} + validation layer',
	create: 'Create',
	enumerate: 'Enumerate',
	get: 'Get',
	allocate: 'Allocate',
	destroy: 'Destroy',
	free: 'Free',
	destroyQ: 'Destroy?',
	progress: 'Created {n}/{total} · validation errors: {errors}',
	startOver: 'Start over',
	startHint: 'Pick an object and press Create. Start at the top: that is where the wires start.',
	// How a handle is obtained (node caption, badge)
	howCreate: 'created',
	howAllocate: 'allocated',
	howEnumerate: 'enumerated',
	howGet: 'retrieved',
	// Course stages (legend, node caption)
	stageInstance: 'instance',
	stageDevices: 'devices',
	stagePresentation: 'presentation',
	stagePipeline: 'pipeline',
	stageCommands: 'commands',
	stageSync: 'sync',
	stageMemory: 'memory',
	stageDescriptors: 'descriptors',
	stageImages: 'images',
	// Simulator remarks (lib/objects.ts SimMsg)
	simAlreadyExists: '{obj} already exists (this exercise keeps one of each).',
	simNotLoaded: '{call} is not available: its function pointer is only loaded when {ext} was enabled in vkCreateInstance.',
	simDoesNotExist: '{obj} does not exist.',
	simBackToPool: '{obj} goes back to {owner} when the pool is reset or destroyed.',
	simNotDestroyed: '{obj} is not destroyed by the application: it was obtained with {call} and lives as long as {owner}.',
	simStillAlive: 'Still alive: {objs}. Destroy them first.',
	simReleased: 'Released with it: {objs}.',
	simDangling: 'Now dangling (any use is invalid): {objs}.',
	// Source line under lesson code (Step, Snippet)
	srcLines: 'lines {from}–{to} on GitHub',
	srcFile: 'whole file on GitHub',
	srcRun: 'run in {dir}:',
};

export type CourseText = typeof EN;

const RU: CourseText = {
	hint: 'Щёлкните по объекту: что это, как получается и из чего делается. Блоки можно растаскивать мышью.',
	legendNote: 'Корни наверху; связи идут вниз — к объектам, которые из них создаются.',
	asText: 'Эти объекты текстом',
	madeFrom: 'создаётся из {parents}',
	releasedWith: 'освобождается вызовом {call}',
	glossary: 'глоссарий',
	optional: '(необязательно)',
	enableExt: 'включить {ext} и слой валидации',
	create: 'Создать',
	enumerate: 'Перечислить',
	get: 'Получить',
	allocate: 'Выделить',
	destroy: 'Уничтожить',
	free: 'Освободить',
	destroyQ: 'Уничтожить?',
	progress: 'Создано {n}/{total} · ошибок валидации: {errors}',
	startOver: 'Начать заново',
	startHint: 'Выберите объект и нажмите «Создать». Начинайте сверху — оттуда идут связи.',
	howCreate: 'создаётся',
	howAllocate: 'выделяется',
	howEnumerate: 'перечисляется',
	howGet: 'получается',
	stageInstance: 'instance',
	stageDevices: 'устройства',
	stagePresentation: 'вывод на экран',
	stagePipeline: 'пайплайн',
	stageCommands: 'команды',
	stageSync: 'синхронизация',
	stageMemory: 'память',
	stageDescriptors: 'дескрипторы',
	stageImages: 'изображения',
	simAlreadyExists: '{obj} уже существует (в этом упражнении — по одному объекту каждого типа).',
	simNotLoaded: '{call} недоступна: указатель на неё загружается, только если в vkCreateInstance было включено {ext}.',
	simDoesNotExist: '{obj} не существует.',
	simBackToPool: '{obj} вернётся в {owner}, когда пул сбросят или уничтожат.',
	simNotDestroyed: '{obj} не уничтожается приложением: он получен через {call} и живёт, пока жив {owner}.',
	simStillAlive: 'Ещё живы: {objs}. Сначала уничтожьте их.',
	simReleased: 'Освобождены вместе с ним: {objs}.',
	simDangling: 'Теперь висячие (любое использование — ошибка): {objs}.',
	srcLines: 'строки {from}–{to} на GitHub',
	srcFile: 'весь файл на GitHub',
	srcRun: 'запуск в {dir}:',
};

export const COURSE_TEXT: Readonly<Record<string, CourseText>> = { en: EN, ru: RU };

/** This course's widget texts in a language; throws for a language without them. */
export function courseText(lang: string): CourseText {
	const t = COURSE_TEXT[lang];
	if (!t) throw new Error(`src/i18n.ts has no texts for "${lang}"`);
	return t;
}

/** Text key of a simulator remark. */
export const SIM_KEY: Record<SimMsg, keyof CourseText> = {
	alreadyExists: 'simAlreadyExists',
	notLoaded: 'simNotLoaded',
	doesNotExist: 'simDoesNotExist',
	backToPool: 'simBackToPool',
	notDestroyed: 'simNotDestroyed',
	stillAlive: 'simStillAlive',
	released: 'simReleased',
	dangling: 'simDangling',
};

/** Text key of a course stage name. */
export const STAGE_KEY = {
	instance: 'stageInstance',
	devices: 'stageDevices',
	presentation: 'stagePresentation',
	pipeline: 'stagePipeline',
	commands: 'stageCommands',
	sync: 'stageSync',
	memory: 'stageMemory',
	descriptors: 'stageDescriptors',
	images: 'stageImages',
} as const satisfies Record<string, keyof CourseText>;

/** Text key of how a handle is obtained. */
export const HOW_KEY = { create: 'howCreate', allocate: 'howAllocate', enumerate: 'howEnumerate', get: 'howGet' } as const satisfies Record<
	string,
	keyof CourseText
>;
