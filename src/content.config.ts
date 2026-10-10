import { defineCollection } from 'astro:content';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';

export const collections = {
	docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
	// Declared (with src/content/i18n/en.json = {}) so Astro 7 does not log "collection i18n does not exist" on every build.
	i18n: defineCollection({ loader: i18nLoader(), schema: i18nSchema() }),
};
