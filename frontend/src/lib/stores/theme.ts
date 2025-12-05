import { writable } from 'svelte/store';

type Theme = 'cyber' | 'normie';

function createThemeStore() {
	const { subscribe, set, update } = writable<Theme>('cyber');

	return {
		subscribe,
		set,
		toggle: () => update(t => t === 'cyber' ? 'normie' : 'cyber'),
	};
}

export const theme = createThemeStore();
export const toggleTheme = () => theme.toggle();
