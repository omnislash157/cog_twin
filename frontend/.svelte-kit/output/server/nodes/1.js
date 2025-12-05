

export const index = 1;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/fallbacks/error.svelte.js')).default;
export const imports = ["_app/immutable/nodes/1.Bj0v9h5n.js","_app/immutable/chunks/B2QSKqcC.js","_app/immutable/chunks/6VynnOCg.js","_app/immutable/chunks/WM2WYO2y.js"];
export const stylesheets = [];
export const fonts = [];
