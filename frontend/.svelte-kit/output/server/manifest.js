export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {start:"_app/immutable/entry/start.BE6V0aIV.js",app:"_app/immutable/entry/app.BeTdsDiV.js",imports:["_app/immutable/entry/start.BE6V0aIV.js","_app/immutable/chunks/WM2WYO2y.js","_app/immutable/chunks/B2QSKqcC.js","_app/immutable/entry/app.BeTdsDiV.js","_app/immutable/chunks/B2QSKqcC.js","_app/immutable/chunks/6VynnOCg.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
		nodes: [
			__memo(() => import('./nodes/0.js')),
			__memo(() => import('./nodes/1.js'))
		],
		remotes: {
			
		},
		routes: [
			
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();
