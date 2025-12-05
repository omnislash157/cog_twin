import { writable, derived } from 'svelte/store';

export type PanelMode = 'docked' | 'floating' | 'fullscreen' | 'closed';

export interface PanelState {
	mode: PanelMode;
	// Position for floating mode
	x: number;
	y: number;
	// Size for floating/docked mode
	width: number;
	height: number;
	// Z-index for stacking floating panels
	zIndex: number;
	// Minimized state (collapsed to title bar only)
	minimized: boolean;
}

export interface PanelsState {
	memory3d: PanelState;
	chat: PanelState;
	artifacts: PanelState;
	analytics: PanelState;
	swarm: PanelState;
	// Track which panel is focused (highest z-index)
	focusedPanel: 'memory3d' | 'chat' | 'artifacts' | 'analytics' | 'swarm' | null;
	// Base z-index counter
	zCounter: number;
}

const DEFAULT_PANEL_STATE: Omit<PanelState, 'x' | 'y' | 'width' | 'height'> = {
	mode: 'docked',
	zIndex: 100,
	minimized: false
};

const initialState: PanelsState = {
	memory3d: {
		...DEFAULT_PANEL_STATE,
		x: 0,
		y: 0,
		width: 60, // percentage when docked
		height: 100
	},
	chat: {
		...DEFAULT_PANEL_STATE,
		x: 100,
		y: 50,
		width: 400, // pixels when floating
		height: 500
	},
	artifacts: {
		...DEFAULT_PANEL_STATE,
		x: 150,
		y: 100,
		width: 400,
		height: 400
	},
	analytics: {
		...DEFAULT_PANEL_STATE,
		x: 200,
		y: 150,
		width: 380,
		height: 450
	},
	swarm: {
		...DEFAULT_PANEL_STATE,
		x: 250,
		y: 50,
		width: 600,
		height: 500
	},
	focusedPanel: null,
	zCounter: 100
};

function createPanelsStore() {
	const { subscribe, set, update } = writable<PanelsState>(initialState);

	return {
		subscribe,

		// Set panel mode
		setMode(panel: keyof Omit<PanelsState, 'focusedPanel' | 'zCounter'>, mode: PanelMode) {
			update(state => {
				const newState = { ...state };
				newState[panel] = { ...state[panel], mode };

				// If going fullscreen, bring to front
				if (mode === 'fullscreen' || mode === 'floating') {
					newState.zCounter++;
					newState[panel].zIndex = newState.zCounter;
					newState.focusedPanel = panel;
				}

				return newState;
			});
		},

		// Toggle between docked and floating
		toggleFloat(panel: keyof Omit<PanelsState, 'focusedPanel' | 'zCounter'>) {
			update(state => {
				const current = state[panel].mode;
				const newMode: PanelMode = current === 'docked' ? 'floating' : 'docked';
				const newState = { ...state };
				newState[panel] = { ...state[panel], mode: newMode };

				if (newMode === 'floating') {
					newState.zCounter++;
					newState[panel].zIndex = newState.zCounter;
					newState.focusedPanel = panel;
				}

				return newState;
			});
		},

		// Toggle fullscreen
		toggleFullscreen(panel: keyof Omit<PanelsState, 'focusedPanel' | 'zCounter'>) {
			update(state => {
				const current = state[panel].mode;
				const newMode: PanelMode = current === 'fullscreen' ? 'docked' : 'fullscreen';
				const newState = { ...state };
				newState[panel] = { ...state[panel], mode: newMode };

				if (newMode === 'fullscreen') {
					newState.zCounter++;
					newState[panel].zIndex = newState.zCounter;
					newState.focusedPanel = panel;
				}

				return newState;
			});
		},

		// Toggle minimized
		toggleMinimize(panel: keyof Omit<PanelsState, 'focusedPanel' | 'zCounter'>) {
			update(state => ({
				...state,
				[panel]: { ...state[panel], minimized: !state[panel].minimized }
			}));
		},

		// Close panel
		close(panel: keyof Omit<PanelsState, 'focusedPanel' | 'zCounter'>) {
			update(state => ({
				...state,
				[panel]: { ...state[panel], mode: 'closed' }
			}));
		},

		// Open panel (restore to docked)
		open(panel: keyof Omit<PanelsState, 'focusedPanel' | 'zCounter'>) {
			update(state => ({
				...state,
				[panel]: { ...state[panel], mode: 'docked', minimized: false }
			}));
		},

		// Bring panel to front
		focus(panel: keyof Omit<PanelsState, 'focusedPanel' | 'zCounter'>) {
			update(state => {
				const newState = { ...state };
				newState.zCounter++;
				newState[panel] = { ...state[panel], zIndex: newState.zCounter };
				newState.focusedPanel = panel;
				return newState;
			});
		},

		// Update position (for drag)
		setPosition(panel: keyof Omit<PanelsState, 'focusedPanel' | 'zCounter'>, x: number, y: number) {
			update(state => ({
				...state,
				[panel]: { ...state[panel], x, y }
			}));
		},

		// Update size (for resize)
		setSize(panel: keyof Omit<PanelsState, 'focusedPanel' | 'zCounter'>, width: number, height: number) {
			update(state => ({
				...state,
				[panel]: { ...state[panel], width, height }
			}));
		},

		// Reset all panels to default
		reset() {
			set(initialState);
		},

		// Load a complete panel layout (for workspace switching)
		loadLayout(layout: PanelsState) {
			set(layout);
		}
	};
}

export const panels = createPanelsStore();

// Derived store for checking if any panel is fullscreen
export const hasFullscreenPanel = derived(panels, $panels =>
	$panels.memory3d.mode === 'fullscreen' ||
	$panels.chat.mode === 'fullscreen' ||
	$panels.artifacts.mode === 'fullscreen' ||
	$panels.analytics.mode === 'fullscreen' ||
	$panels.swarm.mode === 'fullscreen'
);

// Derived store for closed panels (for panel switcher UI)
export const closedPanels = derived(panels, $panels => {
	const closed: string[] = [];
	if ($panels.memory3d.mode === 'closed') closed.push('memory3d');
	if ($panels.chat.mode === 'closed') closed.push('chat');
	if ($panels.artifacts.mode === 'closed') closed.push('artifacts');
	if ($panels.analytics.mode === 'closed') closed.push('analytics');
	if ($panels.swarm.mode === 'closed') closed.push('swarm');
	return closed;
});
