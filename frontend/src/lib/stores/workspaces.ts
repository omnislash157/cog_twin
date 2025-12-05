import { writable, derived, get } from 'svelte/store';
import { panels, type PanelsState } from './panels';

export interface Workspace {
	id: string;
	label: string;
	icon: string;
	panelLayout: PanelsState;
}

export interface WorkspacesState {
	workspaces: Workspace[];
	activeIndex: number;
}

const STORAGE_KEY = 'cogtwin_workspaces';

// Get current panel state snapshot
function getPanelSnapshot(): PanelsState {
	return get(panels);
}

// Default panel layout for new workspaces
function getDefaultLayout(): PanelsState {
	return {
		memory3d: {
			mode: 'docked',
			x: 0,
			y: 0,
			width: 60,
			height: 100,
			zIndex: 100,
			minimized: false
		},
		chat: {
			mode: 'docked',
			x: 100,
			y: 50,
			width: 400,
			height: 500,
			zIndex: 100,
			minimized: false
		},
		artifacts: {
			mode: 'docked',
			x: 150,
			y: 100,
			width: 400,
			height: 400,
			zIndex: 100,
			minimized: false
		},
		analytics: {
			mode: 'docked',
			x: 200,
			y: 150,
			width: 380,
			height: 450,
			zIndex: 100,
			minimized: false
		},
		swarm: {
			mode: 'docked',
			x: 250,
			y: 50,
			width: 600,
			height: 500,
			zIndex: 100,
			minimized: false
		},
		focusedPanel: null,
		zCounter: 100
	};
}

// Load from localStorage
function loadFromStorage(): WorkspacesState | null {
	if (typeof window === 'undefined') return null;
	try {
		const stored = localStorage.getItem(STORAGE_KEY);
		if (stored) {
			return JSON.parse(stored);
		}
	} catch (e) {
		console.warn('Failed to load workspaces from storage:', e);
	}
	return null;
}

// Save to localStorage
function saveToStorage(state: WorkspacesState) {
	if (typeof window === 'undefined') return;
	try {
		localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
	} catch (e) {
		console.warn('Failed to save workspaces to storage:', e);
	}
}

// Create default initial state
function createInitialState(): WorkspacesState {
	const stored = loadFromStorage();
	if (stored && stored.workspaces.length > 0) {
		return stored;
	}

	return {
		workspaces: [
			{
				id: crypto.randomUUID(),
				label: 'Memory Lab',
				icon: '🧠',
				panelLayout: getDefaultLayout()
			}
		],
		activeIndex: 0
	};
}

function createWorkspacesStore() {
	const { subscribe, set, update } = writable<WorkspacesState>(createInitialState());

	// Track if we've initialized panel sync
	let initialized = false;

	return {
		subscribe,

		// Initialize workspace system (call once on mount)
		init() {
			if (initialized) return;
			initialized = true;

			// Load active workspace's panel layout
			const state = get({ subscribe });
			if (state.workspaces.length > 0) {
				const activeWorkspace = state.workspaces[state.activeIndex];
				if (activeWorkspace?.panelLayout) {
					panels.loadLayout(activeWorkspace.panelLayout);
				}
			}
		},

		// Create a new workspace
		createWorkspace(label: string, icon: string = '📁') {
			update(state => {
				const newWorkspace: Workspace = {
					id: crypto.randomUUID(),
					label,
					icon,
					panelLayout: getDefaultLayout()
				};

				const newState = {
					...state,
					workspaces: [...state.workspaces, newWorkspace]
				};

				saveToStorage(newState);
				return newState;
			});
		},

		// Delete a workspace (prevent deleting last one)
		deleteWorkspace(id: string) {
			update(state => {
				if (state.workspaces.length <= 1) {
					console.warn('Cannot delete the last workspace');
					return state;
				}

				const index = state.workspaces.findIndex(w => w.id === id);
				if (index === -1) return state;

				const newWorkspaces = state.workspaces.filter(w => w.id !== id);
				let newActiveIndex = state.activeIndex;

				// Adjust active index if needed
				if (index <= state.activeIndex) {
					newActiveIndex = Math.max(0, state.activeIndex - 1);
				}

				const newState = {
					workspaces: newWorkspaces,
					activeIndex: newActiveIndex
				};

				// Load the new active workspace's layout
				if (newWorkspaces[newActiveIndex]) {
					panels.loadLayout(newWorkspaces[newActiveIndex].panelLayout);
				}

				saveToStorage(newState);
				return newState;
			});
		},

		// Switch to a different workspace
		switchWorkspace(index: number) {
			update(state => {
				if (index < 0 || index >= state.workspaces.length) return state;
				if (index === state.activeIndex) return state;

				// Save current panel layout to current workspace
				const currentPanelState = getPanelSnapshot();
				const updatedWorkspaces = [...state.workspaces];
				updatedWorkspaces[state.activeIndex] = {
					...updatedWorkspaces[state.activeIndex],
					panelLayout: currentPanelState
				};

				// Load new workspace's panel layout
				const targetWorkspace = updatedWorkspaces[index];
				if (targetWorkspace?.panelLayout) {
					panels.loadLayout(targetWorkspace.panelLayout);
				}

				const newState = {
					workspaces: updatedWorkspaces,
					activeIndex: index
				};

				saveToStorage(newState);
				return newState;
			});
		},

		// Rename a workspace
		renameWorkspace(id: string, label: string) {
			update(state => {
				const newWorkspaces = state.workspaces.map(w =>
					w.id === id ? { ...w, label } : w
				);

				const newState = {
					...state,
					workspaces: newWorkspaces
				};

				saveToStorage(newState);
				return newState;
			});
		},

		// Update workspace icon
		setIcon(id: string, icon: string) {
			update(state => {
				const newWorkspaces = state.workspaces.map(w =>
					w.id === id ? { ...w, icon } : w
				);

				const newState = {
					...state,
					workspaces: newWorkspaces
				};

				saveToStorage(newState);
				return newState;
			});
		},

		// Save current panel state to active workspace
		saveCurrentLayout() {
			update(state => {
				const currentPanelState = getPanelSnapshot();
				const updatedWorkspaces = [...state.workspaces];
				updatedWorkspaces[state.activeIndex] = {
					...updatedWorkspaces[state.activeIndex],
					panelLayout: currentPanelState
				};

				const newState = {
					...state,
					workspaces: updatedWorkspaces
				};

				saveToStorage(newState);
				return newState;
			});
		},

		// Navigate to next workspace (for swipe right)
		nextWorkspace() {
			const state = get({ subscribe });
			if (state.activeIndex < state.workspaces.length - 1) {
				this.switchWorkspace(state.activeIndex + 1);
			}
		},

		// Navigate to previous workspace (for swipe left)
		prevWorkspace() {
			const state = get({ subscribe });
			if (state.activeIndex > 0) {
				this.switchWorkspace(state.activeIndex - 1);
			}
		},

		// Reset to defaults
		reset() {
			const defaultState: WorkspacesState = {
				workspaces: [
					{
						id: crypto.randomUUID(),
						label: 'Memory Lab',
						icon: '🧠',
						panelLayout: getDefaultLayout()
					}
				],
				activeIndex: 0
			};

			panels.loadLayout(defaultState.workspaces[0].panelLayout);
			saveToStorage(defaultState);
			set(defaultState);
		}
	};
}

export const workspaces = createWorkspacesStore();

// Derived store for active workspace
export const activeWorkspace = derived(workspaces, $workspaces =>
	$workspaces.workspaces[$workspaces.activeIndex] || null
);

// Derived store for workspace count
export const workspaceCount = derived(workspaces, $workspaces =>
	$workspaces.workspaces.length
);
