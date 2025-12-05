<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { panels, type PanelMode } from '$lib/stores/panels';

	export let panelId: 'memory3d' | 'chat' | 'artifacts' | 'analytics' | 'swarm';
	export let title: string;
	export let icon: string = '◻';

	const dispatch = createEventDispatcher();

	// Get panel state reactively
	$: panelState = $panels[panelId];
	$: mode = panelState.mode;
	$: minimized = panelState.minimized;
	$: zIndex = panelState.zIndex;

	// Drag state
	let isDragging = false;
	let dragStart = { x: 0, y: 0 };
	let panelStart = { x: 0, y: 0 };

	// Resize state
	let isResizing = false;
	let resizeStart = { x: 0, y: 0 };
	let sizeStart = { width: 0, height: 0 };

	function handleMouseDown(e: MouseEvent) {
		if (mode !== 'floating') return;
		panels.focus(panelId);
	}

	function startDrag(e: MouseEvent) {
		if (mode !== 'floating' || e.button !== 0) return;
		e.preventDefault();
		isDragging = true;
		dragStart = { x: e.clientX, y: e.clientY };
		panelStart = { x: panelState.x, y: panelState.y };
		panels.focus(panelId);

		window.addEventListener('mousemove', onDrag);
		window.addEventListener('mouseup', stopDrag);
	}

	function onDrag(e: MouseEvent) {
		if (!isDragging) return;
		const dx = e.clientX - dragStart.x;
		const dy = e.clientY - dragStart.y;
		panels.setPosition(panelId, panelStart.x + dx, panelStart.y + dy);
	}

	function stopDrag() {
		isDragging = false;
		window.removeEventListener('mousemove', onDrag);
		window.removeEventListener('mouseup', stopDrag);
	}

	function startResize(e: MouseEvent) {
		if (mode !== 'floating' || e.button !== 0) return;
		e.preventDefault();
		e.stopPropagation();
		isResizing = true;
		resizeStart = { x: e.clientX, y: e.clientY };
		sizeStart = { width: panelState.width, height: panelState.height };

		window.addEventListener('mousemove', onResize);
		window.addEventListener('mouseup', stopResize);
	}

	function onResize(e: MouseEvent) {
		if (!isResizing) return;
		const dx = e.clientX - resizeStart.x;
		const dy = e.clientY - resizeStart.y;
		const newWidth = Math.max(200, sizeStart.width + dx);
		const newHeight = Math.max(100, sizeStart.height + dy);
		panels.setSize(panelId, newWidth, newHeight);
	}

	function stopResize() {
		isResizing = false;
		window.removeEventListener('mousemove', onResize);
		window.removeEventListener('mouseup', stopResize);
	}

	function toggleFloat() {
		panels.toggleFloat(panelId);
	}

	function toggleFullscreen() {
		panels.toggleFullscreen(panelId);
	}

	function toggleMinimize() {
		panels.toggleMinimize(panelId);
	}

	function closePanel() {
		panels.close(panelId);
	}

	// Compute styles based on mode
	$: style = computeStyle(mode, panelState, minimized);

	function computeStyle(mode: PanelMode, state: typeof panelState, min: boolean): string {
		if (mode === 'closed') return 'display: none;';
		if (mode === 'fullscreen') {
			return `
				position: fixed;
				top: 0;
				left: 0;
				width: 100vw;
				height: 100vh;
				z-index: ${state.zIndex + 1000};
			`;
		}
		if (mode === 'floating') {
			return `
				position: fixed;
				top: ${state.y}px;
				left: ${state.x}px;
				width: ${state.width}px;
				height: ${min ? 'auto' : state.height + 'px'};
				z-index: ${state.zIndex};
			`;
		}
		// docked - use flex layout (handled by parent)
		return '';
	}
</script>

{#if mode !== 'closed'}
	<div
		class="floating-panel mode-{mode}"
		class:minimized
		class:dragging={isDragging}
		class:resizing={isResizing}
		{style}
		on:mousedown={handleMouseDown}
	>
		<!-- Title Bar -->
		<header class="panel-header" on:mousedown={startDrag}>
			<span class="panel-icon">{icon}</span>
			<span class="panel-title">{title}</span>

			<div class="panel-controls">
				<!-- Float/Dock toggle -->
				<button
					class="ctrl-btn"
					on:click|stopPropagation={toggleFloat}
					title={mode === 'floating' ? 'Dock panel' : 'Float panel'}
				>
					{mode === 'floating' ? '⊟' : '⊞'}
				</button>

				<!-- Minimize -->
				<button
					class="ctrl-btn"
					on:click|stopPropagation={toggleMinimize}
					title={minimized ? 'Expand' : 'Minimize'}
				>
					{minimized ? '▼' : '▬'}
				</button>

				<!-- Fullscreen -->
				<button
					class="ctrl-btn"
					on:click|stopPropagation={toggleFullscreen}
					title={mode === 'fullscreen' ? 'Exit fullscreen' : 'Fullscreen'}
				>
					{mode === 'fullscreen' ? '⊙' : '⛶'}
				</button>

				<!-- Close -->
				<button
					class="ctrl-btn close-btn"
					on:click|stopPropagation={closePanel}
					title="Close panel"
				>
					×
				</button>
			</div>
		</header>

		<!-- Content -->
		{#if !minimized}
			<div class="panel-content">
				<slot />
			</div>
		{/if}

		<!-- Resize handle (floating mode only) -->
		{#if mode === 'floating' && !minimized}
			<div class="resize-handle" on:mousedown={startResize}></div>
		{/if}
	</div>
{/if}

<style>
	.floating-panel {
		display: flex;
		flex-direction: column;
		background: var(--bg-primary, #0a0a0a);
		border: 1px solid var(--border-color, #333);
		border-radius: 8px;
		overflow: hidden;
		box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
	}

	.floating-panel.mode-docked {
		position: relative;
		border-radius: 0;
		box-shadow: none;
	}

	.floating-panel.mode-floating {
		box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), 0 0 1px var(--neon-green, #00ff41);
	}

	.floating-panel.mode-fullscreen {
		border-radius: 0;
		border: none;
	}

	.floating-panel.dragging,
	.floating-panel.resizing {
		user-select: none;
		opacity: 0.9;
	}

	.panel-header {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 0.75rem;
		background: var(--bg-tertiary, #0d0d0d);
		border-bottom: 1px solid var(--border-color, #333);
		cursor: default;
		flex-shrink: 0;
	}

	.mode-floating .panel-header {
		cursor: grab;
	}

	.mode-floating .panel-header:active {
		cursor: grabbing;
	}

	.panel-icon {
		font-size: 1rem;
	}

	.panel-title {
		flex: 1;
		font-weight: 600;
		font-size: 0.875rem;
		color: var(--neon-green, #00ff41);
	}

	.panel-controls {
		display: flex;
		gap: 0.25rem;
	}

	.ctrl-btn {
		background: none;
		border: 1px solid transparent;
		color: var(--text-muted, #888);
		cursor: pointer;
		font-size: 0.875rem;
		width: 24px;
		height: 24px;
		display: flex;
		align-items: center;
		justify-content: center;
		border-radius: 4px;
		transition: all 0.15s;
	}

	.ctrl-btn:hover {
		color: var(--text-primary, #fff);
		background: var(--bg-secondary, #1a1a1a);
		border-color: var(--border-color, #333);
	}

	.close-btn:hover {
		color: #ff4444;
		border-color: #ff4444;
	}

	.panel-content {
		flex: 1;
		overflow: auto;
		min-height: 0;
	}

	.minimized .panel-content {
		display: none;
	}

	.resize-handle {
		position: absolute;
		bottom: 0;
		right: 0;
		width: 16px;
		height: 16px;
		cursor: se-resize;
		background: linear-gradient(
			135deg,
			transparent 50%,
			var(--border-color, #333) 50%,
			var(--border-color, #333) 60%,
			transparent 60%,
			transparent 70%,
			var(--border-color, #333) 70%,
			var(--border-color, #333) 80%,
			transparent 80%
		);
	}

	.resize-handle:hover {
		background: linear-gradient(
			135deg,
			transparent 50%,
			var(--neon-green, #00ff41) 50%,
			var(--neon-green, #00ff41) 60%,
			transparent 60%,
			transparent 70%,
			var(--neon-green, #00ff41) 70%,
			var(--neon-green, #00ff41) 80%,
			transparent 80%
		);
	}
</style>
