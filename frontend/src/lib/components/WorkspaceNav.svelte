<script lang="ts">
	import { onMount } from 'svelte';
	import { workspaces, activeWorkspace } from '$lib/stores';

	// Swipe state
	let navContainer: HTMLElement;
	let isDragging = false;
	let startX = 0;
	let currentX = 0;
	const SWIPE_THRESHOLD = 50;

	// Context menu state
	let contextMenu: { x: number; y: number; workspaceId: string } | null = null;
	let editingId: string | null = null;
	let editValue = '';

	// Initialize workspaces on mount
	onMount(() => {
		workspaces.init();

		// Close context menu on outside click
		const handleClickOutside = (e: MouseEvent) => {
			if (contextMenu && !(e.target as HTMLElement).closest('.context-menu')) {
				contextMenu = null;
			}
		};
		document.addEventListener('click', handleClickOutside);
		return () => document.removeEventListener('click', handleClickOutside);
	});

	// Swipe handlers
	function handlePointerDown(e: PointerEvent) {
		if ((e.target as HTMLElement).closest('.workspace-tab, .add-btn, .context-menu')) return;
		isDragging = true;
		startX = e.clientX;
		currentX = e.clientX;
		navContainer?.setPointerCapture(e.pointerId);
	}

	function handlePointerMove(e: PointerEvent) {
		if (!isDragging) return;
		currentX = e.clientX;
	}

	function handlePointerUp(e: PointerEvent) {
		if (!isDragging) return;
		isDragging = false;
		navContainer?.releasePointerCapture(e.pointerId);

		const delta = currentX - startX;
		if (Math.abs(delta) >= SWIPE_THRESHOLD) {
			if (delta > 0) {
				workspaces.prevWorkspace();
			} else {
				workspaces.nextWorkspace();
			}
		}
	}

	// Tab click
	function selectWorkspace(index: number) {
		workspaces.switchWorkspace(index);
	}

	// Create new workspace
	function createWorkspace() {
		const name = `Workspace ${$workspaces.workspaces.length + 1}`;
		workspaces.createWorkspace(name, '📁');
		// Switch to the new workspace
		workspaces.switchWorkspace($workspaces.workspaces.length);
	}

	// Context menu handlers
	function handleContextMenu(e: MouseEvent, workspaceId: string) {
		e.preventDefault();
		contextMenu = { x: e.clientX, y: e.clientY, workspaceId };
	}

	function startRename(id: string) {
		const ws = $workspaces.workspaces.find(w => w.id === id);
		if (ws) {
			editingId = id;
			editValue = ws.label;
		}
		contextMenu = null;
	}

	function finishRename() {
		if (editingId && editValue.trim()) {
			workspaces.renameWorkspace(editingId, editValue.trim());
		}
		editingId = null;
		editValue = '';
	}

	function handleRenameKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			finishRename();
		} else if (e.key === 'Escape') {
			editingId = null;
			editValue = '';
		}
	}

	function deleteWorkspace(id: string) {
		workspaces.deleteWorkspace(id);
		contextMenu = null;
	}

	// Long press for touch devices
	let longPressTimer: ReturnType<typeof setTimeout> | null = null;

	function handleTouchStart(e: TouchEvent, workspaceId: string) {
		longPressTimer = setTimeout(() => {
			const touch = e.touches[0];
			contextMenu = { x: touch.clientX, y: touch.clientY, workspaceId };
		}, 500);
	}

	function handleTouchEnd() {
		if (longPressTimer) {
			clearTimeout(longPressTimer);
			longPressTimer = null;
		}
	}
</script>

<nav
	class="workspace-nav"
	bind:this={navContainer}
	on:pointerdown={handlePointerDown}
	on:pointermove={handlePointerMove}
	on:pointerup={handlePointerUp}
	on:pointercancel={handlePointerUp}
>
	<div class="workspace-tabs">
		{#each $workspaces.workspaces as workspace, index (workspace.id)}
			<button
				class="workspace-tab"
				class:active={index === $workspaces.activeIndex}
				on:click={() => selectWorkspace(index)}
				on:contextmenu={(e) => handleContextMenu(e, workspace.id)}
				on:touchstart={(e) => handleTouchStart(e, workspace.id)}
				on:touchend={handleTouchEnd}
				on:touchcancel={handleTouchEnd}
			>
				{#if editingId === workspace.id}
					<input
						type="text"
						class="rename-input"
						bind:value={editValue}
						on:blur={finishRename}
						on:keydown={handleRenameKeydown}
						autofocus
					/>
				{:else}
					<span class="workspace-icon">{workspace.icon}</span>
					<span class="workspace-label">{workspace.label}</span>
				{/if}
				<span class="active-indicator" class:visible={index === $workspaces.activeIndex}></span>
			</button>
		{/each}

		<button class="add-btn" on:click={createWorkspace} title="New workspace">
			+
		</button>
	</div>

	<!-- Dot indicators for swipe -->
	<div class="dot-indicators">
		{#each $workspaces.workspaces as _, index}
			<span
				class="dot"
				class:active={index === $workspaces.activeIndex}
				on:click={() => selectWorkspace(index)}
				on:keydown={(e) => e.key === 'Enter' && selectWorkspace(index)}
				role="button"
				tabindex="0"
			></span>
		{/each}
	</div>
</nav>

<!-- Context Menu -->
{#if contextMenu}
	{@const menuData = contextMenu}
	<div
		class="context-menu"
		style="left: {menuData.x}px; top: {menuData.y}px;"
	>
		<button on:click={() => startRename(menuData.workspaceId)}>
			Rename
		</button>
		{#if $workspaces.workspaces.length > 1}
			<button class="danger" on:click={() => deleteWorkspace(menuData.workspaceId)}>
				Delete
			</button>
		{/if}
	</div>
{/if}

<style>
	.workspace-nav {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 1rem;
		background: var(--bg-secondary);
		border: 1px solid var(--border-dim);
		border-radius: 8px;
		user-select: none;
		touch-action: pan-y;
	}

	.workspace-tabs {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		overflow-x: auto;
		max-width: 100%;
		scrollbar-width: none;
	}

	.workspace-tabs::-webkit-scrollbar {
		display: none;
	}

	.workspace-tab {
		position: relative;
		display: flex;
		align-items: center;
		gap: 0.375rem;
		padding: 0.5rem 0.75rem;
		background: var(--bg-tertiary);
		border: 1px solid var(--border-dim);
		border-radius: 6px;
		color: var(--text-muted);
		cursor: pointer;
		transition: all 0.2s ease;
		white-space: nowrap;
		font-size: 0.875rem;
	}

	.workspace-tab:hover {
		border-color: var(--border-glow);
		color: var(--text-primary);
	}

	.workspace-tab.active {
		background: var(--bg-primary);
		border-color: var(--neon-cyan, #00ffff);
		color: var(--neon-cyan, #00ffff);
	}

	.workspace-icon {
		font-size: 1rem;
	}

	.workspace-label {
		max-width: 120px;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.active-indicator {
		position: absolute;
		bottom: -1px;
		left: 50%;
		transform: translateX(-50%);
		width: 0;
		height: 2px;
		background: var(--neon-cyan, #00ffff);
		transition: width 0.2s ease;
		border-radius: 1px;
	}

	.active-indicator.visible {
		width: 60%;
		box-shadow: 0 0 8px var(--neon-cyan, #00ffff);
	}

	.rename-input {
		width: 100px;
		padding: 0.125rem 0.25rem;
		background: var(--bg-primary);
		border: 1px solid var(--neon-cyan, #00ffff);
		border-radius: 3px;
		color: var(--text-primary);
		font-size: 0.875rem;
		outline: none;
	}

	.add-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 32px;
		height: 32px;
		background: var(--bg-tertiary);
		border: 1px dashed var(--border-dim);
		border-radius: 6px;
		color: var(--text-muted);
		cursor: pointer;
		font-size: 1.25rem;
		transition: all 0.2s ease;
	}

	.add-btn:hover {
		border-color: var(--neon-green);
		color: var(--neon-green);
		border-style: solid;
	}

	.dot-indicators {
		display: flex;
		gap: 0.5rem;
	}

	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--text-muted);
		opacity: 0.4;
		cursor: pointer;
		transition: all 0.2s ease;
	}

	.dot:hover {
		opacity: 0.7;
	}

	.dot.active {
		background: var(--neon-cyan, #00ffff);
		opacity: 1;
		box-shadow: 0 0 8px var(--neon-cyan, #00ffff);
	}

	.context-menu {
		position: fixed;
		z-index: 10000;
		background: var(--bg-secondary);
		border: 1px solid var(--border-glow);
		border-radius: 6px;
		padding: 0.25rem;
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
	}

	.context-menu button {
		display: block;
		width: 100%;
		padding: 0.5rem 1rem;
		background: none;
		border: none;
		color: var(--text-primary);
		text-align: left;
		cursor: pointer;
		border-radius: 4px;
		font-size: 0.875rem;
	}

	.context-menu button:hover {
		background: var(--bg-tertiary);
	}

	.context-menu button.danger {
		color: var(--neon-red, #ff4444);
	}

	.context-menu button.danger:hover {
		background: rgba(255, 68, 68, 0.1);
	}
</style>
