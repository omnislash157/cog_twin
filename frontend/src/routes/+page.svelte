<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { theme, toggleTheme } from '$lib/stores/theme';
	import { websocket } from '$lib/stores/websocket';
	import { session } from '$lib/stores/session';
	import { visibleArtifacts, panels, closedPanels, workspaces } from '$lib/stores';
	import ArtifactPane from '$lib/components/ArtifactPane.svelte';
	import AnalyticsDashboard from '$lib/components/AnalyticsDashboard.svelte';
	import SwarmPanel from '$lib/components/SwarmPanel.svelte';
	import FloatingPanel from '$lib/components/FloatingPanel.svelte';
	import WorkspaceNav from '$lib/components/WorkspaceNav.svelte';
	import MemoryCanvas from '$lib/threlte/MemoryCanvas.svelte';
	import { marked } from 'marked';

	// Configure marked for safe inline rendering
	import { showSwarm, showMemorySpace, showAnalytics, config } from '$lib/stores/config';

	// Configure marked for safe inline rendering
	marked.setOptions({
		breaks: true,
		gfm: true
	});

	// Local input state
	let inputValue = '';

	// Upload state
	let uploadInput: HTMLInputElement;
	let uploading = false;
	let uploadStatus: string | null = null;
	let uploadJobId: string | null = null;

	// Generate session ID on mount
	onMount(() => {
		const sessionId = crypto.randomUUID();
		session.init(sessionId);
	});

	onDestroy(() => {
		session.cleanup();
	});

	// Send message handler
	function sendMessage() {
		if (!inputValue.trim() || !$websocket.connected) return;

		// Send message with local input value
		session.sendMessage(inputValue.trim());
		inputValue = '';
	}

	// Upload handlers
	function triggerUpload() {
		uploadInput?.click();
	}

	async function handleFileSelect(event: Event) {
		const input = event.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;

		uploading = true;
		uploadStatus = 'Uploading...';

		try {
			const formData = new FormData();
			formData.append('file', file);
			formData.append('provider', 'auto');

			const response = await fetch('/upload', {
				method: 'POST',
				body: formData
			});

			if (!response.ok) {
				const error = await response.json();
				throw new Error(error.detail || 'Upload failed');
			}

			const result = await response.json();
			uploadJobId = result.job_id;
			uploadStatus = 'Processing...';

			// Poll for completion
			pollIngestStatus();

		} catch (error) {
			uploadStatus = `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
			setTimeout(() => {
				uploadStatus = null;
				uploading = false;
			}, 5000);
		}

		// Reset input
		input.value = '';
	}

	async function pollIngestStatus() {
		if (!uploadJobId) return;

		try {
			const response = await fetch(`/upload/status/${uploadJobId}`);
			const status = await response.json();

			if (status.status === 'complete') {
				uploadStatus = `Done! ${status.parsed} convos → ${status.nodes} nodes`;
				setTimeout(() => {
					uploadStatus = null;
					uploading = false;
					uploadJobId = null;
				}, 5000);
			} else if (status.status === 'error') {
				uploadStatus = `Error: ${status.error}`;
				setTimeout(() => {
					uploadStatus = null;
					uploading = false;
					uploadJobId = null;
				}, 5000);
			} else {
				// Still processing
				uploadStatus = `${status.status}... (${status.parsed} parsed)`;
				setTimeout(pollIngestStatus, 2000);
			}
		} catch {
			setTimeout(pollIngestStatus, 2000);
		}
	}

	// Panel label map for closed panels menu
	const panelLabels: Record<string, { label: string; icon: string }> = {
		memory3d: { label: '3D Space', icon: '🧠' },
		chat: { label: 'Chat', icon: '💬' },
		artifacts: { label: 'Artifacts', icon: '📦' },
		analytics: { label: 'Analytics', icon: '📊' },
		swarm: { label: 'Swarm', icon: '🐝' }
	};
</script>

<svelte:head>
	<title>CogTwin Dashboard</title>
</svelte:head>

<div class="dashboard">
	<!-- Header -->
	<header class="header">
		<div class="logo">
			<span class="glow-text font-mono text-xl">COG_TWIN</span>
			<span class="text-text-muted text-sm ml-2">v2.6.1</span>
		</div>

		<div class="header-controls">
			<!-- Closed panels menu -->
			{#if $closedPanels.length > 0}
				<div class="closed-panels-menu">
					{#each $closedPanels as panelId}
						<button
							class="btn restore-btn"
							on:click={() => panels.open(panelId)}
							title="Restore {panelLabels[panelId]?.label}"
						>
							{panelLabels[panelId]?.icon} {panelLabels[panelId]?.label}
						</button>
					{/each}
				</div>
			{/if}

			<!-- Upload Status -->
			{#if uploadStatus}
				<div class="upload-status">
					{#if uploading && !uploadStatus.startsWith('Error') && !uploadStatus.startsWith('Done')}
						<span class="spinner"></span>
					{/if}
					<span class="text-sm">{uploadStatus}</span>
				</div>
			{/if}

			<!-- Hidden file input -->
			<input
				bind:this={uploadInput}
				type="file"
				accept=".json,.zip"
				on:change={handleFileSelect}
				style="display: none;"
			/>

			<!-- Upload Button -->
			<button
				class="btn upload-btn"
				on:click={triggerUpload}
				disabled={uploading}
				title="Upload chat logs (Claude, ChatGPT, Grok)"
			>
				{#if uploading}
					⏳
				{:else}
					📤 Import
				{/if}
			</button>

			<div class="connection-status" class:connected={$websocket.connected}>
				<span class="status-dot"></span>
				<span class="text-sm">{$websocket.connected ? 'Connected' : 'Disconnected'}</span>
			</div>

			<button class="btn" on:click={toggleTheme} title="Toggle theme">
				{$theme === 'cyber' ? '👔' : '🌙'}
			</button>

			<!-- Reset layout button -->
			<button class="btn" on:click={() => panels.reset()} title="Reset panel layout">
				⟲
			</button>
		</div>
	</header>

	<!-- Workspace Navigation -->
	<WorkspaceNav />

	<!-- Main Content -->
	<main class="dashboard-main">
		<!-- 3D Memory Space Panel (only in full mode) -->
		{#if $showMemorySpace}
		<FloatingPanel panelId="memory3d" title="Memory Space" icon="🧠">
			<div class="memory-space-content">
				<MemoryCanvas
					memories={$visibleArtifacts.map(a => ({
						id: a.id,
						content_preview: a.artifact.title || a.artifact.type,
						source: 'vector',
						relevance: 0.7,
						readonly: false
					}))}
					on:selectMemory={(e) => console.log('Selected:', e.detail.id)}
					on:synthesize={(e) => console.log('Synthesize:', e.detail)}
				/>
			</div>
		</FloatingPanel>
		{/if}

		<!-- Right side container for docked panels -->
		<div class="right-panels" class:has-docked={$panels.chat.mode === 'docked' || $panels.artifacts.mode === 'docked' || $panels.analytics.mode === 'docked' || $panels.swarm?.mode === 'docked'}>
			<!-- Chat Panel -->
			<FloatingPanel panelId="chat" title="Chat" icon="💬">
				<div class="chat-section">
					<div class="messages-container">
						{#each $session.messages as message}
							<div class="message {message.role}">
								<div class="message-content">{@html marked.parse(message.content)}</div>
							</div>
						{/each}
						{#if $session.currentStream}
							<div class="message assistant streaming">
								<div class="message-content">{@html marked.parse($session.currentStream)}</div>
								<span class="cursor">▊</span>
							</div>
						{/if}
					</div>
					<form class="chat-input" on:submit|preventDefault={sendMessage}>
						<input
							type="text"
							bind:value={inputValue}
							placeholder="Ask anything..."
							disabled={!$websocket.connected}
							on:keydown={(e) => e.key === 'Enter' && sendMessage()}
						/>
						<button type="submit" disabled={!$websocket.connected || !inputValue.trim()}>
							Send
						</button>
					</form>
				</div>
			</FloatingPanel>

			<!-- Artifacts Panel -->
			<FloatingPanel panelId="artifacts" title="Artifacts ({$visibleArtifacts.length})" icon="📦">
				<div class="artifacts-section">
					<div class="artifact-list">
						{#each $visibleArtifacts as item (item.id)}
							<ArtifactPane {item} />
						{/each}
						{#if $visibleArtifacts.length === 0}
							<p class="empty">No artifacts yet.</p>
						{/if}
					</div>
				</div>
			</FloatingPanel>

			<!-- Analytics Panel (only if enabled) -->
			{#if $showAnalytics}
			<FloatingPanel panelId="analytics" title="Cognitive State" icon="📊">
				<AnalyticsDashboard />
			</FloatingPanel>
			{/if}

			<!-- Swarm Dashboard Panel (only if enabled) -->
			{#if $showSwarm}
			<FloatingPanel panelId="swarm" title="Swarm Dashboard" icon="🐝">
				<SwarmPanel />
			</FloatingPanel>
			{/if}
		</div>
	</main>

	<!-- Bottom: Status Bar -->
	<footer class="status-bar">
		<div class="status-item">
			<span class="text-text-muted">Phase:</span>
			<span class="text-neon-green font-mono">{$session.cognitiveState.phase}</span>
		</div>
		<div class="status-item">
			<span class="text-text-muted">Temp:</span>
			<span class="text-neon-yellow font-mono">{$session.cognitiveState.temperature.toFixed(2)}</span>
		</div>
		<div class="status-item">
			<span class="text-text-muted">Session:</span>
			<span class="font-mono">{$websocket.sessionId || 'none'}</span>
		</div>
	</footer>
</div>

<style>
	.dashboard {
		display: flex;
		flex-direction: column;
		height: 100vh;
		padding: 1rem;
		gap: 1rem;
		overflow: hidden;
	}

	.header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.5rem 1rem;
		background: var(--bg-secondary);
		border: 1px solid var(--border-dim);
		border-radius: 8px;
		flex-shrink: 0;
	}

	.logo {
		display: flex;
		align-items: baseline;
	}

	.header-controls {
		display: flex;
		align-items: center;
		gap: 1rem;
	}

	.closed-panels-menu {
		display: flex;
		gap: 0.5rem;
	}

	.restore-btn {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		padding: 0.25rem 0.5rem;
		background: var(--bg-tertiary);
		border: 1px solid var(--neon-cyan, #00ffff);
		color: var(--neon-cyan, #00ffff);
		border-radius: 4px;
		cursor: pointer;
		font-size: 0.75rem;
		transition: all 0.2s;
	}

	.restore-btn:hover {
		background: var(--neon-cyan, #00ffff);
		color: black;
	}

	.upload-btn {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		padding: 0.375rem 0.75rem;
		background: var(--bg-tertiary);
		border: 1px solid var(--neon-green);
		color: var(--neon-green);
		border-radius: 4px;
		cursor: pointer;
		transition: all 0.2s;
	}

	.upload-btn:hover:not(:disabled) {
		background: var(--neon-green);
		color: black;
	}

	.upload-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.upload-status {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.25rem 0.75rem;
		background: var(--bg-tertiary);
		border: 1px solid var(--border-dim);
		border-radius: 4px;
		color: var(--neon-yellow);
		font-size: 0.75rem;
	}

	.spinner {
		width: 12px;
		height: 12px;
		border: 2px solid var(--neon-green);
		border-top-color: transparent;
		border-radius: 50%;
		animation: spin 1s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	.connection-status {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		color: var(--text-muted);
	}

	.connection-status.connected {
		color: var(--neon-green);
	}

	.status-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--text-muted);
	}

	.connection-status.connected .status-dot {
		background: var(--neon-green);
		box-shadow: 0 0 10px var(--neon-green);
	}

	.dashboard-main {
		display: flex;
		flex: 1;
		gap: 1rem;
		overflow: hidden;
		min-height: 0;
	}

	/* Memory space takes remaining width when docked */
	.dashboard-main > :global(.floating-panel.mode-docked:first-child) {
		flex: 1;
		min-width: 0;
	}

	.right-panels {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		width: 400px;
		flex-shrink: 0;
		min-height: 0;
	}

	.right-panels:not(.has-docked) {
		display: none;
	}

	.right-panels > :global(.floating-panel.mode-docked) {
		flex: 1;
		min-height: 0;
	}

	/* Hide docked panels container content when panels are floating/closed */
	.right-panels > :global(.floating-panel.mode-floating),
	.right-panels > :global(.floating-panel.mode-fullscreen),
	.right-panels > :global(.floating-panel.mode-closed) {
		position: fixed;
	}

	.memory-space-content {
		width: 100%;
		height: 100%;
		min-height: 300px;
	}

	.chat-section {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 200px;
	}

	.messages-container {
		flex: 1;
		overflow-y: auto;
		padding: 0.75rem;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.artifacts-section {
		height: 100%;
		min-height: 150px;
	}

	.artifact-list {
		height: 100%;
		overflow-y: auto;
		padding: 0.75rem;
	}

	.empty {
		color: var(--text-muted, #888);
		text-align: center;
		padding: 2rem 1rem;
		font-size: 0.875rem;
	}

	.message {
		padding: 0.75rem 1rem;
		border-radius: 8px;
		max-width: 85%;
	}

	.message.user {
		background: var(--bg-tertiary);
		align-self: flex-end;
		border: 1px solid var(--border-dim);
	}

	.message.assistant {
		background: var(--bg-secondary);
		align-self: flex-start;
		border: 1px solid var(--border-glow);
	}

	.message.streaming .cursor {
		animation: blink 1s infinite;
		color: var(--neon-green);
	}

	@keyframes blink {
		0%, 50% { opacity: 1; }
		51%, 100% { opacity: 0; }
	}

	.chat-input {
		display: flex;
		gap: 0.5rem;
		padding: 0.75rem;
		border-top: 1px solid var(--border-dim);
		flex-shrink: 0;
	}

	.chat-input input {
		flex: 1;
	}

	.status-bar {
		display: flex;
		gap: 2rem;
		padding: 0.5rem 1rem;
		background: var(--bg-secondary);
		border: 1px solid var(--border-dim);
		border-radius: 8px;
		flex-shrink: 0;
	}

	.status-item {
		display: flex;
		gap: 0.5rem;
		font-size: 0.875rem;
	}
</style>
