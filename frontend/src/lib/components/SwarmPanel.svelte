<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { swarm, waveProgress, latestFailure, latestFileWrite, AGENT_ORDER } from '$lib/stores';
	import type { AgentName, AgentStatus } from '$lib/stores';

	// Agent display config
	const agentConfig: Record<AgentName, { label: string; icon: string; color: string }> = {
		config: { label: 'CONFIG', icon: '🔧', color: '#00ffff' },
		executor: { label: 'EXECUTOR', icon: '⚡', color: '#00ff41' },
		reviewer: { label: 'REVIEWER', icon: '🔍', color: '#ffff00' },
		quality_gate: { label: 'Q-GATE', icon: '🚦', color: '#ff00ff' },
	};

	// Status display
	const statusDisplay: Record<AgentStatus, { icon: string; class: string }> = {
		idle: { icon: '○', class: 'status-idle' },
		thinking: { icon: '●', class: 'status-thinking' },
		done: { icon: '✓', class: 'status-done' },
		failed: { icon: '✗', class: 'status-failed' },
	};

	// Format tokens
	function formatTokens(tokens: number): string {
		if (tokens < 1000) return tokens.toString();
		if (tokens < 1000000) return (tokens / 1000).toFixed(1) + 'K';
		return (tokens / 1000000).toFixed(2) + 'M';
	}

	// Format latency
	function formatLatency(ms: number): string {
		if (ms < 1000) return ms + 'ms';
		return (ms / 1000).toFixed(1) + 's';
	}

	// Selected agent for detail view
	let selectedAgent: AgentName | null = null;

	// Expanded reasoning
	let expandedReasoning = false;

	// Connect on mount
	onMount(() => {
		swarm.connect();
	});

	onDestroy(() => {
		swarm.disconnect();
	});

	// Reactive: auto-select current agent
	$: if ($swarm.currentAgent && !selectedAgent) {
		selectedAgent = $swarm.currentAgent;
	}

	// Reactive: show failure alert
	$: showFailureAlert = $latestFailure &&
		Date.now() - new Date($latestFailure.timestamp).getTime() < 10000;
</script>

<div class="swarm-panel">
	<!-- Header -->
	<header class="panel-header">
		<div class="header-left">
			<span class="panel-title">SWARM DASHBOARD</span>
			<span class="connection-status" class:connected={$swarm.connected}>
				{$swarm.connected ? 'LIVE' : 'OFFLINE'}
			</span>
		</div>
		<div class="header-right">
			{#if $swarm.active}
				<span class="wave-badge">WAVE {$swarm.currentWave}</span>
			{/if}
		</div>
	</header>

	<!-- Failure Alert Banner -->
	{#if showFailureAlert && $latestFailure}
		<div class="failure-alert">
			<span class="alert-icon">⚠</span>
			<div class="alert-content">
				<strong>QUALITY GATE FAILED</strong>
				<span class="alert-detail">
					{$latestFailure.failure_type} - {$latestFailure.agent_blamed} blamed
				</span>
			</div>
		</div>
	{/if}

	<!-- Agent Cards Row -->
	<div class="agents-row">
		{#each AGENT_ORDER as agent}
			{@const state = $swarm.agents[agent]}
			{@const config = agentConfig[agent]}
			{@const status = statusDisplay[state.status]}
			<button
				class="agent-card"
				class:active={$swarm.currentAgent === agent}
				class:selected={selectedAgent === agent}
				class:done={state.status === 'done'}
				class:failed={state.status === 'failed'}
				class:thinking={state.status === 'thinking'}
				style="--agent-color: {config.color}"
				on:click={() => selectedAgent = agent}
			>
				<div class="agent-header">
					<span class="agent-icon">{config.icon}</span>
					<span class="agent-name">{config.label}</span>
				</div>
				<div class="agent-status">
					<span class="status-icon {status.class}">{status.icon}</span>
					<span class="status-label">{state.status}</span>
				</div>
				{#if state.status === 'done' || state.status === 'failed'}
					<div class="agent-stats">
						<span class="stat">{formatTokens(state.tokens_in)} in</span>
						<span class="stat">{formatTokens(state.tokens_out)} out</span>
						<span class="stat">{formatLatency(state.latency_ms)}</span>
					</div>
				{/if}
			</button>
		{/each}
	</div>

	<!-- Progress Bar -->
	<div class="progress-section">
		<div class="progress-header">
			<span class="progress-label">WAVE PROGRESS</span>
			<span class="progress-value">{$waveProgress.completed}/{$waveProgress.total}</span>
		</div>
		<div class="progress-bar">
			<div
				class="progress-fill"
				style="width: {$waveProgress.percentage}%"
			></div>
			{#each AGENT_ORDER as agent, i}
				<div
					class="progress-marker"
					class:complete={$swarm.agents[agent].status === 'done'}
					class:active={$swarm.currentAgent === agent}
					class:failed={$swarm.agents[agent].status === 'failed'}
					style="left: {((i + 0.5) / AGENT_ORDER.length) * 100}%"
				></div>
			{/each}
		</div>
	</div>

	<!-- Detail Panel -->
	<div class="detail-section">
		<div class="detail-columns">
			<!-- Reasoning Panel -->
			<div class="reasoning-panel">
				<div class="panel-section-header">
					<span class="section-title">REASONING</span>
					{#if selectedAgent && $swarm.agents[selectedAgent].reasoning.length > 0}
						<button
							class="expand-btn"
							on:click={() => expandedReasoning = !expandedReasoning}
						>
							{expandedReasoning ? '−' : '+'}
						</button>
					{/if}
				</div>
				<div class="reasoning-content" class:expanded={expandedReasoning}>
					{#if selectedAgent}
						{@const reasoning = $swarm.agents[selectedAgent].reasoning}
						{#if reasoning.length > 0}
							{#each reasoning as step}
								<div class="reasoning-step">
									<span class="step-number">Step {step.step}:</span>
									<span class="step-content">{step.content}</span>
								</div>
							{/each}
						{:else if $swarm.agents[selectedAgent].status === 'thinking'}
							<div class="reasoning-thinking">
								<span class="thinking-dot"></span>
								<span class="thinking-dot"></span>
								<span class="thinking-dot"></span>
							</div>
						{:else}
							<p class="empty-state">No reasoning data</p>
						{/if}
					{:else}
						<p class="empty-state">Select an agent to view reasoning</p>
					{/if}
				</div>
			</div>

			<!-- Code Preview Panel -->
			<div class="code-panel">
				<div class="panel-section-header">
					<span class="section-title">CODE PREVIEW</span>
					{#if $latestFileWrite}
						<span class="file-badge">{$latestFileWrite.file_path}</span>
					{/if}
				</div>
				<div class="code-content">
					{#if $latestFileWrite}
						<pre><code>{$latestFileWrite.preview}</code></pre>
					{:else if $swarm.agents.executor.preview}
						<pre><code>{$swarm.agents.executor.preview}</code></pre>
					{:else}
						<p class="empty-state">No code output yet</p>
					{/if}
				</div>
			</div>
		</div>
	</div>

	<!-- Footer Stats -->
	<footer class="panel-footer">
		<div class="footer-stat">
			<span class="stat-label">Total In</span>
			<span class="stat-value">{formatTokens($swarm.totalTokensIn)}</span>
		</div>
		<div class="footer-stat">
			<span class="stat-label">Total Out</span>
			<span class="stat-value">{formatTokens($swarm.totalTokensOut)}</span>
		</div>
		<div class="footer-stat">
			<span class="stat-label">Waves</span>
			<span class="stat-value">{$swarm.waves.length}</span>
		</div>
		<div class="footer-stat">
			<span class="stat-label">Failures</span>
			<span class="stat-value failure">{$swarm.failures.length}</span>
		</div>
	</footer>
</div>

<style>
	.swarm-panel {
		display: flex;
		flex-direction: column;
		height: 100%;
		font-family: 'JetBrains Mono', 'Fira Code', monospace;
		font-size: 0.8rem;
		background: var(--bg-primary, #0a0a0a);
		border: 1px solid var(--border-dim, #222);
		border-radius: 8px;
		overflow: hidden;
	}

	.panel-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.75rem 1rem;
		background: var(--bg-secondary, #111);
		border-bottom: 1px solid var(--border-dim, #222);
	}

	.header-left {
		display: flex;
		align-items: center;
		gap: 0.75rem;
	}

	.panel-title {
		font-weight: 700;
		font-size: 0.9rem;
		color: var(--neon-cyan, #00ffff);
		letter-spacing: 0.1em;
	}

	.connection-status {
		padding: 0.2rem 0.5rem;
		background: rgba(255, 0, 102, 0.2);
		color: var(--neon-pink, #ff0066);
		border-radius: 4px;
		font-size: 0.65rem;
		font-weight: 600;
	}

	.connection-status.connected {
		background: rgba(0, 255, 65, 0.2);
		color: var(--neon-green, #00ff41);
		animation: pulse-glow 2s ease-in-out infinite;
	}

	@keyframes pulse-glow {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.6; }
	}

	.wave-badge {
		padding: 0.25rem 0.75rem;
		background: var(--neon-cyan, #00ffff);
		color: #000;
		border-radius: 4px;
		font-weight: 700;
		font-size: 0.75rem;
	}

	/* Failure Alert */
	.failure-alert {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.75rem 1rem;
		background: rgba(255, 0, 102, 0.15);
		border-bottom: 1px solid var(--neon-pink, #ff0066);
		animation: flash-alert 0.5s ease-out;
	}

	@keyframes flash-alert {
		0%, 50% { background: rgba(255, 0, 102, 0.4); }
		100% { background: rgba(255, 0, 102, 0.15); }
	}

	.alert-icon {
		font-size: 1.2rem;
		animation: blink 1s infinite;
	}

	@keyframes blink {
		0%, 50% { opacity: 1; }
		51%, 100% { opacity: 0.3; }
	}

	.alert-content {
		display: flex;
		flex-direction: column;
	}

	.alert-content strong {
		color: var(--neon-pink, #ff0066);
	}

	.alert-detail {
		font-size: 0.7rem;
		color: var(--text-muted, #888);
	}

	/* Agent Cards */
	.agents-row {
		display: flex;
		gap: 0.5rem;
		padding: 0.75rem;
		overflow-x: auto;
	}

	.agent-card {
		flex: 1;
		min-width: 100px;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		padding: 0.75rem;
		background: var(--bg-tertiary, #0d0d0d);
		border: 1px solid var(--border-dim, #222);
		border-radius: 6px;
		cursor: pointer;
		transition: all 0.2s;
		text-align: left;
	}

	.agent-card:hover {
		border-color: var(--agent-color);
	}

	.agent-card.selected {
		border-color: var(--agent-color);
		box-shadow: 0 0 10px color-mix(in srgb, var(--agent-color) 30%, transparent);
	}

	.agent-card.active {
		background: color-mix(in srgb, var(--agent-color) 10%, var(--bg-tertiary));
	}

	.agent-card.thinking {
		animation: thinking-pulse 1.5s ease-in-out infinite;
	}

	@keyframes thinking-pulse {
		0%, 100% {
			box-shadow: 0 0 5px color-mix(in srgb, var(--agent-color) 30%, transparent);
		}
		50% {
			box-shadow: 0 0 20px color-mix(in srgb, var(--agent-color) 50%, transparent);
		}
	}

	.agent-header {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.agent-icon {
		font-size: 1rem;
	}

	.agent-name {
		font-weight: 600;
		font-size: 0.7rem;
		color: var(--agent-color);
		letter-spacing: 0.05em;
	}

	.agent-status {
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}

	.status-icon {
		font-size: 0.9rem;
	}

	.status-icon.status-idle { color: var(--text-muted, #888); }
	.status-icon.status-thinking {
		color: var(--neon-cyan, #00ffff);
		animation: pulse-glow 1s infinite;
	}
	.status-icon.status-done { color: var(--neon-green, #00ff41); }
	.status-icon.status-failed { color: var(--neon-pink, #ff0066); }

	.status-label {
		font-size: 0.65rem;
		color: var(--text-muted, #888);
		text-transform: uppercase;
	}

	.agent-stats {
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem;
	}

	.stat {
		font-size: 0.6rem;
		color: var(--text-muted, #888);
		padding: 0.1rem 0.3rem;
		background: var(--bg-secondary, #111);
		border-radius: 3px;
	}

	/* Progress Section */
	.progress-section {
		padding: 0.5rem 1rem;
	}

	.progress-header {
		display: flex;
		justify-content: space-between;
		margin-bottom: 0.25rem;
	}

	.progress-label {
		font-size: 0.65rem;
		color: var(--text-muted, #888);
		letter-spacing: 0.05em;
	}

	.progress-value {
		font-size: 0.7rem;
		font-weight: 600;
		color: var(--neon-cyan, #00ffff);
	}

	.progress-bar {
		position: relative;
		height: 6px;
		background: var(--bg-tertiary, #0d0d0d);
		border-radius: 3px;
	}

	.progress-fill {
		height: 100%;
		background: linear-gradient(90deg, var(--neon-cyan, #00ffff), var(--neon-green, #00ff41));
		border-radius: 3px;
		transition: width 0.3s ease;
	}

	.progress-marker {
		position: absolute;
		top: -3px;
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: var(--bg-tertiary, #0d0d0d);
		border: 2px solid var(--border-dim, #222);
		transform: translateX(-50%);
		transition: all 0.2s;
	}

	.progress-marker.complete {
		background: var(--neon-green, #00ff41);
		border-color: var(--neon-green, #00ff41);
	}

	.progress-marker.active {
		background: var(--neon-cyan, #00ffff);
		border-color: var(--neon-cyan, #00ffff);
		animation: marker-pulse 1s infinite;
	}

	.progress-marker.failed {
		background: var(--neon-pink, #ff0066);
		border-color: var(--neon-pink, #ff0066);
	}

	@keyframes marker-pulse {
		0%, 100% { transform: translateX(-50%) scale(1); }
		50% { transform: translateX(-50%) scale(1.3); }
	}

	/* Detail Section */
	.detail-section {
		flex: 1;
		min-height: 0;
		padding: 0.5rem;
		overflow: hidden;
	}

	.detail-columns {
		display: flex;
		gap: 0.5rem;
		height: 100%;
	}

	.reasoning-panel,
	.code-panel {
		flex: 1;
		display: flex;
		flex-direction: column;
		background: var(--bg-tertiary, #0d0d0d);
		border: 1px solid var(--border-dim, #222);
		border-radius: 6px;
		overflow: hidden;
	}

	.panel-section-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.5rem 0.75rem;
		background: var(--bg-secondary, #111);
		border-bottom: 1px solid var(--border-dim, #222);
	}

	.section-title {
		font-size: 0.65rem;
		font-weight: 600;
		color: var(--text-muted, #888);
		letter-spacing: 0.1em;
	}

	.expand-btn {
		width: 20px;
		height: 20px;
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--bg-tertiary, #0d0d0d);
		border: 1px solid var(--border-dim, #222);
		border-radius: 4px;
		color: var(--text-muted, #888);
		cursor: pointer;
		font-size: 0.9rem;
	}

	.file-badge {
		font-size: 0.6rem;
		padding: 0.1rem 0.4rem;
		background: var(--neon-green, #00ff41);
		color: #000;
		border-radius: 3px;
	}

	.reasoning-content,
	.code-content {
		flex: 1;
		padding: 0.5rem 0.75rem;
		overflow-y: auto;
	}

	.reasoning-content.expanded {
		max-height: none;
	}

	.reasoning-step {
		margin-bottom: 0.5rem;
		padding: 0.4rem;
		background: var(--bg-secondary, #111);
		border-radius: 4px;
	}

	.step-number {
		font-size: 0.65rem;
		color: var(--neon-cyan, #00ffff);
		font-weight: 600;
	}

	.step-content {
		display: block;
		margin-top: 0.25rem;
		font-size: 0.7rem;
		color: var(--text-primary, #fff);
		line-height: 1.4;
	}

	.reasoning-thinking {
		display: flex;
		gap: 0.5rem;
		justify-content: center;
		padding: 1rem;
	}

	.thinking-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--neon-cyan, #00ffff);
		animation: thinking-dots 1.4s infinite ease-in-out;
	}

	.thinking-dot:nth-child(1) { animation-delay: -0.32s; }
	.thinking-dot:nth-child(2) { animation-delay: -0.16s; }

	@keyframes thinking-dots {
		0%, 80%, 100% { transform: scale(0); }
		40% { transform: scale(1); }
	}

	.code-content pre {
		margin: 0;
		font-size: 0.7rem;
		line-height: 1.4;
		white-space: pre-wrap;
		word-break: break-all;
	}

	.code-content code {
		color: var(--neon-green, #00ff41);
	}

	.empty-state {
		color: var(--text-muted, #888);
		font-size: 0.75rem;
		text-align: center;
		padding: 1rem;
	}

	/* Footer */
	.panel-footer {
		display: flex;
		justify-content: space-around;
		padding: 0.5rem 1rem;
		background: var(--bg-secondary, #111);
		border-top: 1px solid var(--border-dim, #222);
	}

	.footer-stat {
		display: flex;
		flex-direction: column;
		align-items: center;
	}

	.stat-label {
		font-size: 0.6rem;
		color: var(--text-muted, #888);
		text-transform: uppercase;
	}

	.stat-value {
		font-size: 0.9rem;
		font-weight: 700;
		color: var(--neon-cyan, #00ffff);
	}

	.stat-value.failure {
		color: var(--neon-pink, #ff0066);
	}
</style>
