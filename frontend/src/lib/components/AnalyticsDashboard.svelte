<script lang="ts">
	import { session } from '$lib/stores/session';

	const phaseColors: Record<string, string> = {
		exploration: '#00ffff',
		exploitation: '#00ff00',
		consolidation: '#ffff00',
		crisis: '#ff0066',
		idle: '#888888',
		divergent: '#ff9900',
		convergent: '#00ff99',
	};

	function getPhaseColor(phase: string): string {
		return phaseColors[phase.toLowerCase()] || '#00ffff';
	}

	function formatDuration(minutes: number): string {
		if (minutes < 60) return Math.round(minutes) + 'm';
		const hrs = Math.floor(minutes / 60);
		const mins = Math.round(minutes % 60);
		return hrs + 'h ' + mins + 'm';
	}

	function formatTokens(tokens: number): string {
		if (tokens < 1000) return tokens.toString();
		if (tokens < 1000000) return (tokens / 1000).toFixed(1) + 'K';
		return (tokens / 1000000).toFixed(2) + 'M';
	}

	$: stabilityWarning = $session.analytics.stability < 0.4;
	$: hasSuggestion = $session.analytics.suggestion && $session.analytics.suggestion.trim().length > 0;
</script>

<div class="analytics-dashboard">
	<section class="phase-section">
		<div class="phase-indicator" style="--phase-color: {getPhaseColor($session.analytics.phase)}">
			<span class="phase-dot"></span>
			<span class="phase-name">{$session.analytics.phase.toUpperCase()}</span>
		</div>
		<p class="phase-description">{$session.analytics.phaseDescription}</p>
	</section>

	<section class="gauge-section">
		<div class="gauge-header">
			<span class="gauge-label">STABILITY</span>
			<span class="gauge-value" class:warning={stabilityWarning}>
				{($session.analytics.stability * 100).toFixed(0)}%
			</span>
		</div>
		<div class="gauge-bar">
			<div class="gauge-fill" class:warning={stabilityWarning}
				style="width: {$session.analytics.stability * 100}%">
			</div>
			<div class="gauge-threshold" style="left: 40%"></div>
		</div>
	</section>

	<section class="metrics-row">
		<div class="metric">
			<span class="metric-label">TEMP</span>
			<span class="metric-value temp">{$session.analytics.temperature.toFixed(2)}</span>
		</div>
		<div class="metric">
			<span class="metric-label">FOCUS</span>
			<span class="metric-value focus">{($session.analytics.focusScore * 100).toFixed(0)}%</span>
		</div>
		<div class="metric">
			<span class="metric-label">ACCURACY</span>
			<span class="metric-value accuracy">{($session.analytics.predictionAccuracy * 100).toFixed(0)}%</span>
		</div>
	</section>

	{#if $session.analytics.driftSignal}
		<section class="drift-section">
			<div class="drift-header">
				<span class="drift-icon">⚠</span>
				<span class="drift-label">DRIFT DETECTED</span>
			</div>
			<div class="drift-info">
				<span class="drift-signal">{$session.analytics.driftSignal}</span>
				<span class="drift-magnitude">
					{($session.analytics.driftMagnitude * 100).toFixed(0)}% magnitude
				</span>
			</div>
		</section>
	{/if}

	{#if $session.analytics.recurringPatterns.length > 0}
		<section class="patterns-section">
			<h4 class="section-title">RECURRING PATTERNS</h4>
			<ul class="patterns-list">
				{#each $session.analytics.recurringPatterns.slice(0, 5) as pattern}
					<li class="pattern-item">
						<span class="pattern-topic">{pattern.topic}</span>
						<span class="pattern-freq">{pattern.frequency}x</span>
					</li>
				{/each}
			</ul>
		</section>
	{/if}

	{#if hasSuggestion}
		<section class="suggestion-section" class:pulse={hasSuggestion}>
			<div class="suggestion-header">
				<span class="suggestion-icon">💡</span>
				<span class="suggestion-label">SUGGESTION</span>
			</div>
			<p class="suggestion-text">{$session.analytics.suggestion}</p>
		</section>
	{/if}

	<section class="stats-row">
		<div class="stat">
			<span class="stat-value">{$session.analytics.totalQueries}</span>
			<span class="stat-label">queries</span>
		</div>
		<div class="stat">
			<span class="stat-value">{formatTokens($session.analytics.totalTokens)}</span>
			<span class="stat-label">tokens</span>
		</div>
		<div class="stat">
			<span class="stat-value">{formatDuration($session.analytics.sessionDurationMinutes)}</span>
			<span class="stat-label">session</span>
		</div>
	</section>

	{#if $session.analytics.recentTransitions.length > 0}
		<section class="transitions-section">
			<h4 class="section-title">TRANSITIONS</h4>
			<div class="transitions-list">
				{#each $session.analytics.recentTransitions.slice(-4) as transition}
					<div class="transition-item">
						<span class="transition-from" style="color: {getPhaseColor(transition.from)}">
							{transition.from}
						</span>
						<span class="transition-arrow">→</span>
						<span class="transition-to" style="color: {getPhaseColor(transition.to)}">
							{transition.to}
						</span>
					</div>
				{/each}
			</div>
		</section>
	{/if}
</div>

<style>
	.analytics-dashboard {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		padding: 0.75rem;
		font-family: 'JetBrains Mono', 'Fira Code', monospace;
		font-size: 0.8rem;
		height: 100%;
		overflow-y: auto;
	}

	.phase-section {
		padding: 0.75rem;
		background: var(--bg-tertiary, #0d0d0d);
		border: 1px solid var(--border-dim, #222);
		border-radius: 6px;
	}

	.phase-indicator {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.phase-dot {
		width: 10px;
		height: 10px;
		border-radius: 50%;
		background: var(--phase-color);
		box-shadow: 0 0 8px var(--phase-color), 0 0 16px var(--phase-color);
		animation: pulse-glow 2s ease-in-out infinite;
	}

	@keyframes pulse-glow {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.6; }
	}

	.phase-name {
		font-weight: 700;
		font-size: 1rem;
		color: var(--phase-color);
		letter-spacing: 0.1em;
	}

	.phase-description {
		margin: 0.5rem 0 0;
		color: var(--text-muted, #888);
		font-size: 0.75rem;
	}

	.gauge-section {
		padding: 0.5rem 0.75rem;
		background: var(--bg-tertiary, #0d0d0d);
		border: 1px solid var(--border-dim, #222);
		border-radius: 6px;
	}

	.gauge-header {
		display: flex;
		justify-content: space-between;
		margin-bottom: 0.25rem;
	}

	.gauge-label {
		color: var(--text-muted, #888);
		font-size: 0.7rem;
		letter-spacing: 0.05em;
	}

	.gauge-value {
		font-weight: 600;
		color: var(--neon-green, #00ff41);
	}

	.gauge-value.warning {
		color: var(--neon-pink, #ff0066);
	}

	.gauge-bar {
		position: relative;
		height: 6px;
		background: var(--bg-secondary, #1a1a1a);
		border-radius: 3px;
		overflow: visible;
	}

	.gauge-fill {
		height: 100%;
		background: linear-gradient(90deg, var(--neon-green, #00ff41), var(--neon-cyan, #00ffff));
		border-radius: 3px;
		transition: width 0.3s ease;
	}

	.gauge-fill.warning {
		background: linear-gradient(90deg, var(--neon-pink, #ff0066), var(--neon-yellow, #ffff00));
	}

	.gauge-threshold {
		position: absolute;
		top: -2px;
		width: 2px;
		height: 10px;
		background: var(--text-muted, #888);
	}

	.metrics-row {
		display: flex;
		gap: 0.5rem;
	}

	.metric {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		padding: 0.5rem;
		background: var(--bg-tertiary, #0d0d0d);
		border: 1px solid var(--border-dim, #222);
		border-radius: 6px;
	}

	.metric-label {
		font-size: 0.6rem;
		color: var(--text-muted, #888);
		letter-spacing: 0.1em;
	}

	.metric-value {
		font-size: 1rem;
		font-weight: 700;
	}

	.metric-value.temp { color: var(--neon-yellow, #ffff00); }
	.metric-value.focus { color: var(--neon-cyan, #00ffff); }
	.metric-value.accuracy { color: var(--neon-green, #00ff41); }

	.drift-section {
		padding: 0.5rem 0.75rem;
		background: rgba(255, 0, 102, 0.1);
		border: 1px solid var(--neon-pink, #ff0066);
		border-radius: 6px;
	}

	.drift-header {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.drift-icon {
		animation: blink 1s infinite;
	}

	@keyframes blink {
		0%, 50% { opacity: 1; }
		51%, 100% { opacity: 0.3; }
	}

	.drift-label {
		font-weight: 600;
		color: var(--neon-pink, #ff0066);
		letter-spacing: 0.05em;
	}

	.drift-info {
		display: flex;
		justify-content: space-between;
		margin-top: 0.25rem;
		font-size: 0.75rem;
	}

	.drift-signal {
		color: var(--text-primary, #fff);
	}

	.drift-magnitude {
		color: var(--text-muted, #888);
	}

	.patterns-section {
		padding: 0.5rem 0.75rem;
		background: var(--bg-tertiary, #0d0d0d);
		border: 1px solid var(--border-dim, #222);
		border-radius: 6px;
	}

	.section-title {
		margin: 0 0 0.5rem;
		font-size: 0.7rem;
		color: var(--text-muted, #888);
		letter-spacing: 0.1em;
	}

	.patterns-list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.pattern-item {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.pattern-topic {
		color: var(--text-primary, #fff);
		font-size: 0.75rem;
	}

	.pattern-freq {
		padding: 0.1rem 0.4rem;
		background: var(--neon-cyan, #00ffff);
		color: #000;
		border-radius: 10px;
		font-size: 0.65rem;
		font-weight: 600;
	}

	.suggestion-section {
		padding: 0.75rem;
		background: rgba(0, 255, 255, 0.05);
		border: 1px solid var(--neon-cyan, #00ffff);
		border-radius: 6px;
	}

	.suggestion-section.pulse {
		animation: suggestion-pulse 2s ease-in-out infinite;
	}

	@keyframes suggestion-pulse {
		0%, 100% {
			box-shadow: 0 0 5px rgba(0, 255, 255, 0.3);
		}
		50% {
			box-shadow: 0 0 15px rgba(0, 255, 255, 0.6), 0 0 25px rgba(0, 255, 255, 0.3);
		}
	}

	.suggestion-header {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.5rem;
	}

	.suggestion-icon {
		font-size: 1.2rem;
	}

	.suggestion-label {
		font-weight: 600;
		color: var(--neon-cyan, #00ffff);
		letter-spacing: 0.05em;
	}

	.suggestion-text {
		margin: 0;
		color: var(--text-primary, #fff);
		font-size: 0.8rem;
		line-height: 1.4;
	}

	.stats-row {
		display: flex;
		gap: 0.5rem;
		padding: 0.5rem 0;
		border-top: 1px solid var(--border-dim, #222);
		border-bottom: 1px solid var(--border-dim, #222);
	}

	.stat {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
	}

	.stat-value {
		font-size: 1rem;
		font-weight: 700;
		color: var(--text-primary, #fff);
	}

	.stat-label {
		font-size: 0.6rem;
		color: var(--text-muted, #888);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.transitions-section {
		padding: 0.5rem 0.75rem;
		background: var(--bg-tertiary, #0d0d0d);
		border: 1px solid var(--border-dim, #222);
		border-radius: 6px;
	}

	.transitions-list {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.transition-item {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.75rem;
	}

	.transition-from,
	.transition-to {
		font-weight: 600;
		text-transform: uppercase;
	}

	.transition-arrow {
		color: var(--text-muted, #888);
	}
</style>
