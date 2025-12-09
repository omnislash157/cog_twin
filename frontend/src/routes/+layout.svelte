<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { theme } from '$lib/stores/theme';
	import { loadConfig, configLoading } from '$lib/stores/config';

	onMount(async () => {
		// Load config from backend on app startup
		const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000';
		await loadConfig(apiBase);
	});
</script>

{#if $configLoading}
	<div class="loading-screen">
		<div class="spinner"></div>
		<p>Loading...</p>
	</div>
{:else}
	<div class:normie-mode={$theme === 'normie'}>
		<slot />
	</div>
{/if}

<style>
	div {
		min-height: 100vh;
	}

	.loading-screen {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		min-height: 100vh;
		background: var(--bg-primary, #1a1a2e);
		color: var(--text-primary, #e0e0e0);
	}

	.spinner {
		width: 40px;
		height: 40px;
		border: 3px solid var(--border-color, #333);
		border-top: 3px solid var(--accent-color, #00ff88);
		border-radius: 50%;
		animation: spin 1s linear infinite;
	}

	@keyframes spin {
		0% { transform: rotate(0deg); }
		100% { transform: rotate(360deg); }
	}
</style>
