<script lang="ts">
	import '../app.css';
	import { theme } from '$lib/stores/theme';
</script>

<div class:normie-mode={$theme === 'normie'}>
	<slot />
</div>

<style>
	div {
		min-height: 100vh;
	}
</style>
