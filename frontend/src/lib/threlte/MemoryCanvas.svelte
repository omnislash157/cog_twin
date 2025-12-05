<script lang="ts">
  import { Canvas } from '@threlte/core';
  import Scene from './Scene.svelte';
  import MemorySpace from './MemorySpace.svelte';
  import SwarmSpace from './SwarmSpace.svelte';
  import { createEventDispatcher } from 'svelte';
  import { swarm } from '$lib/stores';

  export let memories: Array<{
    id: string;
    content_preview: string;
    source: 'vector' | 'episodic' | 'grep' | 'live' | 'synthesis';
    relevance: number;
    cluster_id?: number;
    readonly?: boolean;
    parentIds?: string[];
  }> = [];

  // View mode: 'memory' or 'swarm'
  export let mode: 'memory' | 'swarm' = 'memory';

  const dispatch = createEventDispatcher<{
    selectMemory: { id: string };
    synthesize: { sourceId: string; targetId: string };
  }>();

  function handleSelectMemory(e: CustomEvent<{ id: string }>) {
    dispatch('selectMemory', e.detail);
  }

  function handleSynthesize(e: CustomEvent<{ sourceId: string; targetId: string }>) {
    dispatch('synthesize', e.detail);
  }

  function toggleMode() {
    mode = mode === 'memory' ? 'swarm' : 'memory';
  }
</script>

<div class="canvas-container">
  <Canvas>
    <Scene />
    {#if mode === 'swarm'}
      <SwarmSpace />
    {:else}
      <MemorySpace
        {memories}
        on:selectMemory={handleSelectMemory}
        on:synthesize={handleSynthesize}
      />
    {/if}
  </Canvas>

  <!-- Overlay HUD -->
  <div class="hud-overlay">
    <div class="hud-header">
      <div class="hud-title">{mode === 'swarm' ? 'SWARM SPACE' : 'MEMORY SPACE'}</div>
      <button class="mode-toggle" on:click={toggleMode}>
        {mode === 'swarm' ? '🧠' : '🐝'}
      </button>
    </div>
    {#if mode === 'swarm'}
      <div class="hud-stat">STATUS: {$swarm.active ? 'ACTIVE' : 'IDLE'}</div>
      <div class="hud-stat">WAVE: {$swarm.currentWave}</div>
    {:else}
      <div class="hud-stat">NODES: {memories.length}</div>
    {/if}
    <div class="hud-hint">
      {mode === 'swarm'
        ? 'Watch agents process in real-time'
        : 'Drag nodes to synthesize'} &bull; Scroll to zoom &bull; Drag to orbit
    </div>
  </div>
</div>

<style>
  .canvas-container {
    width: 100%;
    height: 100%;
    position: relative;
    background: #050505;
  }

  .hud-overlay {
    position: absolute;
    top: 1rem;
    left: 1rem;
    pointer-events: none;
    z-index: 10;
  }

  .hud-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .hud-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: #00ff41;
    text-shadow: 0 0 10px #00ff41, 0 0 20px #00ff41;
    letter-spacing: 0.2em;
    text-transform: uppercase;
  }

  .hud-stat {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #00ff41;
    opacity: 0.7;
    margin-top: 0.25rem;
  }

  .hud-hint {
    position: absolute;
    bottom: 1rem;
    left: 1rem;
    right: 1rem;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.625rem;
    color: rgba(255, 255, 255, 0.4);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .mode-toggle {
    background: rgba(0, 255, 65, 0.2);
    border: 1px solid #00ff41;
    border-radius: 4px;
    padding: 0.25rem 0.5rem;
    cursor: pointer;
    font-size: 1rem;
    pointer-events: auto;
    transition: all 0.2s;
  }

  .mode-toggle:hover {
    background: #00ff41;
    transform: scale(1.1);
  }
</style>
