<script lang="ts">
  import { Canvas } from '@threlte/core';
  import Scene from './Scene.svelte';
  import MemorySpace from './MemorySpace.svelte';
  import { createEventDispatcher } from 'svelte';

  export let memories: Array<{
    id: string;
    content_preview: string;
    source: 'vector' | 'episodic' | 'grep' | 'live' | 'synthesis';
    relevance: number;
    cluster_id?: number;
    readonly?: boolean;
    parentIds?: string[];
  }> = [];

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
</script>

<div class="canvas-container">
  <Canvas>
    <Scene />
    <MemorySpace
      {memories}
      on:selectMemory={handleSelectMemory}
      on:synthesize={handleSynthesize}
    />
  </Canvas>

  <!-- Overlay HUD -->
  <div class="hud-overlay">
    <div class="hud-title">MEMORY SPACE</div>
    <div class="hud-stat">NODES: {memories.length}</div>
    <div class="hud-hint">Drag nodes to synthesize &bull; Scroll to zoom &bull; Drag to orbit</div>
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
</style>
