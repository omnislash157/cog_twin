<script lang="ts">
  import { ARTIFACT_TYPES } from '../artifacts/registry';
  import type { StoredArtifact } from '../stores/artifacts';
  import { artifacts } from '../stores';

  export let item: StoredArtifact;

  $: meta = ARTIFACT_TYPES[item.artifact.type] || { label: 'Unknown', icon: '❓' };
  $: title = item.artifact.title || meta.label;
</script>

<div class="artifact-pane" class:suggested={item.suggested}>
  <header>
    <span class="icon">{meta.icon}</span>
    <span class="title">{title}</span>
    {#if item.suggested}
      <span class="badge">suggested</span>
    {/if}
    <button class="dismiss" on:click={() => artifacts.dismiss(item.id)}>×</button>
  </header>

  <div class="content">
    {#if item.artifact.type === 'code' && item.artifact.code}
      <pre><code class="lang-{item.artifact.lang || 'text'}">{item.artifact.code}</code></pre>
    {:else if item.artifact.type === 'list' && item.artifact.items}
      <ul>
        {#each item.artifact.items as listItem}
          <li>{listItem}</li>
        {/each}
      </ul>
    {:else if item.artifact.ids}
      <p class="meta">Memories: {item.artifact.ids.join(', ')}</p>
    {:else if item.artifact.query}
      <p class="meta">Query: {item.artifact.query}</p>
      {#if item.artifact.range}
        <p class="meta">Range: {item.artifact.range}</p>
      {/if}
    {:else}
      <p class="meta">Artifact ready for rendering</p>
    {/if}
  </div>
</div>

<style>
  .artifact-pane {
    background: var(--bg-secondary, #1a1a1a);
    border: 1px solid var(--neon-green, #00ff41);
    border-radius: 8px;
    padding: 0;
    margin-bottom: 0.5rem;
    overflow: hidden;
  }

  .artifact-pane.suggested {
    border-color: var(--neon-cyan, #00ffff);
  }

  header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--bg-tertiary, #0d0d0d);
    border-bottom: 1px solid var(--border-color, #333);
  }

  .icon { font-size: 1rem; }
  .title { flex: 1; font-weight: 600; font-size: 0.875rem; }

  .badge {
    font-size: 0.625rem;
    text-transform: uppercase;
    padding: 0.125rem 0.375rem;
    background: var(--neon-cyan, #00ffff);
    color: black;
    border-radius: 4px;
  }

  .dismiss {
    background: none;
    border: none;
    color: var(--text-muted, #888);
    cursor: pointer;
    font-size: 1.25rem;
    line-height: 1;
    padding: 0;
  }
  .dismiss:hover { color: var(--text-primary, #fff); }

  .content {
    padding: 0.75rem;
    font-size: 0.875rem;
  }

  pre {
    margin: 0;
    padding: 0.5rem;
    background: var(--bg-primary, #0a0a0a);
    border-radius: 4px;
    overflow-x: auto;
  }

  code { font-family: 'JetBrains Mono', monospace; }

  ul {
    margin: 0;
    padding-left: 1.25rem;
  }

  .meta {
    margin: 0.25rem 0;
    color: var(--text-muted, #888);
  }
</style>
