<script lang="ts">
  import { T } from '@threlte/core';
  import { OrbitControls } from '@threlte/extras';
  import { createEventDispatcher } from 'svelte';
  import CoreBrain from './CoreBrain.svelte';
  import MemoryNode from './MemoryNode.svelte';
  import ConnectionLines from './ConnectionLines.svelte';

  // Memory data from parent
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

  // Generate initial positions - spherical distribution
  function generatePosition(index: number, total: number): [number, number, number] {
    const phi = Math.acos(-1 + (2 * index) / total);
    const theta = Math.sqrt(total * Math.PI) * phi;
    const radius = 12 + Math.random() * 10; // 12-22 units from center

    return [
      radius * Math.cos(theta) * Math.sin(phi),
      radius * Math.sin(theta) * Math.sin(phi),
      radius * Math.cos(phi)
    ];
  }

  // Track all node positions for connections + synthesis detection
  let nodePositions: Map<string, {x: number, y: number, z: number}> = new Map();

  // Initialize positions for all memories
  $: {
    const newPositions = new Map<string, {x: number, y: number, z: number}>();
    memories.forEach((mem, i) => {
      if (!nodePositions.has(mem.id)) {
        const [x, y, z] = generatePosition(i, memories.length);
        newPositions.set(mem.id, { x, y, z });
      } else {
        newPositions.set(mem.id, nodePositions.get(mem.id)!);
      }
    });
    nodePositions = newPositions;
  }

  // Handlers
  function handleSelect(e: CustomEvent<{ id: string }>) {
    dispatch('selectMemory', { id: e.detail.id });
  }

  function handleSynthesize(e: CustomEvent<{ sourceId: string; targetId: string }>) {
    dispatch('synthesize', { sourceId: e.detail.sourceId, targetId: e.detail.targetId });
  }

  function handleDragEnd(e: CustomEvent<{ id: string; position: {x: number, y: number, z: number} }>) {
    // Update position tracking
    nodePositions.set(e.detail.id, e.detail.position);
    nodePositions = nodePositions; // Trigger reactivity
  }

  const corePosition = { x: 0, y: 0, z: 0 };
</script>

<!-- Camera with controls -->
<T.PerspectiveCamera
  makeDefault
  position={[0, 20, 40]}
  fov={60}
>
  <OrbitControls
    enableDamping
    dampingFactor={0.05}
    autoRotate
    autoRotateSpeed={0.3}
    maxDistance={80}
    minDistance={15}
  />
</T.PerspectiveCamera>

<!-- The central brain -->
<CoreBrain />

<!-- Dynamic connection lines -->
<ConnectionLines {nodePositions} {corePosition} />

<!-- Memory nodes -->
{#each memories as memory, i (memory.id)}
  {@const pos = nodePositions.get(memory.id) || generatePosition(i, memories.length)}
  <MemoryNode
    id={memory.id}
    initialPosition={[pos.x, pos.y, pos.z]}
    source={memory.source}
    relevance={memory.relevance}
    label={memory.content_preview}
    readonly={memory.readonly || false}
    parentIds={memory.parentIds || []}
    allNodePositions={nodePositions}
    on:select={handleSelect}
    on:synthesize={handleSynthesize}
    on:dragEnd={handleDragEnd}
  />
{/each}

<!-- Ambient particles / dust (optional visual flair) -->
<T.Points>
  <T.BufferGeometry>
    <T.BufferAttribute
      attach="attributes-position"
      args={[new Float32Array(
        Array.from({ length: 200 * 3 }, () => (Math.random() - 0.5) * 60)
      ), 3]}
    />
  </T.BufferGeometry>
  <T.PointsMaterial
    color="#00ff41"
    size={0.1}
    transparent
    opacity={0.3}
    sizeAttenuation
  />
</T.Points>
