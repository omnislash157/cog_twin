<script lang="ts">
  import { T, useTask } from '@threlte/core';
  import { interactivity } from '@threlte/extras';
  import { createEventDispatcher } from 'svelte';

  export let id: string;
  export let initialPosition: [number, number, number];
  export let source: 'vector' | 'episodic' | 'grep' | 'live' | 'synthesis' = 'vector';
  export let relevance: number = 0.5;
  export let label: string = '';
  export let readonly: boolean = false;
  export let parentIds: string[] = []; // For synthesis nodes

  // All nodes in scene (passed from parent for proximity detection)
  export let allNodePositions: Map<string, {x: number, y: number, z: number}> = new Map();

  const dispatch = createEventDispatcher<{
    select: { id: string };
    synthesize: { sourceId: string; targetId: string };
    dragStart: { id: string };
    dragEnd: { id: string; position: {x: number, y: number, z: number} };
  }>();

  // Color by source type
  const SOURCE_COLORS: Record<string, string> = {
    vector: '#00ff41',     // Matrix green
    episodic: '#00ffff',   // Cyan
    grep: '#ffff00',       // Yellow
    live: '#ff00ff',       // Magenta
    synthesis: '#ff8800'   // Orange (born from fusion)
  };

  const SYNTHESIS_ZONE_RADIUS = 3.5;

  // Physics state
  let position = { x: initialPosition[0], y: initialPosition[1], z: initialPosition[2] };
  let velocity = { x: 0, y: 0, z: 0 };
  let isDragging = false;
  let isHovered = false;
  let nearbyNodeId: string | null = null;
  let inSynthesisZone = false;

  // Original position for spring return
  const origin = { x: initialPosition[0], y: initialPosition[1], z: initialPosition[2] };

  // Physics constants
  const SPRING_STRENGTH = 0.08;
  const DAMPING = 0.90;
  const NOISE_SCALE = 0.02;

  // Visual properties
  $: color = inSynthesisZone ? '#ccff00' : SOURCE_COLORS[source] || SOURCE_COLORS.vector;
  $: size = 0.5 + (relevance * 0.5);
  $: emissiveIntensity = inSynthesisZone ? 3.0 : (isHovered ? 1.5 : (isDragging ? 2.5 : 0.6));
  $: scale = inSynthesisZone ? 1.6 : (isHovered ? 1.3 : (isDragging ? 1.5 : 1.0));
  $: nodeShape = source === 'synthesis' ? 'icosahedron' : 'octahedron';

  interactivity();

  // Physics + synthesis detection
  useTask((delta) => {
    const time = performance.now() / 1000;

    if (!isDragging) {
      // Spring force
      const forceX = (origin.x - position.x) * SPRING_STRENGTH;
      const forceY = (origin.y - position.y) * SPRING_STRENGTH;
      const forceZ = (origin.z - position.z) * SPRING_STRENGTH;

      velocity.x = (velocity.x + forceX) * DAMPING;
      velocity.y = (velocity.y + forceY) * DAMPING;
      velocity.z = (velocity.z + forceZ) * DAMPING;

      position.x += velocity.x;
      position.y += velocity.y;
      position.z += velocity.z;

      // Floating noise
      const nodeOffset = parseInt(id.replace(/\D/g, '') || '0') * 0.1;
      position.x += Math.sin(time + nodeOffset) * NOISE_SCALE;
      position.y += Math.cos(time + nodeOffset * 0.5) * NOISE_SCALE;
      position.z += Math.sin(time * 0.5 + nodeOffset) * NOISE_SCALE;

      inSynthesisZone = false;
      nearbyNodeId = null;
    } else {
      // While dragging: check proximity to other nodes
      let closestDist = Infinity;
      let closestId: string | null = null;

      allNodePositions.forEach((pos, nodeId) => {
        if (nodeId === id) return;
        const dist = Math.sqrt(
          Math.pow(position.x - pos.x, 2) +
          Math.pow(position.y - pos.y, 2) +
          Math.pow(position.z - pos.z, 2)
        );
        if (dist < SYNTHESIS_ZONE_RADIUS && dist < closestDist) {
          closestDist = dist;
          closestId = nodeId;
        }
      });

      inSynthesisZone = closestId !== null;
      nearbyNodeId = closestId;
    }
  });

  function handlePointerEnter() {
    isHovered = true;
    document.body.style.cursor = readonly ? 'not-allowed' : 'grab';
  }

  function handlePointerLeave() {
    isHovered = false;
    document.body.style.cursor = 'default';
  }

  function handlePointerDown(e: PointerEvent) {
    if (readonly) return;
    isDragging = true;
    document.body.style.cursor = 'grabbing';
    dispatch('dragStart', { id });
    e.stopPropagation();
  }

  function handlePointerUp() {
    if (!isDragging) return;

    // Check for synthesis
    if (inSynthesisZone && nearbyNodeId) {
      dispatch('synthesize', { sourceId: id, targetId: nearbyNodeId });
    }

    isDragging = false;
    velocity = { x: 0, y: 0, z: 0 }; // Reset for slingshot
    document.body.style.cursor = 'default';
    dispatch('dragEnd', { id, position: { ...position } });
  }

  function handleClick() {
    if (!isDragging) {
      dispatch('select', { id });
    }
  }
</script>

<T.Group position.x={position.x} position.y={position.y} position.z={position.z}>
  <!-- Main node geometry -->
  <T.Mesh
    scale={[scale * size, scale * size, scale * size]}
    on:pointerenter={handlePointerEnter}
    on:pointerleave={handlePointerLeave}
    on:pointerdown={handlePointerDown}
    on:pointerup={handlePointerUp}
    on:click={handleClick}
  >
    {#if source === 'synthesis'}
      <T.IcosahedronGeometry args={[1, 1]} />
    {:else}
      <T.OctahedronGeometry args={[1, 0]} />
    {/if}
    <T.MeshStandardMaterial
      {color}
      emissive={color}
      {emissiveIntensity}
      roughness={0.2}
      metalness={0.8}
    />
  </T.Mesh>

  <!-- Synthesis zone indicator ring -->
  {#if isDragging}
    <T.Mesh rotation.x={Math.PI / 2}>
      <T.RingGeometry args={[SYNTHESIS_ZONE_RADIUS - 0.1, SYNTHESIS_ZONE_RADIUS, 32]} />
      <T.MeshBasicMaterial
        color={inSynthesisZone ? '#ccff00' : '#00ff41'}
        transparent
        opacity={0.3}
        side={2}
      />
    </T.Mesh>
  {/if}

  <!-- Node glow -->
  <T.PointLight
    {color}
    intensity={inSynthesisZone ? 4 : (isHovered || isDragging ? 2 : 0.5)}
    distance={inSynthesisZone ? 8 : 5}
  />

  <!-- Readonly indicator (subtle inner glow) -->
  {#if readonly}
    <T.Mesh scale={[0.3, 0.3, 0.3]}>
      <T.SphereGeometry args={[1, 8, 8]} />
      <T.MeshBasicMaterial color="#ff0000" transparent opacity={0.5} />
    </T.Mesh>
  {/if}
</T.Group>
