<script lang="ts">
  import { T, useTask } from '@threlte/core';
  import { OrbitControls } from '@threlte/extras';
  import AgentNode from './AgentNode.svelte';
  import { swarm, AGENT_ORDER } from '$lib/stores';

  // Core brain pulse
  let corePulse = 1;
  let coreRotation = 0;
  let dataFlowOffset = 0;

  useTask((delta) => {
    const time = performance.now() / 1000;

    // Core pulsing
    corePulse = 1 + Math.sin(time * 2) * 0.1;
    coreRotation += delta * 0.5;

    // Data flow animation
    dataFlowOffset = (time * 2) % 1;
  });

  // Check if swarm is active
  $: isActive = $swarm.active;
</script>

<!-- Camera -->
<T.PerspectiveCamera
  makeDefault
  position={[0, 15, 25]}
  fov={60}
/>
<OrbitControls
  enableDamping
  dampingFactor={0.05}
  minDistance={10}
  maxDistance={50}
  target={[0, 2, 0]}
/>

<!-- Central core (orchestrator) -->
<T.Group position.y={0}>
  <!-- Core sphere -->
  <T.Mesh
    rotation.y={coreRotation}
    scale={[corePulse * 2, corePulse * 2, corePulse * 2]}
  >
    <T.IcosahedronGeometry args={[1, 2]} />
    <T.MeshStandardMaterial
      color={isActive ? '#00ffff' : '#333333'}
      emissive={isActive ? '#00ffff' : '#111111'}
      emissiveIntensity={isActive ? 1.5 : 0.2}
      roughness={0.1}
      metalness={0.9}
      wireframe
    />
  </T.Mesh>

  <!-- Inner solid core -->
  <T.Mesh
    rotation.y={-coreRotation * 2}
    scale={[corePulse * 0.8, corePulse * 0.8, corePulse * 0.8]}
  >
    <T.OctahedronGeometry args={[1, 0]} />
    <T.MeshStandardMaterial
      color="#00ff41"
      emissive="#00ff41"
      emissiveIntensity={isActive ? 2 : 0.3}
      roughness={0.2}
      metalness={0.8}
    />
  </T.Mesh>

  <!-- Core glow -->
  <T.PointLight
    color={isActive ? '#00ffff' : '#333333'}
    intensity={isActive ? 3 : 0.5}
    distance={20}
  />

  <!-- Status ring -->
  <T.Mesh rotation.x={Math.PI / 2} position.y={-0.1}>
    <T.RingGeometry args={[2.5, 2.8, 64]} />
    <T.MeshBasicMaterial
      color={isActive ? '#00ff41' : '#222222'}
      transparent
      opacity={0.5}
      side={2}
    />
  </T.Mesh>
</T.Group>

<!-- Agent nodes in arc -->
{#each AGENT_ORDER as agent, i}
  <AgentNode
    {agent}
    status={$swarm.agents[agent].status}
    index={i}
    totalAgents={AGENT_ORDER.length}
  />
{/each}

<!-- Data flow particles (when active) -->
{#if isActive}
  {#each Array(20) as _, i}
    {@const angle = (i / 20) * Math.PI * 2}
    {@const radius = 10 + Math.sin(i * 0.5) * 2}
    {@const height = Math.sin((dataFlowOffset + i * 0.05) * Math.PI * 2) * 3}
    <T.Mesh
      position.x={Math.cos(angle + dataFlowOffset * 2) * radius}
      position.y={1 + height}
      position.z={Math.sin(angle + dataFlowOffset * 2) * radius}
    >
      <T.SphereGeometry args={[0.08, 8, 8]} />
      <T.MeshBasicMaterial color="#00ff41" />
    </T.Mesh>
  {/each}
{/if}

<!-- Ground grid -->
<T.GridHelper args={[40, 40, '#1a1a1a', '#0d0d0d']} />

<!-- Ground plane (for shadows) -->
<T.Mesh rotation.x={-Math.PI / 2} position.y={-0.01} receiveShadow>
  <T.PlaneGeometry args={[50, 50]} />
  <T.MeshStandardMaterial
    color="#050505"
    transparent
    opacity={0.8}
  />
</T.Mesh>
