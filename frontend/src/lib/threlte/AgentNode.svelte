<script lang="ts">
  import { T, useTask } from '@threlte/core';
  import { Text } from '@threlte/extras';
  import type { AgentName, AgentStatus } from '$lib/stores';

  export let agent: AgentName;
  export let status: AgentStatus = 'idle';
  export let index: number = 0;
  export let totalAgents: number = 4;

  // Agent colors
  const AGENT_COLORS: Record<AgentName, string> = {
    config: '#00ffff',      // Cyan
    executor: '#00ff41',    // Green
    reviewer: '#ffff00',    // Yellow
    quality_gate: '#ff00ff' // Magenta
  };

  // Status colors
  const STATUS_COLORS: Record<AgentStatus, string> = {
    idle: '#444444',
    thinking: '#00ffff',
    done: '#00ff41',
    failed: '#ff0066'
  };

  // Agent labels
  const AGENT_LABELS: Record<AgentName, string> = {
    config: 'CONFIG',
    executor: 'EXEC',
    reviewer: 'REVIEW',
    quality_gate: 'Q-GATE'
  };

  // Position in arc around center (radius 15, spread 180 degrees)
  const RADIUS = 12;
  const arcAngle = (index / (totalAgents - 1)) * Math.PI - Math.PI / 2;
  const baseX = Math.cos(arcAngle) * RADIUS;
  const baseZ = Math.sin(arcAngle) * RADIUS;
  const baseY = 2;

  // Animated position
  let posY = baseY;
  let pulseScale = 1;
  let rotationY = 0;
  let glowIntensity = 0.3;

  // Animation parameters based on status
  $: isActive = status === 'thinking';
  $: isDone = status === 'done';
  $: isFailed = status === 'failed';
  $: currentColor = isActive ? AGENT_COLORS[agent] : STATUS_COLORS[status];
  $: targetGlow = isActive ? 2.5 : (isDone ? 1.5 : (isFailed ? 2.0 : 0.3));

  useTask((delta) => {
    const time = performance.now() / 1000;

    // Floating animation
    posY = baseY + Math.sin(time * 2 + index * 0.5) * 0.2;

    // Pulsing when active
    if (isActive) {
      pulseScale = 1 + Math.sin(time * 4) * 0.15;
      rotationY += delta * 2;
    } else {
      pulseScale += (1 - pulseScale) * 0.1;
      rotationY += delta * 0.2;
    }

    // Smooth glow transition
    glowIntensity += (targetGlow - glowIntensity) * 0.1;
  });

  // Node size
  const nodeSize = 1.2;
</script>

<T.Group position.x={baseX} position.y={posY} position.z={baseZ}>
  <!-- Main agent node (cube for agents) -->
  <T.Mesh
    rotation.y={rotationY}
    scale={[pulseScale * nodeSize, pulseScale * nodeSize, pulseScale * nodeSize]}
  >
    <T.BoxGeometry args={[1, 1, 1]} />
    <T.MeshStandardMaterial
      color={currentColor}
      emissive={currentColor}
      emissiveIntensity={glowIntensity}
      roughness={0.2}
      metalness={0.8}
      wireframe={status === 'idle'}
    />
  </T.Mesh>

  <!-- Inner core (shows when active) -->
  {#if isActive}
    <T.Mesh rotation.y={-rotationY * 2}>
      <T.OctahedronGeometry args={[0.4, 0]} />
      <T.MeshBasicMaterial
        color={AGENT_COLORS[agent]}
        transparent
        opacity={0.8}
      />
    </T.Mesh>
  {/if}

  <!-- Status ring -->
  <T.Mesh rotation.x={Math.PI / 2} position.y={-1.2}>
    <T.RingGeometry args={[0.8, 1.0, 32]} />
    <T.MeshBasicMaterial
      color={currentColor}
      transparent
      opacity={isActive ? 0.6 : 0.2}
      side={2}
    />
  </T.Mesh>

  <!-- Failed X indicator -->
  {#if isFailed}
    <T.Mesh position={[0, 0, 0.8]} rotation.z={Math.PI / 4}>
      <T.BoxGeometry args={[0.1, 1.2, 0.1]} />
      <T.MeshBasicMaterial color="#ff0066" />
    </T.Mesh>
    <T.Mesh position={[0, 0, 0.8]} rotation.z={-Math.PI / 4}>
      <T.BoxGeometry args={[0.1, 1.2, 0.1]} />
      <T.MeshBasicMaterial color="#ff0066" />
    </T.Mesh>
  {/if}

  <!-- Done checkmark indicator -->
  {#if isDone}
    <T.PointLight
      color="#00ff41"
      intensity={1}
      distance={5}
    />
  {/if}

  <!-- Agent glow light -->
  <T.PointLight
    color={currentColor}
    intensity={glowIntensity}
    distance={isActive ? 10 : 5}
  />

  <!-- Connection line to center (data flow) -->
  <T.Line
    points={[
      [0, -0.6, 0],
      [-baseX * 0.8, -1.5, -baseZ * 0.8]
    ]}
  >
    <T.LineBasicMaterial
      color={currentColor}
      transparent
      opacity={isActive ? 0.8 : 0.2}
    />
  </T.Line>

  <!-- Label (floating above node) -->
  <T.Group position.y={1.8}>
    <Text
      text={AGENT_LABELS[agent]}
      fontSize={0.4}
      color={currentColor}
      anchorX="center"
      anchorY="middle"
    />
  </T.Group>
</T.Group>
