<script lang="ts">
  import { T, useTask } from '@threlte/core';
  import { spring } from 'svelte/motion';

  // Core configuration
  const CORE_SIZE = 4;
  const COLORS = {
    core: '#ff0055',
    emissive: '#ff0055'
  };

  // Reactive scale for pulse effect
  let scale = 1;
  let rotationY = 0;
  let rotationZ = 0;

  // Pulse animation
  useTask((delta) => {
    const time = performance.now() / 1000;

    // Breathing pulse
    scale = 1 + Math.sin(time * 2) * 0.1;

    // Slow rotation
    rotationY -= delta * 0.5;
    rotationZ += delta * 0.2;
  });
</script>

<!-- Outer wireframe shell -->
<T.Group scale={[scale, scale, scale]}>
  <T.Mesh rotation.y={rotationY} rotation.z={rotationZ}>
    <T.IcosahedronGeometry args={[CORE_SIZE, 1]} />
    <T.MeshStandardMaterial
      color="#000000"
      emissive={COLORS.emissive}
      emissiveIntensity={0.8}
      wireframe={true}
      roughness={0.1}
      metalness={0.9}
    />
  </T.Mesh>

  <!-- Inner solid glow -->
  <T.Mesh rotation.y={rotationY * 1.5} rotation.z={rotationZ * -1}>
    <T.IcosahedronGeometry args={[CORE_SIZE * 0.6, 2]} />
    <T.MeshBasicMaterial color={COLORS.core} />
  </T.Mesh>

  <!-- Core point light - pulses with scale -->
  <T.PointLight
    color={COLORS.core}
    intensity={2 + scale}
    distance={30}
  />
</T.Group>
