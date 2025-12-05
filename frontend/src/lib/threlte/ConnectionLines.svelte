<script lang="ts">
  import { T, useTask } from '@threlte/core';
  import * as THREE from 'three';

  export let nodePositions: Map<string, {x: number, y: number, z: number}> = new Map();
  export let corePosition: {x: number, y: number, z: number} = { x: 0, y: 0, z: 0 };

  // Connection thresholds
  const NODE_CONNECTION_DISTANCE = 18;
  const CORE_CONNECTION_DISTANCE = 14;

  // Line geometry - updated each frame
  let linePositions: Float32Array = new Float32Array(0);
  let lineColors: Float32Array = new Float32Array(0);
  let lineGeometry: THREE.BufferGeometry;

  const COLOR_NODE = new THREE.Color('#0044aa');
  const COLOR_CORE = new THREE.Color('#ff0055');
  const COLOR_CLOSE = new THREE.Color('#00ff41');

  useTask(() => {
    const positions: number[] = [];
    const colors: number[] = [];

    const nodes = Array.from(nodePositions.entries());

    // Node-to-node connections
    for (let i = 0; i < nodes.length; i++) {
      const [idA, posA] = nodes[i];

      for (let j = i + 1; j < nodes.length; j++) {
        const [idB, posB] = nodes[j];

        const dist = Math.sqrt(
          Math.pow(posA.x - posB.x, 2) +
          Math.pow(posA.y - posB.y, 2) +
          Math.pow(posA.z - posB.z, 2)
        );

        if (dist < NODE_CONNECTION_DISTANCE) {
          // Closer = brighter
          const intensity = 1 - (dist / NODE_CONNECTION_DISTANCE);
          const color = intensity > 0.6 ? COLOR_CLOSE : COLOR_NODE;

          // Line from A to B
          positions.push(posA.x, posA.y, posA.z);
          positions.push(posB.x, posB.y, posB.z);

          // Vertex colors
          colors.push(color.r * intensity, color.g * intensity, color.b * intensity);
          colors.push(color.r * intensity, color.g * intensity, color.b * intensity);
        }
      }

      // Node-to-core connections
      const distToCore = Math.sqrt(
        Math.pow(posA.x - corePosition.x, 2) +
        Math.pow(posA.y - corePosition.y, 2) +
        Math.pow(posA.z - corePosition.z, 2)
      );

      if (distToCore < CORE_CONNECTION_DISTANCE) {
        const intensity = 1 - (distToCore / CORE_CONNECTION_DISTANCE);

        positions.push(posA.x, posA.y, posA.z);
        positions.push(corePosition.x, corePosition.y, corePosition.z);

        colors.push(COLOR_CORE.r * intensity, COLOR_CORE.g * intensity, COLOR_CORE.b * intensity);
        colors.push(COLOR_CORE.r * intensity, COLOR_CORE.g * intensity, COLOR_CORE.b * intensity);
      }
    }

    linePositions = new Float32Array(positions);
    lineColors = new Float32Array(colors);

    // Update geometry
    if (lineGeometry) {
      lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
      lineGeometry.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));
      lineGeometry.attributes.position.needsUpdate = true;
      lineGeometry.attributes.color.needsUpdate = true;
    }
  });
</script>

<T.LineSegments>
  <T.BufferGeometry bind:ref={lineGeometry}>
    <T.BufferAttribute
      attach="attributes-position"
      args={[linePositions, 3]}
    />
    <T.BufferAttribute
      attach="attributes-color"
      args={[lineColors, 3]}
    />
  </T.BufferGeometry>
  <T.LineBasicMaterial
    vertexColors
    transparent
    opacity={0.4}
    blending={THREE.AdditiveBlending}
  />
</T.LineSegments>
