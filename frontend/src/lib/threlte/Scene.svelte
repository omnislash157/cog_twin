<script lang="ts">
  import { T, useTask } from '@threlte/core';
  import {
    EffectComposer,
    EffectPass,
    RenderPass,
    BloomEffect,
    NoiseEffect,
    BlendFunction,
    VignetteEffect
  } from 'postprocessing';
  import { useThrelte } from '@threlte/core';
  import { onMount } from 'svelte';

  // Toxic green color palette
  const COLORS = {
    background: '#050505',
    matrixGreen: '#00ff41',
    voltageYellow: '#ccff00',
    coreGlow: '#00ff41'
  };

  const { scene, renderer, camera, size } = useThrelte();

  let composer: EffectComposer;

  onMount(() => {
    // Build post-processing pipeline
    composer = new EffectComposer(renderer);

    // Base render
    const renderPass = new RenderPass(scene, camera.current);
    composer.addPass(renderPass);

    // Radioactive bloom - high strength for that glow
    const bloomEffect = new BloomEffect({
      luminanceThreshold: 0.15,
      luminanceSmoothing: 0.9,
      intensity: 1.8,
      radius: 0.6
    });

    // CRT film grain texture
    const noiseEffect = new NoiseEffect({
      blendFunction: BlendFunction.OVERLAY,
      premultiply: true
    });
    noiseEffect.blendMode.opacity.value = 0.5;

    // Vignette for that cockpit feel
    const vignetteEffect = new VignetteEffect({
      darkness: 0.5,
      offset: 0.3
    });

    const effectPass = new EffectPass(camera.current, bloomEffect, noiseEffect, vignetteEffect);
    composer.addPass(effectPass);

    return () => {
      composer.dispose();
    };
  });

  // Replace default render with composer
  useTask((delta) => {
    if (composer) {
      composer.render(delta);
    }
  }, { autoInvalidate: false });
</script>

<!-- Fog for depth -->
<T.FogExp2 args={[COLORS.background, 0.015]} attach="fog" />

<!-- Ambient base -->
<T.AmbientLight intensity={0.4} color={COLORS.matrixGreen} />

<!-- Key light from above -->
<T.PointLight
  position={[0, 30, 0]}
  intensity={2}
  color="#ffffff"
  distance={100}
/>

<!-- Rim light - cyan accent -->
<T.SpotLight
  position={[-50, 0, 50]}
  intensity={10}
  color="#00ffff"
  angle={0.5}
  penumbra={0.5}
/>
