/**
 * Config Store - Feature flags from backend /api/config endpoint.
 *
 * Loaded once on app startup to determine which UI components to render.
 */

import { writable, derived } from 'svelte/store';

export interface AppConfig {
    features: {
        swarm_loop: boolean;
        memory_space_3d: boolean;
        chat_basic: boolean;
        dark_mode: boolean;
        analytics_dashboard: boolean;
    };
    tier: 'basic' | 'advanced' | 'full';
    mode: 'personal' | 'enterprise';
    memory_enabled: boolean;
}

// Default config - full features (personal mode)
const defaultConfig: AppConfig = {
    features: {
        swarm_loop: true,
        memory_space_3d: true,
        chat_basic: true,
        dark_mode: true,
        analytics_dashboard: true,
    },
    tier: 'full',
    mode: 'personal',
    memory_enabled: true,
};

// Main config store
export const config = writable<AppConfig>(defaultConfig);

// Loading state
export const configLoading = writable<boolean>(true);

// Derived stores for easy access
export const isEnterpriseMode = derived(config, $config => $config.mode === 'enterprise');
export const isBasicTier = derived(config, $config => $config.tier === 'basic');
export const isMemoryEnabled = derived(config, $config => $config.memory_enabled);

// Feature flag derived stores
export const showSwarm = derived(config, $config => $config.features.swarm_loop);
export const showMemorySpace = derived(config, $config => $config.features.memory_space_3d);
export const showAnalytics = derived(config, $config => $config.features.analytics_dashboard);

/**
 * Load config from backend API.
 *
 * Call once on app startup (in +layout.svelte).
 */
export async function loadConfig(apiBase: string = ''): Promise<void> {
    configLoading.set(true);

    try {
        const response = await fetch(`${apiBase}/api/config`);

        if (response.ok) {
            const data = await response.json();
            config.set({
                features: {
                    swarm_loop: data.features?.swarm_loop ?? true,
                    memory_space_3d: data.features?.memory_space_3d ?? true,
                    chat_basic: data.features?.chat_basic ?? true,
                    dark_mode: data.features?.dark_mode ?? true,
                    analytics_dashboard: data.features?.analytics_dashboard ?? true,
                },
                tier: data.tier ?? 'full',
                mode: data.mode ?? 'personal',
                memory_enabled: data.memory_enabled ?? true,
            });
            console.log('[Config] Loaded:', data);
        } else {
            console.warn('[Config] Failed to load, using defaults');
        }
    } catch (err) {
        console.error('[Config] Error loading config:', err);
        // Keep default config
    } finally {
        configLoading.set(false);
    }
}
