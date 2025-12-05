/**
 * Artifact Store - holds emitted artifacts for rendering
 */

import { writable, derived } from 'svelte/store';
import type { ArtifactData } from '../artifacts/registry';

export interface StoredArtifact {
  id: string;
  artifact: ArtifactData;
  suggested: boolean;
  timestamp: Date;
  dismissed: boolean;
}

function createArtifactStore() {
  const { subscribe, update, set } = writable<StoredArtifact[]>([]);

  let counter = 0;

  return {
    subscribe,

    /** Add new artifact from WebSocket */
    add(artifact: ArtifactData, suggested: boolean = false) {
      const id = `artifact_${Date.now()}_${counter++}`;
      update(items => [...items, {
        id,
        artifact,
        suggested,
        timestamp: new Date(),
        dismissed: false
      }]);
      return id;
    },

    /** Dismiss artifact (hide but keep in history) */
    dismiss(id: string) {
      update(items => items.map(item =>
        item.id === id ? { ...item, dismissed: true } : item
      ));
    },

    /** Clear all artifacts */
    clear() {
      set([]);
    },

    /** Remove old artifacts (keep last N) */
    prune(keep: number = 20) {
      update(items => items.slice(-keep));
    }
  };
}

export const artifacts = createArtifactStore();

/** Only visible (non-dismissed) artifacts */
export const visibleArtifacts = derived(artifacts, $artifacts =>
  $artifacts.filter(a => !a.dismissed)
);
