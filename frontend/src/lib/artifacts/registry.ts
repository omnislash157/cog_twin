/**
 * Artifact Registry - maps artifact types to Svelte components
 */

import type { ComponentType } from 'svelte';

export interface ArtifactData {
  type: string;
  title?: string;
  ids?: string[];
  query?: string;
  range?: string;
  lang?: string;
  code?: string;
  items?: string[];
}

export interface ArtifactPayload {
  type: 'artifact_emit';
  artifact: ArtifactData;
  suggested: boolean;
}

// Registry will be populated with actual components
// For now, we track artifact types and their display names
export const ARTIFACT_TYPES: Record<string, { label: string; icon: string }> = {
  memory_card: { label: 'Memory', icon: '🧠' },
  comparison: { label: 'Compare', icon: '⚖️' },
  timeline: { label: 'Timeline', icon: '📅' },
  code: { label: 'Code', icon: '💻' },
  synthesis: { label: 'Synthesis', icon: '✨' },
  list: { label: 'List', icon: '📋' },
};

export function isValidArtifactType(type: string): boolean {
  return type in ARTIFACT_TYPES;
}
