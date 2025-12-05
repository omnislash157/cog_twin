import { writable, get } from 'svelte/store';
import { websocket } from './websocket';

interface Message {
	role: 'user' | 'assistant';
	content: string;
	timestamp: Date;
	traceId?: string;
}

interface CognitiveState {
	phase: string;
	temperature: number;
	driftDetected: boolean;
	gapCount: number;
}

export interface SessionAnalytics {
	phase: string;
	phaseDescription: string;
	stability: number;
	temperature: number;
	focusScore: number;
	driftSignal: string | null;
	driftMagnitude: number;
	recurringPatterns: Array<{ topic: string; frequency: number; recency: number }>;
	hotspotTopics: Array<{ memory_id: string; temperature: number }>;
	emergingTopics: Array<{ memory_id: string; burst_intensity: number }>;
	sessionDurationMinutes: number;
	totalQueries: number;
	totalTokens: number;
	predictionAccuracy: number;
	suggestion: string;
	recentTransitions: Array<{ timestamp: string; from: string; to: string }>;
}

const DEFAULT_ANALYTICS: SessionAnalytics = {
	phase: 'idle',
	phaseDescription: 'Waiting for input',
	stability: 1.0,
	temperature: 0.5,
	focusScore: 0.0,
	driftSignal: null,
	driftMagnitude: 0.0,
	recurringPatterns: [],
	hotspotTopics: [],
	emergingTopics: [],
	sessionDurationMinutes: 0,
	totalQueries: 0,
	totalTokens: 0,
	predictionAccuracy: 0.0,
	suggestion: '',
	recentTransitions: [],
};

interface SessionState {
	messages: Message[];
	currentStream: string;
	inputValue: string;
	cognitiveState: CognitiveState;
	analytics: SessionAnalytics;
	isStreaming: boolean;
}

function createSessionStore() {
	const store = writable<SessionState>({
		messages: [],
		currentStream: '',
		inputValue: '',
		cognitiveState: {
			phase: 'idle',
			temperature: 0.5,
			driftDetected: false,
			gapCount: 0,
		},
		analytics: { ...DEFAULT_ANALYTICS },
		isStreaming: false,
	});

	const { subscribe, set, update } = store;

	// Set up WebSocket message handler
	let unsubscribe: (() => void) | null = null;

	function initMessageHandler() {
		unsubscribe = websocket.onMessage((data) => {
			switch (data.type) {
				case 'stream_chunk':
					update(s => ({
						...s,
						currentStream: s.currentStream + data.content,
						isStreaming: !data.done,
					}));

					if (data.done) {
						update(s => {
							const assistantMsg: Message = {
								role: 'assistant',
								content: s.currentStream,
								timestamp: new Date(),
								traceId: data.trace_id,
							};
							return {
								...s,
								messages: [...s.messages, assistantMsg],
								currentStream: '',
								isStreaming: false,
							};
						});
					}
					break;

				case 'cognitive_state':
					update(s => ({
						...s,
						cognitiveState: {
							phase: data.phase,
							temperature: data.temperature,
							driftDetected: data.drift_detected,
							gapCount: data.gap_count,
						},
					}));
					break;

				case 'session_analytics':
					update(s => ({
						...s,
						analytics: {
							phase: data.phase ?? s.analytics.phase,
							phaseDescription: data.phase_description ?? s.analytics.phaseDescription,
							stability: data.stability ?? s.analytics.stability,
							temperature: data.temperature ?? s.analytics.temperature,
							focusScore: data.focus_score ?? s.analytics.focusScore,
							driftSignal: data.drift_signal ?? s.analytics.driftSignal,
							driftMagnitude: data.drift_magnitude ?? s.analytics.driftMagnitude,
							recurringPatterns: data.recurring_patterns ?? s.analytics.recurringPatterns,
							hotspotTopics: data.hotspot_topics ?? s.analytics.hotspotTopics,
							emergingTopics: data.emerging_topics ?? s.analytics.emergingTopics,
							sessionDurationMinutes: data.session_duration_minutes ?? s.analytics.sessionDurationMinutes,
							totalQueries: data.total_queries ?? s.analytics.totalQueries,
							totalTokens: data.total_tokens ?? s.analytics.totalTokens,
							predictionAccuracy: data.prediction_accuracy ?? s.analytics.predictionAccuracy,
							suggestion: data.suggestion ?? s.analytics.suggestion,
							recentTransitions: data.recent_transitions ?? s.analytics.recentTransitions,
						},
					}));
					break;

				case 'connected':
					console.log('[Session] Connected:', data.session_id);
					break;

				case 'error':
					console.error('[Session] Error:', data.message);
					break;
			}
		});
	}

	return {
		subscribe,

		init(sessionId: string) {
			websocket.connect(sessionId);
			initMessageHandler();
		},

		cleanup() {
			if (unsubscribe) {
				unsubscribe();
				unsubscribe = null;
			}
			websocket.disconnect();
		},

		sendMessage(content?: string) {
			// Use parameter or fall back to store's inputValue
			const messageContent = content || get(store).inputValue.trim();

			if (!messageContent) return;

			// Add user message
			const userMsg: Message = {
				role: 'user',
				content: messageContent,
				timestamp: new Date(),
			};

			update(s => ({
				...s,
				messages: [...s.messages, userMsg],
				inputValue: '',
				currentStream: '',
				isStreaming: true,
			}));

			// Send to backend
			websocket.send({
				type: 'message',
				content: messageContent,
			});
		},

		clearMessages() {
			update(s => ({
				...s,
				messages: [],
				currentStream: '',
			}));
		},
	};
}

export const session = createSessionStore();
