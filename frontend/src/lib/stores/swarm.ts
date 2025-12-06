import { writable, derived, get } from 'svelte/store';

// Agent status types
export type AgentStatus = 'idle' | 'thinking' | 'done' | 'failed';

// Agent names in order
export const AGENT_ORDER = ['config', 'executor', 'reviewer', 'quality_gate'] as const;
export type AgentName = typeof AGENT_ORDER[number];

// Reasoning step from model
export interface ReasoningStep {
	step: number;
	content: string;
}

// Turn data from WebSocket
export interface AgentTurn {
	agent: AgentName;
	tokens_in: number;
	tokens_out: number;
	latency_ms: number;
	reasoning: ReasoningStep[];
	preview: string;
	parsed: Record<string, string>;
	timestamp: string;
}

// Failure event from quality gate
export interface SwarmFailure {
	wave: string;
	failure_type: string;
	agent_blamed: string;
	root_cause: string;
	recommendation: string;
	timestamp: string;
}

// File write event
export interface FileWrite {
	wave: string;
	file_path: string;
	operation: 'create' | 'append' | 'modify';
	lines_added: number;
	preview: string;
	timestamp: string;
}

// Wave summary
export interface WaveSummary {
	wave: string;
	verdict: 'pass' | 'fail' | 'human_review';
	total_tokens_in: number;
	total_tokens_out: number;
	files_modified: string[];
	summary: string;
	timestamp: string;
}

// Per-agent state
export interface AgentState {
	status: AgentStatus;
	tokens_in: number;
	tokens_out: number;
	latency_ms: number;
	reasoning: ReasoningStep[];
	preview: string;
}

// Full swarm state
export interface SwarmState {
	connected: boolean;
	active: boolean;
	projectId: string | null;
	projectName: string | null;
	goal: string | null;
	currentWave: string;
	currentAgent: AgentName | null;
	agents: Record<AgentName, AgentState>;
	turns: AgentTurn[];
	waves: WaveSummary[];
	failures: SwarmFailure[];
	fileWrites: FileWrite[];
	totalTokensIn: number;
	totalTokensOut: number;
	error: string | null;
}

// Initial agent state
const initialAgentState = (): AgentState => ({
	status: 'idle',
	tokens_in: 0,
	tokens_out: 0,
	latency_ms: 0,
	reasoning: [],
	preview: '',
});

// Initial state
const initialState: SwarmState = {
	connected: false,
	active: false,
	projectId: null,
	projectName: null,
	goal: null,
	currentWave: '001',
	currentAgent: null,
	agents: {
		config: initialAgentState(),
		executor: initialAgentState(),
		reviewer: initialAgentState(),
		quality_gate: initialAgentState(),
	},
	turns: [],
	waves: [],
	failures: [],
	fileWrites: [],
	totalTokensIn: 0,
	totalTokensOut: 0,
	error: null,
};

function createSwarmStore() {
	const { subscribe, set, update } = writable<SwarmState>({ ...initialState });

	let ws: WebSocket | null = null;
	let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
	let reconnectAttempts = 0;
	const maxReconnectAttempts = 5;

	function resetAgents() {
		return {
			config: initialAgentState(),
			executor: initialAgentState(),
			reviewer: initialAgentState(),
			quality_gate: initialAgentState(),
		};
	}

	function handleMessage(event: MessageEvent) {
		try {
			const data = JSON.parse(event.data);

			switch (data.type) {
				case 'swarm_status':
					update(s => ({
						...s,
						active: data.active,
						projectId: data.current_project,
					}));
					break;

				case 'swarm_project_start':
					update(s => ({
						...s,
						active: true,
						projectId: data.project_id,
						projectName: data.project_name,
						goal: data.goal,
						currentWave: '001',
						agents: resetAgents(),
						turns: [],
						waves: [],
						failures: [],
						fileWrites: [],
						totalTokensIn: 0,
						totalTokensOut: 0,
					}));
					break;

				case 'swarm_project_end':
					update(s => ({
						...s,
						active: false,
						currentAgent: null,
					}));
					break;

				case 'swarm_wave_start':
					update(s => ({
						...s,
						currentWave: data.wave,
						agents: resetAgents(),
						currentAgent: null,
					}));
					break;

				case 'swarm_wave_end':
					update(s => ({
						...s,
						waves: [...s.waves, {
							wave: data.wave,
							verdict: data.verdict,
							total_tokens_in: data.total_tokens_in,
							total_tokens_out: data.total_tokens_out,
							files_modified: data.files_modified || [],
							summary: data.summary,
							timestamp: data.timestamp,
						}],
						currentAgent: null,
					}));
					break;

				case 'swarm_agent_start':
					update(s => {
						const agent = data.agent as AgentName;
						return {
							...s,
							currentAgent: agent,
							agents: {
								...s.agents,
								[agent]: {
									...s.agents[agent],
									status: 'thinking' as AgentStatus,
								},
							},
						};
					});
					break;

				case 'swarm_turn':
					update(s => {
						const agent = data.agent as AgentName;
						const turn: AgentTurn = {
							agent,
							tokens_in: data.tokens_in,
							tokens_out: data.tokens_out,
							latency_ms: data.latency_ms,
							reasoning: data.reasoning || [],
							preview: data.preview,
							parsed: data.parsed || {},
							timestamp: data.timestamp,
						};

						return {
							...s,
							currentAgent: agent,
							agents: {
								...s.agents,
								[agent]: {
									status: 'done' as AgentStatus,
									tokens_in: data.tokens_in,
									tokens_out: data.tokens_out,
									latency_ms: data.latency_ms,
									reasoning: data.reasoning || [],
									preview: data.preview,
								},
							},
							turns: [...s.turns, turn],
							totalTokensIn: s.totalTokensIn + data.tokens_in,
							totalTokensOut: s.totalTokensOut + data.tokens_out,
						};
					});
					break;

				case 'swarm_failure':
					update(s => {
						const agent = data.agent_blamed as AgentName;
						return {
							...s,
							failures: [...s.failures, {
								wave: data.wave,
								failure_type: data.failure_type,
								agent_blamed: data.agent_blamed,
								root_cause: data.root_cause,
								recommendation: data.recommendation,
								timestamp: data.timestamp,
							}],
							agents: {
								...s.agents,
								[agent]: {
									...s.agents[agent],
									status: 'failed' as AgentStatus,
								},
							},
						};
					});
					break;

				case 'swarm_file_written':
					update(s => ({
						...s,
						fileWrites: [...s.fileWrites, {
							wave: data.wave,
							file_path: data.file_path,
							operation: data.operation,
							lines_added: data.lines_added,
							preview: data.preview,
							timestamp: data.timestamp,
						}],
					}));
					break;

				case 'swarm_diagnostic':
					// CONFIG entered diagnostic mode - log for visibility
					console.log('[Swarm] Diagnostic:', data.failure_type, data.fix_strategy);
					update(s => ({
						...s,
						error: `Diagnosing: ${data.failure_type} (${Math.round(data.confidence * 100)}% conf)`,
					}));
					break;

				case 'swarm_consultation':
					// Agent consultation result
					console.log('[Swarm] Consultation:', data.agent, data.recommended_action);
					break;

				case 'pong':
					// Heartbeat response
					break;

				case 'error':
					update(s => ({ ...s, error: data.message }));
					break;
			}
		} catch (e) {
			console.error('[Swarm] Failed to parse message:', e);
		}
	}

	return {
		subscribe,

		connect() {
			if (ws && ws.readyState === WebSocket.OPEN) {
				return;
			}

			try {
				ws = new WebSocket('ws://localhost:8000/ws/swarm');

				ws.onopen = () => {
					reconnectAttempts = 0;
					update(s => ({ ...s, connected: true, error: null }));
					console.log('[Swarm] Connected');
				};

				ws.onmessage = handleMessage;

				ws.onerror = (error) => {
					console.error('[Swarm] WebSocket error:', error);
					update(s => ({ ...s, error: 'Connection error' }));
				};

				ws.onclose = () => {
					update(s => ({ ...s, connected: false }));
					console.log('[Swarm] Disconnected');

					// Auto-reconnect
					if (reconnectAttempts < maxReconnectAttempts) {
						reconnectAttempts++;
						const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
						console.log(`[Swarm] Reconnecting in ${delay}ms...`);
						reconnectTimer = setTimeout(() => this.connect(), delay);
					}
				};
			} catch (e) {
				console.error('[Swarm] Failed to connect:', e);
				update(s => ({ ...s, error: `Failed to connect: ${e}` }));
			}
		},

		disconnect() {
			if (reconnectTimer) {
				clearTimeout(reconnectTimer);
				reconnectTimer = null;
			}
			if (ws) {
				ws.close();
				ws = null;
			}
			update(s => ({ ...s, connected: false }));
		},

		send(data: unknown) {
			if (ws && ws.readyState === WebSocket.OPEN) {
				ws.send(JSON.stringify(data));
			}
		},

		requestStatus() {
			this.send({ type: 'status' });
		},

		reset() {
			set({ ...initialState });
		},
	};
}

export const swarm = createSwarmStore();

// Derived stores for specific data
export const currentAgentTurn = derived(swarm, $s =>
	$s.currentAgent ? $s.agents[$s.currentAgent] : null
);

export const waveProgress = derived(swarm, $s => {
	const completed = AGENT_ORDER.filter(a => $s.agents[a].status === 'done').length;
	return {
		current: $s.currentAgent,
		completed,
		total: AGENT_ORDER.length,
		percentage: (completed / AGENT_ORDER.length) * 100,
	};
});

export const latestFailure = derived(swarm, $s =>
	$s.failures.length > 0 ? $s.failures[$s.failures.length - 1] : null
);

export const latestFileWrite = derived(swarm, $s =>
	$s.fileWrites.length > 0 ? $s.fileWrites[$s.fileWrites.length - 1] : null
);
