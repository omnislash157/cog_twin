import { writable } from 'svelte/store';
import { artifacts } from './artifacts';

interface WebSocketState {
	connected: boolean;
	sessionId: string | null;
	error: string | null;
}

function createWebSocketStore() {
	const { subscribe, set, update } = writable<WebSocketState>({
		connected: false,
		sessionId: null,
		error: null,
	});

	let ws: WebSocket | null = null;
	let messageHandlers: ((data: any) => void)[] = [];
	let reconnectAttempts = 0;
	const maxReconnectAttempts = 5;

	return {
		subscribe,

		connect(sessionId: string) {
			const url = `ws://localhost:8000/ws/${sessionId}`;
			
			try {
				ws = new WebSocket(url);

				ws.onopen = () => {
					reconnectAttempts = 0;
					update(s => ({ ...s, connected: true, sessionId, error: null }));
					console.log(`[WS] Connected to session ${sessionId}`);
				};

				ws.onmessage = (event) => {
					try {
						const data = JSON.parse(event.data);
						messageHandlers.forEach(handler => handler(data));

					// Auto-handle artifact emissions
					if (data.type === 'artifact_emit' && data.artifact) {
						artifacts.add(data.artifact, data.suggested || false);
					}
					} catch (e) {
						console.error('[WS] Failed to parse message:', e);
					}
				};

				ws.onerror = (error) => {
					console.error('[WS] Error:', error);
					update(s => ({ ...s, error: 'Connection error' }));
				};

				ws.onclose = () => {
					update(s => ({ ...s, connected: false }));
					console.log('[WS] Disconnected');
				};
			} catch (e) {
				update(s => ({ ...s, error: `Failed to connect: ${e}` }));
			}
		},

		disconnect() {
			if (ws) {
				ws.close();
				ws = null;
			}
			update(s => ({ ...s, connected: false, sessionId: null }));
		},

		send(data: any) {
			if (ws && ws.readyState === WebSocket.OPEN) {
				ws.send(JSON.stringify(data));
			} else {
				console.warn('[WS] Cannot send - not connected');
			}
		},

		onMessage(handler: (data: any) => void) {
			messageHandlers.push(handler);
			return () => {
				messageHandlers = messageHandlers.filter(h => h !== handler);
			};
		},

		async commitAndClose() {
			if (ws && ws.readyState === WebSocket.OPEN) {
				ws.send(JSON.stringify({ type: 'commit' }));
				await new Promise(resolve => setTimeout(resolve, 500));
				ws.close();
			}
		}
	};
}

export const websocket = createWebSocketStore();
