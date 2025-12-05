/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{html,js,svelte,ts}'],
	theme: {
		extend: {
			colors: {
				// Cyberpunk palette
				'neon-green': '#00ff41',
				'neon-cyan': '#00ffff',
				'neon-magenta': '#ff00ff',
				'neon-yellow': '#ffff00',
				'neon-red': '#ff0040',
				// Backgrounds
				'bg-primary': '#0a0a0a',
				'bg-secondary': '#111111',
				'bg-tertiary': '#1a1a1a',
				// Borders
				'border-dim': '#222222',
				'border-glow': '#00ff4140',
				// Text
				'text-primary': '#e0e0e0',
				'text-muted': '#808080',
				// Normie mode
				'normie-bg': '#ffffff',
				'normie-text': '#1a1a1a',
				'normie-border': '#e0e0e0',
			},
			fontFamily: {
				'mono': ['JetBrains Mono', 'Fira Code', 'monospace'],
				'sans': ['Inter', 'system-ui', 'sans-serif'],
			},
			boxShadow: {
				'neon': '0 0 10px #00ff41, 0 0 20px #00ff4180',
				'neon-cyan': '0 0 10px #00ffff, 0 0 20px #00ffff80',
				'neon-magenta': '0 0 10px #ff00ff, 0 0 20px #ff00ff80',
			},
			animation: {
				'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
				'scan-line': 'scan-line 4s linear infinite',
			},
			keyframes: {
				'pulse-glow': {
					'0%, 100%': { opacity: '0.6' },
					'50%': { opacity: '1' },
				},
				'scan-line': {
					'0%': { transform: 'translateY(-100%)' },
					'100%': { transform: 'translateY(100%)' },
				},
			},
		},
	},
	plugins: [],
};
