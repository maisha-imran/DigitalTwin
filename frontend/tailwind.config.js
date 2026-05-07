/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        cyber: {
          bg: '#030712',
          panel: '#0a0f1e',
          border: '#0f2040',
          accent: '#00d4ff',
          green: '#00ff9d',
          yellow: '#ffd700',
          orange: '#ff8c00',
          red: '#ff2d55',
          purple: '#bf5af2',
        },
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'monospace'],
        display: ['"Orbitron"', 'sans-serif'],
        body: ['"Exo 2"', 'sans-serif'],
      },
      boxShadow: {
        neon: '0 0 20px rgba(0,212,255,0.3)',
        'neon-green': '0 0 20px rgba(0,255,157,0.3)',
        'neon-red': '0 0 20px rgba(255,45,85,0.3)',
      },
    },
  },
  plugins: [],
}
