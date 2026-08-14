import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Miroir du nginx.conf du conteneur frontend en prod : le code appelle
    // toujours des chemins relatifs (/api/...), donc aucune divergence entre
    // dev et prod. Sans Docker, le backend tourne sur localhost:8000.
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
