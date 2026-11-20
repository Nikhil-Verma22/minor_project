import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import path from 'path';
import fs from 'fs';

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.{wasm,mjs}',
          dest: 'ort-dist',
        },
      ],
    }),
    {
      name: 'serve-ort-dist',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url && req.url.startsWith('/ort-dist/')) {
            const cleanUrl = req.url.replace(/^\/ort-dist\//, '').split('?')[0];
            const distDir = path.resolve(import.meta.dirname, 'node_modules/onnxruntime-web/dist');
            const filePath = path.resolve(distDir, cleanUrl);
            if (fs.existsSync(filePath)) {
              if (cleanUrl.endsWith('.wasm')) {
                res.setHeader('Content-Type', 'application/wasm');
              } else if (cleanUrl.endsWith('.mjs') || cleanUrl.endsWith('.js')) {
                res.setHeader('Content-Type', 'application/javascript');
              }
              res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
              res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
              return fs.createReadStream(filePath).pipe(res);
            }
          }
          next();
        });
      },
    },
  ],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
