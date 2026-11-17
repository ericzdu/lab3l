import { defineConfig, type Plugin, type ViteDevServer, type PreviewServer } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'node:path';
import fs from 'node:fs';
import { execFile } from 'node:child_process';
import type { IncomingMessage, ServerResponse } from 'node:http';

const repo = path.resolve(process.env.LAB3L_REPO || path.join(__dirname, '..'));
const python = process.env.LAB3L_PYTHON || path.join(repo, 'shadow-hand-teleop', '.venv',
  process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');

function repositoryData(): Plugin {
  const install = (server: ViteDevServer | PreviewServer) => {
    server.middlewares.use('/api/repository', (req: IncomingMessage, res: ServerResponse) => {
      res.setHeader('Cache-Control', 'no-store');
      if (req.method !== 'GET') { res.statusCode = 405; res.setHeader('Allow', 'GET'); res.end(); return; }
      const url = new URL(req.url || '/', 'http://localhost');
      if (url.pathname === '/file') {
        const name = url.searchParams.get('path');
        const directory = path.join(repo, 'shadow-hand-teleop', 'data');
        try {
          if (!name) throw new Error('Missing file');
          const real = fs.realpathSync(path.resolve(directory, name));
          const relative = path.relative(fs.realpathSync(directory), real);
          if (!relative || relative.startsWith('..') || path.isAbsolute(relative) ||
              !['.json', '.h5', '.hdf5', '.npz'].includes(path.extname(real).toLowerCase()) ||
              !fs.statSync(real).isFile()) throw new Error('Invalid file');
          res.setHeader('Content-Type', 'application/octet-stream');
          res.setHeader('Content-Disposition', `attachment; filename="${path.basename(real).replace(/[^a-zA-Z0-9_.-]/g, '_')}"`);
          const stream = fs.createReadStream(real);
          stream.on('error', () => { if (!res.headersSent) res.statusCode = 404; res.end('Dataset unavailable'); });
          stream.pipe(res);
        } catch { res.statusCode = 404; res.end('Dataset file not found'); }
        return;
      }
      if (url.pathname !== '/' && url.pathname !== '') { res.statusCode = 404; res.end(); return; }
      execFile(python, [path.join(__dirname, 'scripts', 'read_repository_data.py')], {
        env: { ...process.env, LAB3L_REPO: repo, PYTHONDONTWRITEBYTECODE: '1' },
        timeout: 20000, maxBuffer: 5 * 1024 * 1024,
      }, (error, stdout, stderr) => {
        res.setHeader('Content-Type', 'application/json');
        if (error) { res.statusCode = 503; res.end(JSON.stringify({ error: stderr.trim() || error.message })); return; }
        res.end(stdout);
      });
    });
  };
  return { name: 'lab3l-local-data', configureServer: install, configurePreviewServer: install };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), repositoryData()],
  resolve: { alias: { '@': path.resolve(__dirname, 'src') } },
  server: { host: '127.0.0.1', port: 5173, strictPort: true },
  preview: { host: '127.0.0.1' },
});
