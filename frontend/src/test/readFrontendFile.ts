/**
 * The text of a file under frontend/, read from disk, for the few tests that check a file as text (the stylesheet,
 * index.html). vitest hands a `?raw` import of a .css file an empty string, and the project has no @types/node, so
 * Node's fs and cwd come from the process global (`getBuiltinModule` exists from Node 22.3; package.json "engines"
 * and frontend/.nvmrc require 22). vitest's root, and so its cwd, is frontend/: `npm test` and CI both run it there.
 */
const nodeProcess = (globalThis as unknown as {
  process: { cwd(): string; getBuiltinModule(id: 'node:fs'): { readFileSync(path: string, encoding: 'utf8'): string } };
}).process;

export function readFrontendFile(relativePath: string): string {
  return nodeProcess.getBuiltinModule('node:fs').readFileSync(`${nodeProcess.cwd()}/${relativePath}`, 'utf8');
}
