import { describe, expect, it } from 'vitest';
import switchFile from '../../waymo_derived_outputs.json';
import { WAYMO_DERIVED_OUTPUTS_ENABLED } from './waymoDerivedOutputs';
import { readFrontendFile } from './test/readFrontendFile';

/*
 * The one switch for Waymo-derived outputs (waymo_derived_outputs.json at the repository root; the Python side is
 * tests/test_waymo_derived_switch.py). The Waymo licence reply is outstanding, so the value must be false: if it is ever true, this
 * file's first two tests fail. The file is exactly one boolean, the constant reads only that file, and nothing else reads it.
 */

describe('the Waymo-derived outputs switch', () => {
  it('is off: if this fails, the switch file says true and the licence reply has not been settled', () => {
    expect(switchFile.enabled).toBe(false);
    expect(WAYMO_DERIVED_OUTPUTS_ENABLED).toBe(false);
  });

  it('is exactly one boolean called enabled, and nothing else', () => {
    expect(JSON.parse(readFrontendFile('../waymo_derived_outputs.json'))).toEqual({ enabled: false });
    expect(Object.keys(switchFile)).toEqual(['enabled']);
  });

  it('is the value of the file and of nothing else: the constant is the file’s boolean true, and only that', () => {
    expect(WAYMO_DERIVED_OUTPUTS_ENABLED).toBe((switchFile as { enabled: unknown }).enabled === true);
    const source = readFrontendFile('src/waymoDerivedOutputs.ts');
    expect(source).toContain("(file as { enabled?: unknown }).enabled === true");
  });

  it('can be read by the dev server, which serves files from one directory above the project only when told to', () => {
    expect(readFrontendFile('vite.config.ts')).toContain("fs: { allow: ['..'] }");
  });

  it('is read in one module only, and there is no second constant for it in the app', () => {
    const nodeProcess = (globalThis as unknown as {
      process: { cwd(): string; getBuiltinModule(id: 'node:fs'): { readdirSync(p: string, o: { recursive: true }): string[] } };
    }).process;
    const files = nodeProcess
      .getBuiltinModule('node:fs')
      .readdirSync(`${nodeProcess.cwd()}/src`, { recursive: true })
      .filter((f) => /\.(ts|tsx)$/.test(f) && !/\.test\.(ts|tsx)$/.test(f) && !f.endsWith('.d.ts') && f !== 'waymoDerivedOutputs.ts');
    expect(files.length).toBeGreaterThan(50);
    for (const f of files) {
      const text = readFrontendFile(`src/${f}`);
      expect(text, f).not.toContain('waymo_derived_outputs.json');
      expect(text, f).not.toMatch(/(const|let|var)\s+\w*(DERIVED|WAYMO)\w*(ENABLED|SWITCH)\w*\s*(:\s*\w+\s*)?=\s*(true|false)/i);
    }
  });
});
