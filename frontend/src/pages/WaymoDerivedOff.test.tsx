import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { UseQueryResult } from '@tanstack/react-query';
import type { PerturbedResponse, TrajectoryResponse } from '../api/client';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { stubFetch, type RecordedRequest, type Reply } from '../test/fetchStub';
import { renderApp } from '../test/renderApp';
import { ScenePanel } from './detail/ScenePanel';
import { WAYMO_DERIVED_OUTPUTS_ENABLED } from '../waymoDerivedOutputs';

/*
 * The switch for Waymo-derived outputs is OFF in the repository (waymo_derived_outputs.json). With it off, the scenario page draws its scene and offers
 * no export of it anywhere: no control, no group, no way to reach the module that produces an image, an SVG, a GIF or a video. A ScenePanel handed
 * `exportEnabled` shows the controls, so the test above is not vacuous. Nothing here edits the file.
 */

const ID = 'synthetic-0001';
const ok = (body: unknown): Reply => ({ status: 200, body });
const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 31), origin: [-20, 0] });
const challenger = makeTrack({ agent_idx: 1, frames: range(0, 31), origin: [0, -10], heading: Math.PI / 2 });
const perturbed = makeTrack({ agent_idx: 1, frames: range(0, 31), origin: [0, -10], heading: Math.PI / 2 + 0.1 });

const EXPORT_WORDS = /export|download|record replay|\bgif\b|webm|\bpng\b|\bsvg\b|save (image|video)/i;

function query<T>(data: T): UseQueryResult<T> {
  return { data, isPending: false, isError: false, isFetching: false, error: null, refetch: async () => ({}) } as unknown as UseQueryResult<T>;
}

describe('the scenario page with the switch off', () => {
  it('is off in the repository', () => {
    expect(WAYMO_DERIVED_OUTPUTS_ENABLED).toBe(false);
  });

  it('draws the scene and offers no export of it', async () => {
    stubFetch((req: RecordedRequest): Reply => {
      if (req.path.endsWith('/trajectories')) return ok(makeTrajectories([sdc, challenger]));
      if (req.path.endsWith('/perturbed')) return ok(makePerturbed({ target_idx: 1, collision_timestep: 12, baseline: challenger, perturbed }));
      if (req.path.endsWith('/provenance')) return ok({ scenario_id: ID, stress_run_id: 'r' });
      return ok(makeDetail({ scenario_id: ID, stress_tested: true, stress_outcome: 'collision_found', min_perturbation: 0.1 }));
    });
    renderApp(`/scenarios/${ID}`);
    await screen.findByRole('img', { name: /Scene paths/ });
    expect(screen.queryByRole('group', { name: 'Export the scene' })).toBeNull();
    for (const button of screen.queryAllByRole('button')) expect(button.textContent, button.outerHTML).not.toMatch(EXPORT_WORDS);
    expect(document.body.textContent).not.toMatch(/Export frame as|Record replay as|Make GIF/);
    expect(document.querySelector('.scene-export')).toBeNull();
    expect(document.querySelector('a[download]')).toBeNull();
  });

  it('shows the controls to a ScenePanel that is told the switch is on, which is what makes the test above mean something', async () => {
    const trajectories = query<TrajectoryResponse>(makeTrajectories([sdc, challenger]));
    const pert = query<PerturbedResponse>(makePerturbed({ target_idx: 1, collision_timestep: 12, baseline: challenger, perturbed }));
    render(<ScenePanel trajectories={trajectories} perturbed={pert} detail={makeDetail({ scenario_id: ID })} exportEnabled />);
    expect(await screen.findByRole('group', { name: 'Export the scene' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Export frame as SVG' })).toBeEnabled();
  });

  it('does not show them to a ScenePanel that is told it is off', async () => {
    const trajectories = query<TrajectoryResponse>(makeTrajectories([sdc, challenger]));
    const pert = query<PerturbedResponse>(makePerturbed({ target_idx: 1, collision_timestep: 12, baseline: challenger, perturbed }));
    render(<ScenePanel trajectories={trajectories} perturbed={pert} detail={makeDetail({ scenario_id: ID })} exportEnabled={false} />);
    await screen.findByRole('img', { name: /Scene paths/ });
    expect(screen.queryByRole('group', { name: 'Export the scene' })).toBeNull();
  });
});

describe('the code path to an exporter', () => {
  const nodeProcess = (globalThis as unknown as { process: { cwd(): string; getBuiltinModule(id: 'node:fs'): { readdirSync(p: string, o: { recursive: true }): string[]; readFileSync(p: string, e: 'utf8'): string } } }).process;
  const fs = nodeProcess.getBuiltinModule('node:fs');
  const files = fs.readdirSync(`${nodeProcess.cwd()}/src`, { recursive: true }).filter((f) => /\.(ts|tsx)$/.test(f) && !/\.test\.(ts|tsx)$/.test(f) && !f.endsWith('.d.ts'));

  it('is imported by nothing but a dynamic import in ScenePanel, behind the switch', () => {
    const importers = files.filter((f) => !f.startsWith('scene/export/') && /(from|import\()\s*'[^']*scene\/export\//.test(fs.readFileSync(`${nodeProcess.cwd()}/src/${f}`, 'utf8')));
    expect(importers).toEqual(['pages/detail/ScenePanel.tsx']);
    const panel = fs.readFileSync(`${nodeProcess.cwd()}/src/pages/detail/ScenePanel.tsx`, 'utf8');
    expect(panel).toMatch(/lazy\(\(\) =>\s*import\('..\/..\/scene\/export\/SceneExportControls'\)/);
    expect(panel).not.toMatch(/from '..\/..\/scene\/export/);
    expect(panel).toContain('exportEnabled === true &&');
    expect(panel).toContain('exportEnabled = WAYMO_DERIVED_OUTPUTS_ENABLED');
  });

  it('has no other place that makes an image, a canvas, a recorder or a download link', () => {
    const makers = /createElement\(\s*'canvas'|toBlob\(|toDataURL\(|new MediaRecorder|captureStream\(|URL\.createObjectURL|\.download\s*=|new XMLSerializer/;
    const offenders = files.filter((f) => !f.startsWith('scene/export/') && makers.test(fs.readFileSync(`${nodeProcess.cwd()}/src/${f}`, 'utf8')));
    expect(offenders).toEqual([]);
  });
});
