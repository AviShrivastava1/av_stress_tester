import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { UseQueryResult } from '@tanstack/react-query';
import type { PerturbedResponse, TrajectoryResponse } from '../api/client';
import { makeDetail, makePerturbed, makeTrack, makeTrajectories, range } from '../test/fixtures';
import { ScenePanel } from './detail/ScenePanel';

/*
 * The scene panel with the switch for Waymo-derived outputs passed in as ON (the file stays false): the controls are wired to what is on screen. The browser's
 * own pieces (a download link, an image, a canvas, a recorder) are stood in for at the global level, because the panel uses the real browser deps.
 */

const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 5), origin: [-20, 0] });
const challenger = makeTrack({ agent_idx: 1, frames: range(0, 5), origin: [0, -10], heading: Math.PI / 2 });
const perturbed = makeTrack({ agent_idx: 1, frames: range(0, 5), origin: [0, -10], heading: Math.PI / 2 + 0.1 });

function query<T>(data: T): UseQueryResult<T> {
  return { data, isPending: false, isError: false, isFetching: false, error: null, refetch: async () => ({}) } as unknown as UseQueryResult<T>;
}

function panel() {
  const trajectories = query<TrajectoryResponse>(makeTrajectories([sdc, challenger], 'abcdef0123456789'));
  const pert = query<PerturbedResponse>(makePerturbed({ target_idx: 1, collision_timestep: 3, baseline: challenger, perturbed }));
  return <ScenePanel trajectories={trajectories} perturbed={pert} detail={makeDetail({ scenario_id: 'abcdef0123456789' })} exportEnabled />;
}

function stubDownload() {
  const blobs: Blob[] = [];
  const links: { download: string; href: string }[] = [];
  vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: (b: Blob) => (blobs.push(b), 'blob:fake'), revokeObjectURL: () => {} }));
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    links.push({ download: this.download, href: this.href });
  });
  return { blobs, links };
}

const read = (blob: Blob): Promise<string> =>
  new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.readAsText(blob);
  });

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('the export controls in the scene panel, with the switch passed as on', () => {
  it('export the frame on screen as an SVG named for the scenario and the frame, with the scene, the caveat and the attribution in it', async () => {
    const { blobs, links } = stubDownload();
    render(panel());
    await userEvent.click(await screen.findByRole('button', { name: 'Export frame as SVG' }));
    await waitFor(() => expect(links).toHaveLength(1));
    expect(links[0]).toEqual({ download: 'abcdef0123456789-frame-3.svg', href: 'blob:fake' });      // the panel opens on the collision frame, 3
    const svg = await read(blobs[0]!);
    expect(svg).toContain('role-challenger_perturbed');
    expect(svg.match(/class="footprint [^"]*\bcolliding\b/g)).toHaveLength(2);                // the colliding pair is marked at the collision frame
    expect(svg).toContain('scenario abcdef0123456789 · frame 3');
    expect(svg).toContain('RECORDED VS PERTURBED, FRAME BY FRAME');
    expect(svg).toContain('This software was made using the Waymo Open Dataset');
    expect(screen.getByRole('status')).toHaveTextContent('Saved abcdef0123456789-frame-3.svg.');
  });

  it('follow the frame the viewer moved to', async () => {
    const { links } = stubDownload();
    render(panel());
    await screen.findByRole('button', { name: 'Export frame as SVG' });
    await userEvent.click(screen.getByRole('button', { name: 'Step back one frame' }));
    await userEvent.click(screen.getByRole('button', { name: 'Export frame as SVG' }));
    await waitFor(() => expect(links).toHaveLength(1));
    expect(links[0]!.download).toBe('abcdef0123456789-frame-2.svg');
    await userEvent.click(screen.getByRole('button', { name: 'Step forward one frame' }));
    await userEvent.click(screen.getByRole('button', { name: 'Step forward one frame' }));
    await userEvent.click(screen.getByRole('button', { name: 'Export frame as SVG' }));
    await waitFor(() => expect(links).toHaveLength(2));
    expect(links[1]!.download).toBe('abcdef0123456789-frame-4.svg');
  });

  it('carry the framing the viewer chose: the note about the other agents only when they are in the view', async () => {
    const { blobs, links } = stubDownload();
    const bystander = makeTrack({ agent_idx: 2, agent_type: 1, frames: range(0, 5), origin: [20, 20] });
    const trajectories = query<TrajectoryResponse>(makeTrajectories([sdc, challenger, bystander], 'abcdef0123456789'));
    const pert = query<PerturbedResponse>(makePerturbed({ target_idx: 1, collision_timestep: 3, baseline: challenger, perturbed }));
    render(<ScenePanel trajectories={trajectories} perturbed={pert} detail={makeDetail({ scenario_id: 'abcdef0123456789' })} exportEnabled />);
    await userEvent.click(await screen.findByRole('button', { name: 'Export frame as SVG' }));
    await waitFor(() => expect(links).toHaveLength(1));
    expect(await read(blobs[0]!)).not.toContain('drawn as logged');                   // the view opens on the challenger and the SDC
    await userEvent.click(screen.getByRole('button', { name: 'Whole scene' }));
    await userEvent.click(screen.getByRole('button', { name: 'Export frame as SVG' }));
    await waitFor(() => expect(links).toHaveLength(2));
    expect(await read(blobs[1]!)).toContain('drawn as logged');
    expect(await read(blobs[1]!)).toContain('role-other');
  });

  it('record the replay over the whole range of frames as a WebM', async () => {
    const { links } = stubDownload();
    class FakeImage {
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      set src(_v: string) { queueMicrotask(() => this.onload?.()); }
    }
    const ctx = { fillStyle: '', font: '', textBaseline: 'top', fillRect() {}, drawImage() {}, fillText() {}, getImageData: () => ({ data: new Uint8ClampedArray(4) }) };
    vi.stubGlobal('Image', FakeImage);
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
    (HTMLCanvasElement.prototype as unknown as { captureStream: () => MediaStream }).captureStream = () => ({}) as MediaStream;
    class FakeRecorder {
      ondataavailable: ((e: { data: Blob }) => void) | null = null;
      onstop: (() => void) | null = null;
      static isTypeSupported = () => true;
      start() {}
      stop() { this.ondataavailable?.({ data: new Blob(['v']) }); this.onstop?.(); }
    }
    vi.stubGlobal('MediaRecorder', FakeRecorder);
    render(panel());
    await userEvent.click(await screen.findByRole('button', { name: 'Record replay as WebM' }));
    await waitFor(() => expect(links).toHaveLength(1), { timeout: 5000 });
    expect(links[0]!.download).toBe('abcdef0123456789-frames-0-4.webm');
    delete (HTMLCanvasElement.prototype as unknown as { captureStream?: unknown }).captureStream;
  });
});
