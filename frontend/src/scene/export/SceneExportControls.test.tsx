import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { makeTrack, range } from '../../test/fixtures';
import { buildScene } from '../sceneModel';
import { SceneExportControls } from './SceneExportControls';
import type { CanvasPort, ExportDeps, ExportScene } from './sceneExport';

/*
 * The export controls (SceneExportControls.tsx) with the switch passed in. Off: nothing at all in the DOM. On: four buttons, and each one calls the
 * exporter with what is on screen; a refusal is shown as its one line and nothing is saved.
 */

const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 21), origin: [-20, 0] });
const challenger = makeTrack({ agent_idx: 1, frames: range(0, 21), origin: [0, -10], heading: Math.PI / 2 });
const scene: ExportScene = (() => {
  const drawn = buildScene([sdc, challenger], undefined);
  return { scenarioId: 'abc123def4567890', drawn, focus: drawn, collisionFrame: null };
})();

function deps(overrides: Partial<ExportDeps> = {}): { deps: ExportDeps; saved: string[] } {
  const saved: string[] = [];
  const canvas: CanvasPort = {
    width: 0, height: 0,
    getContext: () => ({ fillStyle: '', font: '', textBaseline: 'top', fillRect() {}, drawImage() {}, fillText() {}, getImageData: (_x, _y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }) }),
    toBlob: (cb, type) => cb(new Blob(['x'], { type })),
  };
  return {
    saved,
    deps: {
      loadImage: async () => ({}) as CanvasImageSource,
      createCanvas: () => canvas,
      save: (_blob, filename) => void saved.push(filename),
      MediaRecorder: undefined,
      sleep: async () => {},
      ...overrides,
    },
  };
}

const props = { scene, frame: 7, range: { first: 0, last: 20 } };

describe('with the switch off', () => {
  it.each([false, undefined, null, 1, 'true'])('renders nothing at all for %s', (value) => {
    const { container } = render(<SceneExportControls enabled={value as unknown as boolean} {...props} deps={deps().deps} />);
    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByRole('button')).toBeNull();
  });
});

describe('with the switch on', () => {
  it('offers the four exports as buttons in a group of their own', () => {
    render(<SceneExportControls enabled {...props} deps={deps().deps} />);
    const group = screen.getByRole('group', { name: 'Export the scene' });
    expect([...group.querySelectorAll('button')].map((b) => b.textContent)).toEqual([
      'Export frame as SVG', 'Export frame as PNG', 'Record replay as WebM', 'Make GIF of the replay',
    ]);
  });

  it('exports the frame on screen as an SVG and says what was saved', async () => {
    const d = deps();
    render(<SceneExportControls enabled {...props} deps={d.deps} />);
    await userEvent.click(screen.getByRole('button', { name: 'Export frame as SVG' }));
    expect(await screen.findByRole('status')).toHaveTextContent('Saved abc123def4567890-frame-7.svg.');
    expect(d.saved).toEqual(['abc123def4567890-frame-7.svg']);
  });

  it('exports the frame as a PNG, and the replay as a GIF over the whole range', async () => {
    const d = deps();
    render(<SceneExportControls enabled {...props} width={320} deps={d.deps} />);
    await userEvent.click(screen.getByRole('button', { name: 'Export frame as PNG' }));
    await waitFor(() => expect(d.saved).toEqual(['abc123def4567890-frame-7.png']));
    await userEvent.click(screen.getByRole('button', { name: 'Make GIF of the replay' }));
    await waitFor(() => expect(d.saved).toEqual(['abc123def4567890-frame-7.png', 'abc123def4567890-frames-0-20.gif']));
    expect(screen.getByRole('status')).toHaveTextContent('Saved abc123def4567890-frames-0-20.gif.');
  });

  it('shows a refusal as its one line and saves nothing', async () => {
    const d = deps();
    render(<SceneExportControls enabled {...props} width={1281} deps={d.deps} />);
    await userEvent.click(screen.getByRole('button', { name: 'Export frame as SVG' }));
    const status = await screen.findByRole('status');
    expect(status.textContent).toMatch(/^Export refused: 1 frames at 1281 by \d+ pixels is over the limit of 120 frames and 1280 by 1080 pixels\.$/);
    expect(d.saved).toEqual([]);
  });

  it('says a browser with no WebM recording cannot, and saves nothing', async () => {
    const d = deps();
    render(<SceneExportControls enabled {...props} width={320} deps={d.deps} />);
    await userEvent.click(screen.getByRole('button', { name: 'Record replay as WebM' }));
    expect(await screen.findByRole('status')).toHaveTextContent('Export refused: this browser cannot record WebM video.');
    expect(d.saved).toEqual([]);
  });

  it('turns the frame exports off when no frame is drawn, and leaves the replay exports on', () => {
    render(<SceneExportControls enabled {...props} frame={null} deps={deps().deps} />);
    expect(screen.getByRole('button', { name: 'Export frame as SVG' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Export frame as PNG' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Record replay as WebM' })).toBeEnabled();
  });

  it('turns every button off while a job runs and back on when it ends', async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const d = deps({ loadImage: async () => { await gate; return {} as CanvasImageSource; } });
    render(<SceneExportControls enabled {...props} width={320} deps={d.deps} />);
    await userEvent.click(screen.getByRole('button', { name: 'Export frame as PNG' }));
    for (const b of screen.getAllByRole('button')) expect(b).toBeDisabled();
    release();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Record replay as WebM' })).toBeEnabled());
    expect(screen.getByRole('button', { name: 'Export frame as SVG' })).toBeEnabled();
  });

  it('takes the last message away when the next job starts', async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const d = deps({ loadImage: async () => { await gate; return {} as CanvasImageSource; } });
    render(<SceneExportControls enabled {...props} width={320} deps={d.deps} />);
    await userEvent.click(screen.getByRole('button', { name: 'Record replay as WebM' }));
    expect(await screen.findByRole('status')).toHaveTextContent('cannot record WebM');
    await userEvent.click(screen.getByRole('button', { name: 'Export frame as PNG' }));
    expect(screen.queryByRole('status')).toBeNull();
    release();
    expect(await screen.findByRole('status')).toHaveTextContent('Saved abc123def4567890-frame-7.png.');
  });

  it('reports a failure that is not a refusal, without calling it one', async () => {
    const d = deps({ save: () => { throw new Error('disk full'); } });
    render(<SceneExportControls enabled {...props} deps={d.deps} />);
    await userEvent.click(screen.getByRole('button', { name: 'Export frame as SVG' }));
    expect(await screen.findByRole('status')).toHaveTextContent('Export failed: disk full');
  });

  it('uses none of the words the copy lint forbids in its labels', () => {
    render(<SceneExportControls enabled {...props} deps={deps().deps} />);
    const text = screen.getByRole('group').textContent!;
    for (const re of [/\blive\b/i, /\bsafety\b/i, /\bend-to-end\b/i, /\badversarial\b/i, /\bsafe\b/i]) expect(text).not.toMatch(re);
  });
});
