import { useState } from 'react';
import {
  ExportRefused,
  browserDeps,
  exportGif,
  exportPng,
  exportSvg,
  exportWebm,
  type ExportDeps,
  type ExportScene,
  type ExportResult,
} from './sceneExport';

/*
 * The controls that export the scene (an SVG or a PNG of the frame on screen, a WebM or a GIF of the replay). They exist only behind the one switch
 * for Waymo-derived outputs (waymoDerivedOutputs.ts): with `enabled` anything but true this renders nothing at all, and ScenePanel does not even load this
 * module then. What each button does is in sceneExport.ts: the file carries the title, the recorded-versus-perturbed caveat and the
 * attribution as text, and a request over the limits there is refused with a one-line message, which is shown here and nothing else is done.
 */

export interface SceneExportControlsProps {
  enabled: boolean;
  scene: ExportScene;
  /** The frame on screen; null when none is drawn. */
  frame: number | null;
  range: { first: number; last: number };
  deps?: ExportDeps;
  width?: number;
}

type Job = 'svg' | 'png' | 'webm' | 'gif';

export function SceneExportControls({ enabled, scene, frame, range, deps, width }: SceneExportControlsProps) {
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState<Job | null>(null);
  if (enabled !== true) return null;

  async function run(job: Job) {
    setBusy(job);
    setStatus(null);
    const common = { enabled, scene, width, deps: deps ?? browserDeps() };
    try {
      let result: ExportResult;
      if (job === 'svg' || job === 'png') {
        if (frame === null) throw new ExportRefused('Export refused: no frame is drawn.');
        result = job === 'svg' ? await exportSvg({ ...common, frame }) : await exportPng({ ...common, frame });
      } else {
        const replay = { ...common, first: range.first, last: range.last };
        result = job === 'webm' ? await exportWebm(replay) : await exportGif(replay);
      }
      setStatus(`Saved ${result.filename}.`);
    } catch (e) {
      setStatus(e instanceof ExportRefused ? e.message : `Export failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(null);
    }
  }

  const disabled = busy !== null;
  return (
    <div className="scene-export" role="group" aria-label="Export the scene">
      <button type="button" disabled={disabled || frame === null} onClick={() => void run('svg')}>Export frame as SVG</button>
      <button type="button" disabled={disabled || frame === null} onClick={() => void run('png')}>Export frame as PNG</button>
      <button type="button" disabled={disabled} onClick={() => void run('webm')}>Record replay as WebM</button>
      <button type="button" disabled={disabled} onClick={() => void run('gif')}>Make GIF of the replay</button>
      {status !== null && <p className="note" role="status">{status}</p>}
    </div>
  );
}
