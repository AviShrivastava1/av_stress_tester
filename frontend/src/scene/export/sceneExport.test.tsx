import { afterEach, describe, expect, it, vi } from 'vitest';
import { WAYMO_ATTRIBUTION } from '../../components/Footer';
import { makeTrack, range } from '../../test/fixtures';
import { decodeGif } from '../../test/gifDecode';
import { buildScene, OTHERS_AS_LOGGED_NOTE, ROLE_LEGEND, sceneKicker, type DrawnTrack } from '../sceneModel';
import {
  DEFAULT_EXPORT_WIDTH_PX, ExportRefused, MAX_EXPORT_FRAMES, MAX_EXPORT_HEIGHT_PX, MAX_EXPORT_WIDTH_PX, MAX_GIF_TOTAL_PIXELS, MIN_EXPORT_WIDTH_PX,
  browserDeps, buildExportSvg, captionParagraphs, checkLimits, exportGif, exportPng, exportSvg, exportWebm, layoutFrame, requireEnabled, sceneSvgMarkup, wrapText,
  type Canvas2D, type CanvasPort, type ExportDeps, type ExportScene, type RecorderConstructor,
} from './sceneExport';

/*
 * The scene exporters (sceneExport.ts) with the switch for Waymo-derived outputs passed in as ON: the exporters take it as an argument, so the file
 * waymo_derived_outputs.json is never edited here. Each test that exports also has its OFF counterpart: nothing is produced and nothing is asked
 * of the browser. The browser is stood in for by `deps`: a canvas that records what is drawn on it, an image loader, a recorder, a save.
 */

const sdc = makeTrack({ agent_idx: 0, is_sdc: true, frames: range(0, 31), origin: [-20, 0] });
const challengerLogged = makeTrack({ agent_idx: 1, frames: range(0, 31), origin: [0, -10], heading: Math.PI / 2 });
const challengerPerturbed = makeTrack({ agent_idx: 1, frames: range(0, 31), origin: [0, -10], heading: Math.PI / 2 + 0.1 });
const bystander = makeTrack({ agent_idx: 2, agent_type: 3, frames: range(0, 31), origin: [20, 20] });

function sceneOf(drawn: DrawnTrack[], focusAll = true): ExportScene {
  return { scenarioId: 'abc123def4567890', drawn, focus: focusAll ? drawn : drawn.filter((d) => d.role !== 'other'), collisionFrame: 12 };
}
const comparing = (): DrawnTrack[] => buildScene([sdc, challengerLogged, bystander], {
  scenario_id: 'x', target_idx: 1, delta: [0, 0, 0, 0], delta_labels: null, min_perturbation: 0.1, collision_timestep: 12, baseline: challengerLogged, perturbed: challengerPerturbed,
});
const recordedOnly = (): DrawnTrack[] => buildScene([sdc, bystander], undefined);

// the copy lint's words (pages/copyLint.test.tsx): none may be in anything an export says
const BANNED = [/\blive\b/i, /\bsafety\b/i, /\bend-to-end\b/i, /\badversarial\b/i, /\bsafe\b/i];

interface Recorded {
  fillText: [string, number, number][];
  fillRect: number[][];
  drawImage: unknown[][];
  canvases: [number, number][];
  loaded: [string, number, number][];
  saved: { blob: Blob; filename: string }[];
  toBlobTypes: (string | undefined)[];
  captureFps: (number | undefined)[];
  recorder: { options?: { mimeType: string }; started: number; stopped: number; sleeps: number[] };
}

function fakeDeps(opts: { webm?: boolean | 'no-type'; toBlobNull?: boolean; asyncStop?: boolean; noCaptureStream?: boolean; paint?: (frame: number) => [number, number, number] } = {}): { deps: ExportDeps; rec: Recorded } {
  const rec: Recorded = { fillText: [], fillRect: [], drawImage: [], canvases: [], loaded: [], saved: [], toBlobTypes: [], captureFps: [], recorder: { started: 0, stopped: 0, sleeps: [] } };
  let shown = 0;
  const makeCanvas = (width: number, height: number): CanvasPort => {
    rec.canvases.push([width, height]);
    const ctx: Canvas2D = {
      fillStyle: '', font: '', textBaseline: 'alphabetic',
      fillRect: (...a) => void rec.fillRect.push(a),
      drawImage: (...a) => void rec.drawImage.push(a),
      fillText: (text, x, y) => void rec.fillText.push([text, x, y]),
      getImageData: (_x, _y, w, h) => {
        const data = new Uint8ClampedArray(w * h * 4);
        const [r, g, b] = opts.paint ? opts.paint(shown) : [(shown * 40) & 255, 20, 200];
        for (let i = 0; i < w * h; i += 1) data.set([r, g, b, 255], i * 4);
        shown += 1;
        return { data };
      },
    };
    const canvas: CanvasPort = {
      width, height,
      getContext: () => ctx,
      toBlob: (cb, type) => {
        rec.toBlobTypes.push(type);
        cb(opts.toBlobNull ? null : new Blob(['png'], { type: type ?? 'image/png' }));
      },
      captureStream: (fps) => {
        rec.captureFps.push(fps);
        return { tag: 'stream' } as unknown as MediaStream;
      },
    };
    if (opts.noCaptureStream) delete canvas.captureStream;
    return canvas;
  };
  class FakeRecorder {
    ondataavailable: ((e: { data: Blob }) => void) | null = null;
    onstop: (() => void) | null = null;
    constructor(public stream: MediaStream, options: { mimeType: string }) {
      rec.recorder.options = options;
    }
    static isTypeSupported(type: string): boolean {
      return opts.webm !== 'no-type' && type === 'video/webm';
    }
    start(): void { rec.recorder.started += 1; }
    stop(): void {
      rec.recorder.stopped += 1;
      const finish = () => {
        this.ondataavailable?.({ data: new Blob(['chunk-a']) });
        this.ondataavailable?.({ data: new Blob(['chunk-b']) });
        this.onstop?.();
      };
      if (opts.asyncStop) setTimeout(finish, 5);
      else finish();
    }
  }
  const deps: ExportDeps = {
    loadImage: async (svg, w, h) => {
      rec.loaded.push([svg, w, h]);
      return {} as CanvasImageSource;
    },
    createCanvas: makeCanvas,
    save: (blob, filename) => void rec.saved.push({ blob, filename }),
    MediaRecorder: opts.webm === false ? undefined : (FakeRecorder as unknown as RecorderConstructor),
    sleep: async (ms) => void rec.recorder.sleeps.push(ms),
  };
  return { deps, rec };
}

const readAs = <T,>(blob: Blob, how: 'readAsText' | 'readAsArrayBuffer'): Promise<T> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as T);
    reader.onerror = () => reject(reader.error);
    reader[how](blob);
  });
const textOf = (blob: Blob): Promise<string> => readAs<string>(blob, 'readAsText');
const bytesOf = async (blob: Blob): Promise<Uint8Array> => new Uint8Array(await readAs<ArrayBuffer>(blob, 'readAsArrayBuffer'));
const parse = (svg: string): Document => new DOMParser().parseFromString(svg, 'image/svg+xml');

describe('the switch', () => {
  it('lets only the boolean true through', () => {
    expect(() => requireEnabled(true)).not.toThrow();
    for (const v of [false, undefined, null, 1, 'true', {}, [] as unknown]) expect(() => requireEnabled(v as boolean), String(v)).toThrow(ExportRefused);
    expect(() => requireEnabled(false)).toThrow('Export is switched off.');
  });

  it('refuses every exporter when it is off, before the browser is asked for anything', async () => {
    const { deps, rec } = fakeDeps();
    const scene = sceneOf(comparing());
    await expect(exportSvg({ enabled: false, scene, frame: 12, deps })).rejects.toThrow('Export is switched off.');
    await expect(exportPng({ enabled: false, scene, frame: 12, deps })).rejects.toThrow('Export is switched off.');
    await expect(exportWebm({ enabled: false, scene, first: 0, last: 5, deps })).rejects.toThrow('Export is switched off.');
    await expect(exportGif({ enabled: false, scene, first: 0, last: 5, deps })).rejects.toThrow('Export is switched off.');
    expect(rec.canvases).toEqual([]);
    expect(rec.loaded).toEqual([]);
    expect(rec.saved).toEqual([]);
    expect(rec.recorder.started).toBe(0);
  });
});

describe('the caption of an exported frame', () => {
  it('carries the title, the recorded-versus-perturbed caveat, the track names and the attribution, in that order', () => {
    const p = captionParagraphs(sceneOf(comparing()), 12);
    expect(p[0]).toBe('AV Scenario Stress-Tester · scenario abc123def4567890 · frame 12');
    expect(p[1]).toBe(sceneKicker(true));
    expect(p[1]).toBe('RECORDED VS PERTURBED, FRAME BY FRAME');
    expect(p[2]).toBe(`Tracks drawn: ${ROLE_LEGEND.sdc}; ${ROLE_LEGEND.challenger_logged}; ${ROLE_LEGEND.challenger_perturbed}; Other agent: cyclist.`);
    expect(p[2]).toBe('Tracks drawn: SDC (self-driving car); Challenger, as logged; Challenger, perturbed; Other agent: cyclist.');
    expect(p[3]).toBe(OTHERS_AS_LOGGED_NOTE);
    expect(p[4]).toBe(WAYMO_ATTRIBUTION);
    expect(p).toHaveLength(5);
  });

  it('says it is only the recording when no perturbed track is drawn, and has no note about the search then', () => {
    const p = captionParagraphs(sceneOf(recordedOnly()), 3);
    expect(p[1]).toBe('RECORDED SCENE');
    expect(p.join(' ')).not.toContain('perturbed');
    expect(p).not.toContain(OTHERS_AS_LOGGED_NOTE);
    expect(p.at(-1)).toBe(WAYMO_ATTRIBUTION);
  });

  it('leaves out the note about the other agents when only the SDC and the challenger are framed', () => {
    const p = captionParagraphs(sceneOf(comparing(), false), 12);
    expect(p).not.toContain(OTHERS_AS_LOGGED_NOTE);
    expect(p).toHaveLength(4);
    expect(p[3]).toBe(WAYMO_ATTRIBUTION);
  });

  it('is the recording’s heading, with the logged challenger named, when the challenger is drawn as logged and nothing perturbed is', () => {
    const logged = buildScene([sdc, challengerLogged], {
      scenario_id: 'x', target_idx: 1, delta: null, delta_labels: null, min_perturbation: null, collision_timestep: null, baseline: challengerLogged, perturbed: null,
    });
    const p = captionParagraphs(sceneOf(logged), 4);
    expect(p[1]).toBe('RECORDED SCENE');
    expect(p[2]).toBe('Tracks drawn: SDC (self-driving car); Challenger, as logged.');
    expect(p).toHaveLength(4);
  });

  it('uses none of the words the copy lint forbids', () => {
    for (const drawn of [comparing(), recordedOnly()]) {
      for (const text of captionParagraphs(sceneOf(drawn), 5)) for (const re of BANNED) expect(text, `${re}`).not.toMatch(re);
    }
  });

  it('wraps to a width without cutting a word, and keeps a word that is longer than the line', () => {
    expect(wrapText('aaa bbb ccc ddd', 7)).toEqual(['aaa bbb', 'ccc ddd']);
    expect(wrapText('aaaa bbb', 7)).toEqual(['aaaa', 'bbb']);                  // eight characters do not fit in seven
    expect(wrapText('aaaa bbb', 8)).toEqual(['aaaa bbb']);                  // and eight do fit in eight
    expect(wrapText('aaa bbbbbbbbbb c', 5)).toEqual(['aaa', 'bbbbbbbbbb', 'c']);
    expect(wrapText('', 10)).toEqual([]);
    expect(wrapText('  spaced   out  ', 40)).toEqual(['spaced out']);
  });

  it('wraps the attribution so that its rows join back to exactly the attribution', () => {
    const layout = layoutFrame(sceneOf(comparing()), 12, DEFAULT_EXPORT_WIDTH_PX);
    expect(layout.attributionRows.length).toBeGreaterThan(1);
    expect(layout.attributionRows.map((r) => r.text).join(' ')).toBe(WAYMO_ATTRIBUTION);
    expect(layout.rows.slice(-layout.attributionRows.length)).toEqual(layout.attributionRows);
    expect(layout.height).toBe(layout.sceneHeight + layout.bandHeight);
    expect(layout.sceneHeight).toBe(Math.round(DEFAULT_EXPORT_WIDTH_PX * 9 / 16));
  });
});

describe('the SVG', () => {
  const scene = sceneOf(comparing());

  it('is one SVG document with the scene, the style and the caption as text', () => {
    const svg = buildExportSvg(scene, 12, DEFAULT_EXPORT_WIDTH_PX);
    const doc = parse(svg);
    expect(doc.querySelector('parsererror')).toBeNull();
    expect(doc.documentElement.tagName).toBe('svg');
    expect(doc.querySelectorAll('svg.scene').length).toBe(1);
    expect(doc.querySelectorAll('svg.scene polygon').length).toBeGreaterThan(0);
    expect(doc.querySelector('style')!.textContent).toContain('.scene .footprint.colliding polygon');
    expect(doc.querySelector('title')!.textContent).toBe('AV Scenario Stress-Tester · scenario abc123def4567890 · frame 12');
  });

  it('contains the attribution as text, joined back from its rows, exactly as the footer has it', () => {
    const doc = parse(buildExportSvg(scene, 12, DEFAULT_EXPORT_WIDTH_PX));
    const rows = [...doc.querySelectorAll('tspan.attribution')].map((t) => t.textContent);
    expect(rows.join(' ')).toBe(WAYMO_ATTRIBUTION);
    expect(rows.join(' ')).toContain('This software was made using the Waymo Open Dataset');
    expect(rows.join(' ')).toContain('waymo.com/open/terms');
  });

  it('contains the caveat: the heading, the track names and the note that the others are drawn as logged', () => {
    const text = [...parse(buildExportSvg(scene, 12, DEFAULT_EXPORT_WIDTH_PX)).querySelectorAll('tspan')].map((t) => t.textContent).join(' ');
    expect(text).toContain('RECORDED VS PERTURBED, FRAME BY FRAME');
    expect(text).toContain('Challenger, as logged; Challenger, perturbed');
    expect(text).toContain("The SDC and every other agent are drawn as logged.");
  });

  it('is the frame asked for: the colliding pair is marked at the collision frame and not before', () => {
    const at = parse(buildExportSvg(scene, 12, DEFAULT_EXPORT_WIDTH_PX));
    const before = parse(buildExportSvg(scene, 5, DEFAULT_EXPORT_WIDTH_PX));
    expect(at.querySelectorAll('.footprint.colliding').length).toBe(2);
    expect(before.querySelectorAll('.footprint.colliding').length).toBe(0);
    expect(at.documentElement.querySelector('svg.scene')!.getAttribute('aria-label')).toBe('Scene paths, positions at frame 12');
  });

  it('is exported as an image/svg+xml blob named for the scenario and the frame, and saved', async () => {
    const { deps, rec } = fakeDeps();
    const out = await exportSvg({ enabled: true, scene, frame: 12, deps });
    expect(out.filename).toBe('abc123def4567890-frame-12.svg');
    expect(out.blob.type).toBe('image/svg+xml');
    expect(await textOf(out.blob)).toBe(out.svg);
    expect(rec.saved).toEqual([{ blob: out.blob, filename: 'abc123def4567890-frame-12.svg' }]);
    expect(rec.canvases).toEqual([]);
  });

  it('escapes what it puts in the file', () => {
    const odd = { ...scene, scenarioId: 'a<b>&"c' };
    const svg = buildExportSvg(odd, 1, DEFAULT_EXPORT_WIDTH_PX);
    expect(parse(svg).querySelector('parsererror')).toBeNull();
    expect(parse(svg).querySelector('title')!.textContent).toContain('a<b>&"c');
  });

  it('stands alone: the scene markup for a canvas carries the page’s styles, and the one inside the file leaves them to the file', () => {
    expect(sceneSvgMarkup(scene, 12, 0, 0, 640, 360, true)).toContain('<style>');
    expect(sceneSvgMarkup(scene, 12, 0, 0, 640, 360, false)).not.toContain('<style>');
    expect(sceneSvgMarkup(scene, 12, 0, 0, 640, 360, true)).toMatch(/^<svg xmlns="http:\/\/www\.w3\.org\/2000\/svg" x="0" y="0" width="640" height="360" /);
  });

  it('refuses a scene with nothing observed to draw', async () => {
    const empty = { scenarioId: 'e', drawn: [{ key: 'k', role: 'sdc' as const, track: makeTrack({ frames: [] })}], focus: [], collisionFrame: null };
    await expect(exportSvg({ enabled: true, scene: empty, frame: 0, deps: fakeDeps().deps })).rejects.toThrow('nothing to export');
  });
});

describe('the PNG', () => {
  const scene = sceneOf(comparing());

  it('draws the scene image and then the caption on a canvas of the layout’s size, and asks for image/png', async () => {
    const { deps, rec } = fakeDeps();
    const layout = layoutFrame(scene, 12, DEFAULT_EXPORT_WIDTH_PX);
    const out = await exportPng({ enabled: true, scene, frame: 12, deps });
    expect(rec.canvases).toEqual([[layout.width, layout.height]]);
    expect(rec.loaded).toHaveLength(1);
    expect(rec.loaded[0]![1]).toBe(layout.width);
    expect(rec.loaded[0]![2]).toBe(layout.sceneHeight);
    expect(rec.loaded[0]![0]).toContain('<style>');
    expect(rec.drawImage).toEqual([[{}, 0, 0, layout.width, layout.sceneHeight]]);
    expect(rec.toBlobTypes).toEqual(['image/png']);
    expect(out.filename).toBe('abc123def4567890-frame-12.png');
    expect(out.blob.type).toBe('image/png');
    expect(rec.saved).toEqual([{ blob: out.blob, filename: 'abc123def4567890-frame-12.png' }]);
  });

  it('draws the attribution onto the canvas, row by row, and the title and the caveat', async () => {
    const { deps, rec } = fakeDeps();
    const layout = layoutFrame(scene, 12, DEFAULT_EXPORT_WIDTH_PX);
    await exportPng({ enabled: true, scene, frame: 12, deps });
    const drawn = rec.fillText.map(([text]) => text);
    expect(drawn).toEqual(layout.rows.map((r) => r.text));
    expect(drawn.slice(-layout.attributionRows.length).join(' ')).toBe(WAYMO_ATTRIBUTION);
    expect(drawn.join(' ')).toContain('RECORDED VS PERTURBED, FRAME BY FRAME');
    expect(rec.fillText.map(([, x, y]) => [x, y])).toEqual(layout.rows.map((r) => [16, layout.sceneHeight + r.y]));
    expect(rec.fillRect).toEqual([[0, 0, layout.width, layout.height], [0, layout.sceneHeight, layout.width, layout.bandHeight]]);
  });

  it('refuses when the browser gives no PNG', async () => {
    await expect(exportPng({ enabled: true, scene, frame: 12, deps: fakeDeps({ toBlobNull: true }).deps })).rejects.toThrow('could not make a PNG');
  });
});

describe('the WebM', () => {
  const scene = sceneOf(comparing());

  it('records a canvas stream at the asked rate with a WebM recorder, one drawn frame per replay frame', async () => {
    const { deps, rec } = fakeDeps();
    const out = await exportWebm({ enabled: true, scene, first: 10, last: 14, fps: 5, deps });
    expect(rec.captureFps).toEqual([5]);
    expect(rec.recorder.options).toEqual({ mimeType: 'video/webm' });
    expect(rec.recorder.started).toBe(1);
    expect(rec.recorder.stopped).toBe(1);
    expect(rec.loaded).toHaveLength(5);
    expect(rec.recorder.sleeps).toEqual([200, 200, 200, 200, 200]);
    expect(rec.canvases).toHaveLength(1);
    expect(out.filename).toBe('abc123def4567890-frames-10-14.webm');
    expect(out.blob.type).toBe('video/webm');
    expect(await textOf(out.blob)).toBe('chunk-achunk-b');
    expect(rec.saved).toEqual([{ blob: out.blob, filename: 'abc123def4567890-frames-10-14.webm' }]);
  });

  it('draws the frame number of each replay frame in its caption', async () => {
    const { deps, rec } = fakeDeps();
    await exportWebm({ enabled: true, scene, first: 3, last: 5, deps });
    const titles = rec.fillText.map(([t]) => t).filter((t) => t.includes('· frame '));
    expect(titles).toEqual([3, 4, 5].map((f) => `AV Scenario Stress-Tester · scenario abc123def4567890 · frame ${f}`));
  });

  it('defaults to ten frames a second, the rate of the data', async () => {
    const { deps, rec } = fakeDeps();
    await exportWebm({ enabled: true, scene, first: 0, last: 1, deps });
    expect(rec.captureFps).toEqual([10]);
  });

  it('refuses, in one line, a browser with no WebM recording', async () => {
    for (const webm of [false as const, 'no-type' as const]) {
      const { deps, rec } = fakeDeps({ webm });
      await expect(exportWebm({ enabled: true, scene, first: 0, last: 2, deps })).rejects.toThrow('Export refused: this browser cannot record WebM video.');
      expect(rec.saved).toEqual([]);
    }
  });
});

describe('the GIF', () => {
  const scene = sceneOf(comparing());

  it('decodes to the number of frames asked for, at the layout’s size, looping, at the delay of the rate', async () => {
    const { deps, rec } = fakeDeps();
    const layout = layoutFrame(scene, 0, 320);
    const out = await exportGif({ enabled: true, scene, first: 4, last: 9, fps: 10, width: 320, deps });
    const gif = decodeGif(await bytesOf(out.blob));
    expect(gif.frames).toHaveLength(6);
    expect([gif.width, gif.height]).toEqual([layout.width, layout.height]);
    expect(gif.delays).toEqual([10, 10, 10, 10, 10, 10]);
    expect(gif.loops).toBe(true);
    expect(out.blob.type).toBe('image/gif');
    expect(out.filename).toBe('abc123def4567890-frames-4-9.gif');
    expect(rec.saved).toHaveLength(1);
    expect(rec.loaded).toHaveLength(6);
  });

  it('has each drawn frame in its place: the pixels decode to the colour painted for that frame', async () => {
    const { deps } = fakeDeps({ paint: (n) => [[255, 0, 0], [0, 255, 0], [0, 0, 255]][n % 3] as [number, number, number] });
    const out = await exportGif({ enabled: true, scene, first: 0, last: 2, width: 320, deps });
    const gif = decodeGif(await bytesOf(out.blob));
    const colourOf = (i: number) => [...gif.palette.subarray(gif.frames[i]![0]! * 3, gif.frames[i]![0]! * 3 + 3)];
    expect([colourOf(0), colourOf(1), colourOf(2)]).toEqual([[255, 0, 0], [0, 255, 0], [0, 0, 255]]);
    for (const f of gif.frames) expect(new Set(f).size).toBe(1);
  });

  it('turns the rate into a delay in hundredths of a second', async () => {
    const { deps } = fakeDeps();
    const out = await exportGif({ enabled: true, scene, first: 0, last: 1, fps: 4, width: 320, deps });
    expect(decodeGif(await bytesOf(out.blob)).delays).toEqual([25, 25]);
  });

  it('refuses a GIF over the total pixels it allows, in one line, before drawing anything', async () => {
    const { deps, rec } = fakeDeps();
    const layout = layoutFrame(scene, 0, MAX_EXPORT_WIDTH_PX);
    const frames = Math.floor(MAX_GIF_TOTAL_PIXELS / (layout.width * layout.height)) + 1;
    expect(frames).toBeLessThanOrEqual(MAX_EXPORT_FRAMES);
    const err = await exportGif({ enabled: true, scene, first: 0, last: frames - 1, width: MAX_EXPORT_WIDTH_PX, deps }).catch((e) => e);
    expect(err).toBeInstanceOf(ExportRefused);
    expect(err.message).toMatch(/^Export refused: a GIF of \d+ frames at \d+ by \d+ pixels is over the limit of 64000000 pixels in all\.$/);
    expect(rec.canvases).toEqual([]);
  });
});

describe('the limits', () => {
  const scene = sceneOf(comparing());

  it('are stated: 120 frames, 1280 by 1080 pixels, at least 320 wide', () => {
    expect([MAX_EXPORT_FRAMES, MAX_EXPORT_WIDTH_PX, MAX_EXPORT_HEIGHT_PX, MIN_EXPORT_WIDTH_PX, DEFAULT_EXPORT_WIDTH_PX]).toEqual([120, 1280, 1080, 320, 960]);
  });

  it('let the largest allowed request through and refuse one more of anything', () => {
    expect(() => checkLimits(120, 1280, 1080)).not.toThrow();
    expect(() => checkLimits(1, 320, 100)).not.toThrow();
    for (const args of [[121, 1280, 1080], [120, 1281, 1080], [120, 1280, 1081]] as const) {
      expect(() => checkLimits(args[0], args[1], args[2]), String(args)).toThrow(/^Export refused: \d+ frames at \d+ by \d+ pixels is over the limit of 120 frames and 1280 by 1080 pixels\.$/);
    }
    expect(() => checkLimits(1, 319, 100)).toThrow(/under the minimum of 320 pixels/);
    for (const frames of [0, -1, 1.5, NaN]) expect(() => checkLimits(frames, 640, 400), String(frames)).toThrow(/is not a number of frames/);
  });

  it('refuse a replay of more frames than allowed, with a one-line message, having drawn nothing', async () => {
    const { deps, rec } = fakeDeps();
    const err = await exportWebm({ enabled: true, scene, first: 0, last: 120, deps }).catch((e) => e);
    expect(err).toBeInstanceOf(ExportRefused);
    expect(err.message).toBe(`Export refused: 121 frames at 960 by ${layoutFrame(scene, 0, 960).height} pixels is over the limit of 120 frames and 1280 by 1080 pixels.`);
    expect(err.message).not.toContain('\n');
    expect(rec.canvases).toEqual([]);
    await expect(exportGif({ enabled: true, scene, first: 0, last: 120, deps })).rejects.toThrow(/over the limit of 120 frames/);
  });

  it('accept a replay of exactly the most frames allowed', async () => {
    const { deps, rec } = fakeDeps();
    await exportWebm({ enabled: true, scene, first: 0, last: 119, width: 640, deps });
    expect(rec.loaded).toHaveLength(120);
  });

  it('refuse a width over the limit, or under it, for a frame', async () => {
    const { deps, rec } = fakeDeps();
    await expect(exportSvg({ enabled: true, scene, frame: 1, width: 1281, deps })).rejects.toThrow(/over the limit/);
    await expect(exportPng({ enabled: true, scene, frame: 1, width: 1281, deps })).rejects.toThrow(/over the limit/);
    await expect(exportSvg({ enabled: true, scene, frame: 1, width: 319, deps })).rejects.toThrow(/under the minimum/);
    expect(rec.saved).toEqual([]);
    expect(rec.canvases).toEqual([]);
  });

  it('refuse a height over the limit: a width that fits but whose caption makes the frame too tall', () => {
    const tall = layoutFrame(scene, 1, MAX_EXPORT_WIDTH_PX);
    expect(tall.height).toBeLessThanOrEqual(MAX_EXPORT_HEIGHT_PX);
    expect(() => checkLimits(1, 640, MAX_EXPORT_HEIGHT_PX + 1)).toThrow(/over the limit/);
  });

  it('refuse a range that is not one, and a rate outside 1 to 30', async () => {
    const { deps } = fakeDeps();
    await expect(exportWebm({ enabled: true, scene, first: 5, last: 4, deps })).rejects.toThrow(/not a range of frames/);
    await expect(exportGif({ enabled: true, scene, first: 0.5, last: 4, deps })).rejects.toThrow(/not a range of frames/);
    for (const fps of [0, 31, 2.5]) await expect(exportWebm({ enabled: true, scene, first: 0, last: 2, fps, deps })).rejects.toThrow(/frames a second/);
  });
});

describe('the layout of a frame', () => {
  const scene = sceneOf(comparing());

  it('puts the rows a line apart inside a paragraph and a line and a gap apart between paragraphs, from the padding down', () => {
    const layout = layoutFrame(scene, 12, DEFAULT_EXPORT_WIDTH_PX);
    const paragraphs = captionParagraphs(scene, 12);
    const maxChars = Math.floor((DEFAULT_EXPORT_WIDTH_PX - 32) / (13 * 0.58));
    let expectedY = 16;
    let row = 0;
    for (const p of paragraphs) {
      for (const text of wrapText(p, maxChars)) {
        expect(layout.rows[row]).toEqual({ text, y: expectedY });
        expectedY += 18;
        row += 1;
      }
      expectedY += 6;
    }
    expect(row).toBe(layout.rows.length);
    expect(layout.bandHeight).toBe(layout.rows.at(-1)!.y + 18 + 16);                    // the last row’s bottom, then the padding
  });

  it('is as wide as asked, and a narrower frame wraps more', () => {
    const wide = layoutFrame(scene, 12, 1280);
    const narrow = layoutFrame(scene, 12, 320);
    expect([wide.width, narrow.width]).toEqual([1280, 320]);
    expect(narrow.rows.length).toBeGreaterThan(wide.rows.length);
    expect(narrow.attributionRows.map((r) => r.text).join(' ')).toBe(WAYMO_ATTRIBUTION);
  });

  it('gives the file its size, its background, its band and its text, at the places the layout says', () => {
    const layout = layoutFrame(scene, 12, DEFAULT_EXPORT_WIDTH_PX);
    const doc = parse(buildExportSvg(scene, 12, DEFAULT_EXPORT_WIDTH_PX));
    const root = doc.documentElement;
    expect([root.getAttribute('width'), root.getAttribute('height'), root.getAttribute('viewBox')]).toEqual([
      String(layout.width), String(layout.height), `0 0 ${layout.width} ${layout.height}`]);
    const rects = [...root.children].filter((c) => c.tagName === 'rect');
    expect(rects.map((r) => [r.getAttribute('y') ?? '0', r.getAttribute('height'), r.getAttribute('fill')])).toEqual([
      ['0', String(layout.height), '#0b100e'], [String(layout.sceneHeight), String(layout.bandHeight), '#111b17']]);
    const inner = root.querySelector('svg.scene')!;
    expect([inner.getAttribute('x'), inner.getAttribute('y'), inner.getAttribute('width'), inner.getAttribute('height')]).toEqual(['0', '0', String(layout.width), String(layout.sceneHeight)]);
    const tspans = [...root.querySelectorAll('tspan')];
    expect(tspans.map((t) => [t.getAttribute('x'), t.getAttribute('y'), t.textContent])).toEqual(layout.rows.map((r) => ['16', String(layout.sceneHeight + r.y + 13), r.text]));
    const caption = [...root.children].find((c) => c.tagName === 'text')!;
    expect([caption.getAttribute('fill'), caption.getAttribute('font-size'), caption.getAttribute('font-family')]).toEqual(['#c9d1cc', '13', 'Inter, system-ui, sans-serif']);
    expect(tspans.filter((t) => t.getAttribute('class') === 'attribution')).toHaveLength(layout.attributionRows.length);
  });
});

describe('what the exporters leave to the caller', () => {
  const scene = sceneOf(comparing());

  it('waits for the recorder to finish before it hands the video over', async () => {
    const { deps } = fakeDeps({ asyncStop: true });
    const out = await exportWebm({ enabled: true, scene, first: 0, last: 1, deps });
    expect(await textOf(out.blob)).toBe('chunk-achunk-b');
  });

  it('refuses a canvas that cannot be recorded', async () => {
    await expect(exportWebm({ enabled: true, scene, first: 0, last: 1, deps: fakeDeps({ noCaptureStream: true }).deps })).rejects.toThrow('cannot record from a canvas');
  });

  it('refuses when the browser gives no drawing context', async () => {
    const { deps } = fakeDeps();
    const bare = { ...deps, createCanvas: () => ({ width: 1, height: 1, getContext: () => null, toBlob: () => {} }) as unknown as CanvasPort };
    await expect(exportPng({ enabled: true, scene, frame: 1, deps: bare })).rejects.toThrow('no canvas to draw on');
  });

  it('takes one frame, and the lowest and highest rate', async () => {
    const { deps, rec } = fakeDeps();
    await exportWebm({ enabled: true, scene, first: 7, last: 7, fps: 1, deps });
    await exportWebm({ enabled: true, scene, first: 7, last: 8, fps: 30, deps });
    expect(rec.captureFps).toEqual([1, 30]);
    expect(rec.recorder.sleeps).toEqual([1000, 1000 / 30, 1000 / 30]);
    expect(rec.loaded).toHaveLength(3);
  });

  it('names a file from a scenario id that has only the characters a file name should', async () => {
    const { deps } = fakeDeps();
    const out = await exportSvg({ enabled: true, scene: { ...scene, scenarioId: '../a b/c:d' }, frame: 1, deps });
    expect(out.filename).toBe('___a_b_c_d-frame-1.svg');
  });

  it('is 960 pixels wide unless told otherwise', async () => {
    const { deps, rec } = fakeDeps();
    await exportPng({ enabled: true, scene, frame: 1, deps });
    expect(rec.canvases[0]![0]).toBe(960);
  });

  it('refuses a rate that is not a whole number', async () => {
    await expect(exportGif({ enabled: true, scene, first: 0, last: 1, fps: 7.5, deps: fakeDeps().deps })).rejects.toThrow(/frames a second/);
  });
});

describe('the browser’s own deps', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('loads an SVG as an image from a data URL of exactly the text, at the size asked', async () => {
    class FakeImage {
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      constructor(public width: number, public height: number) {}
      set src(value: string) {
        FakeImage.sources.push(value);
        queueMicrotask(() => this.onload?.());
      }
      static sources: string[] = [];
    }
    vi.stubGlobal('Image', FakeImage);
    const image = (await browserDeps().loadImage('<svg a="1&2"/>', 64, 48)) as unknown as FakeImage;
    expect([image.width, image.height]).toEqual([64, 48]);
    expect(FakeImage.sources).toEqual([`data:image/svg+xml;charset=utf-8,${encodeURIComponent('<svg a="1&2"/>')}`]);
  });

  it('refuses, in one line, when the browser cannot draw the image', async () => {
    class BrokenImage {
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      set src(_value: string) {
        queueMicrotask(() => this.onerror?.());
      }
    }
    vi.stubGlobal('Image', BrokenImage);
    await expect(browserDeps().loadImage('<svg/>', 1, 1)).rejects.toThrow('Export refused: the browser could not draw the scene.');
  });

  it('makes a canvas of the size asked', () => {
    const canvas = browserDeps().createCanvas(123, 45);
    expect([canvas.width, canvas.height]).toEqual([123, 45]);
  });

  it('saves a file as a download link with the file name, clicks it, removes it and lets the address go later', () => {
    vi.useFakeTimers();
    const created: Blob[] = [];
    const revoked: string[] = [];
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: (b: Blob) => (created.push(b), 'blob:fake-1'), revokeObjectURL: (u: string) => void revoked.push(u) }));
    const clicks: { href: string; download: string }[] = [];
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      clicks.push({ href: this.href, download: this.download });
      expect(document.body.contains(this)).toBe(true);
    });
    const blob = new Blob(['x']);
    browserDeps().save(blob, 'a-frame-1.svg');
    expect(created).toEqual([blob]);
    expect(clicks).toEqual([{ href: 'blob:fake-1', download: 'a-frame-1.svg' }]);
    expect(document.querySelector('a[download]')).toBeNull();
    expect(revoked).toEqual([]);
    vi.advanceTimersByTime(9_999);
    expect(revoked).toEqual([]);                                   // the address stays good for ten seconds, for the download to start
    vi.advanceTimersByTime(1);
    expect(revoked).toEqual(['blob:fake-1']);
    click.mockRestore();
  });

  it('sleeps for the time asked and reads the recorder from the browser', async () => {
    vi.useFakeTimers();
    let done = false;
    void browserDeps().sleep(250).then(() => { done = true; });
    await vi.advanceTimersByTimeAsync(249);
    expect(done).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect(done).toBe(true);
    vi.stubGlobal('MediaRecorder', class {});
    expect(browserDeps().MediaRecorder).toBeDefined();
  });
});
