import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { PROJECT_NAME, WAYMO_ATTRIBUTION } from '../../components/Footer';
import { describeAgentType } from '../../domain/agentTypes';
import { SceneView } from '../SceneView';
import { othersAsLoggedNote, ROLE_LEGEND, sceneKicker, type DrawnTrack, type Role } from '../sceneModel';
import { EXPORT_STYLE } from './exportStyle';
import { encodeGif, quantizeFrame, type GifFrame } from './gif';

/*
 * Exporting a frame (SVG, PNG) or the replay (WebM, GIF) of a scene. Everything here produces a file from Waymo-derived tracks, so all of it sits
 * behind the one switch (waymoDerivedOutputs.ts): every exporter takes the switch's value as `enabled` and
 * refuses unless it is exactly true, so a test passes `true` without editing the file, and the app passes the constant, which is false until the
 * licence reply is settled. The module is imported lazily by ScenePanel and only when the switch is on.
 *
 * What an exported frame carries, in the file itself (inside the SVG as text; drawn onto the canvas for the PNG, the WebM and the GIF):
 *   - the title: the project's name (Footer PROJECT_NAME), the scenario id and the frame;
 *   - the caveat the page carries about recorded and perturbed tracks: the replay's heading (sceneKicker), the names of the tracks drawn
 *     (ROLE_LEGEND, the legend's words) and, where the page shows it, the note that the SDC and every other agent are drawn as logged
 *     (othersAsLoggedNote);
 *   - the attribution the licence requires (Footer WAYMO_ATTRIBUTION): the same constant the footer on every page renders.
 * The scene itself is the SceneView the page draws, rendered for the frame asked for, with the page's scene styles (exportStyle.ts).
 *
 * Limits, so that a request cannot ask for an unbounded amount of work: at most MAX_EXPORT_FRAMES frames, at most MAX_EXPORT_WIDTH_PX by
 * MAX_EXPORT_HEIGHT_PX pixels a frame (caption included), at least MIN_EXPORT_WIDTH_PX wide, and for a GIF at most MAX_GIF_TOTAL_PIXELS pixels over all
 * its frames. Beyond them the exporter throws ExportRefused with a one-line message and has produced nothing.
 */

export const MAX_EXPORT_FRAMES = 120;
export const MAX_EXPORT_WIDTH_PX = 1280;
export const MAX_EXPORT_HEIGHT_PX = 1080;
export const MIN_EXPORT_WIDTH_PX = 320;
export const MAX_GIF_TOTAL_PIXELS = 64_000_000;
export const DEFAULT_EXPORT_WIDTH_PX = 960;
export const DEFAULT_FPS = 10;
export const MAX_FPS = 30;

const SCENE_ASPECT = 9 / 16;
const PAD = 16;
const FONT_PX = 13;
const LINE_PX = 18;
const PARAGRAPH_GAP_PX = 6;
const CHAR_WIDTH = 0.58;
const BACKGROUND = '#0b100e';
const BAND_BACKGROUND = '#111b17';
const TEXT_COLOUR = '#c9d1cc';
const FONT_FAMILY = 'Inter, system-ui, sans-serif';

/** The one-line reason an export was not made. */
export class ExportRefused extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ExportRefused';
  }
}

export interface ExportScene {
  scenarioId: string;
  drawn: DrawnTrack[];
  /** The tracks the view is framed on, as the page frames them. */
  focus: DrawnTrack[];
  collisionFrame: number | null;
}

export interface FrameRow {
  text: string;
  /** Distance of the text's top from the top of the caption band. */
  y: number;
}

export interface FrameLayout {
  width: number;
  height: number;
  sceneHeight: number;
  bandHeight: number;
  rows: FrameRow[];
  /** The rows of the attribution only, in order: what a test or a reader joins to get the attribution back. */
  attributionRows: FrameRow[];
}

/** The switch, strictly: only the boolean true passes. */
export function requireEnabled(enabled: boolean): void {
  if (enabled !== true) throw new ExportRefused('Export is switched off.');
}

/** Throws ExportRefused when a request is over the limits; the message is one line. */
export function checkLimits(frames: number, width: number, height: number): void {
  if (!Number.isInteger(frames) || frames < 1) throw new ExportRefused(`Export refused: ${frames} is not a number of frames.`);
  if (!Number.isInteger(width) || width < MIN_EXPORT_WIDTH_PX) {
    throw new ExportRefused(`Export refused: a width of ${width} pixels is under the minimum of ${MIN_EXPORT_WIDTH_PX} pixels.`);
  }
  if (frames > MAX_EXPORT_FRAMES || width > MAX_EXPORT_WIDTH_PX || height > MAX_EXPORT_HEIGHT_PX) {
    throw new ExportRefused(
      `Export refused: ${frames} frames at ${width} by ${height} pixels is over the limit of ${MAX_EXPORT_FRAMES} frames and ` +
        `${MAX_EXPORT_WIDTH_PX} by ${MAX_EXPORT_HEIGHT_PX} pixels.`,
    );
  }
}

/** Greedy word wrap to at most `maxChars` characters a line; a word longer than that stays whole on a line of its own. */
export function wrapText(text: string, maxChars: number): string[] {
  const lines: string[] = [];
  let current = '';
  for (const word of text.split(/\s+/).filter(Boolean)) {
    if (current === '') current = word;
    else if (current.length + 1 + word.length <= maxChars) current += ` ${word}`;
    else {
      lines.push(current);
      current = word;
    }
  }
  if (current !== '') lines.push(current);
  return lines;
}

/** The paragraphs of the caption, in order; the last is the attribution. */
export function captionParagraphs(scene: ExportScene, frame: number): string[] {
  const comparing = scene.drawn.some((d) => d.role === 'challenger_perturbed');
  const roles = (Object.keys(ROLE_LEGEND) as Exclude<Role, 'other'>[]).filter((r) => scene.drawn.some((d) => d.role === r));
  const kinds = [...new Set(scene.drawn.filter((d) => d.role === 'other').map((d) => describeAgentType(d.track.agent_type).kind))];
  const legend = [...roles.map((r) => ROLE_LEGEND[r]), ...kinds.map((k) => `Other agent: ${k === 'unknown' ? 'unknown type' : k}`)];
  const paragraphs = [
    `${PROJECT_NAME} · scenario ${scene.scenarioId} · frame ${frame}`,
    sceneKicker(comparing),
    `Tracks drawn: ${legend.join('; ')}.`,
  ];
  const note = othersAsLoggedNote(scene.focus);
  if (note !== null) paragraphs.push(note);
  paragraphs.push(WAYMO_ATTRIBUTION);
  return paragraphs;
}

export function layoutFrame(scene: ExportScene, frame: number, width: number): FrameLayout {
  const maxChars = Math.max(10, Math.floor((width - 2 * PAD) / (FONT_PX * CHAR_WIDTH)));
  const sceneHeight = Math.round(width * SCENE_ASPECT);
  const paragraphs = captionParagraphs(scene, frame);
  const rows: FrameRow[] = [];
  let attributionRows: FrameRow[] = [];
  let y = PAD;
  paragraphs.forEach((p, i) => {
    const made = wrapText(p, maxChars).map((text, j) => ({ text, y: y + j * LINE_PX }));
    y += made.length * LINE_PX + PARAGRAPH_GAP_PX;
    rows.push(...made);
    if (i === paragraphs.length - 1) attributionRows = made;
  });
  const bandHeight = y - PARAGRAPH_GAP_PX + PAD;
  return { width, height: sceneHeight + bandHeight, sceneHeight, bandHeight, rows, attributionRows };
}

/** Escapes text for an XML text node (the file puts text in elements only, never in attributes). */
const xml = (text: string): string => text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/**
 * The scene as the page draws it at one frame, as an <svg> element sized in pixels. `withStyle` puts the page's scene styles inside it (for a file
 * that stands alone, or an image drawn onto a canvas); without it the styles are expected from the enclosing document.
 */
export function sceneSvgMarkup(scene: ExportScene, frame: number, x: number, y: number, width: number, height: number, withStyle: boolean): string {
  const markup = renderToStaticMarkup(
    createElement(SceneView, { drawn: scene.drawn, focus: scene.focus, frame, collisionFrame: scene.collisionFrame }),
  );
  if (!markup.startsWith('<svg ')) throw new ExportRefused('There is nothing to export: no track has an observed position.');
  const attrs = `xmlns="http://www.w3.org/2000/svg" x="${x}" y="${y}" width="${width}" height="${height}" `;
  const style = withStyle ? `<style>${EXPORT_STYLE}</style>` : '';
  return markup.replace(/^<svg /, `<svg ${attrs}`).replace(/^(<svg[^>]*>)/, `$1${style}`);
}

/** The whole frame as one SVG file: the scene, then the caption band with the title, the caveat and the attribution as text. */
export function buildExportSvg(scene: ExportScene, frame: number, width: number): string {
  const layout = layoutFrame(scene, frame, width);
  const { height, sceneHeight, bandHeight, rows } = layout;
  const attributionStart = rows.length - layout.attributionRows.length;
  const text = rows
    .map((r, i) => `<tspan x="${PAD}" y="${sceneHeight + r.y + FONT_PX}"${i >= attributionStart ? ' class="attribution"' : ''}>${xml(r.text)}</tspan>`)
    .join('');
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">` +
    `<title>${xml(`${PROJECT_NAME} · scenario ${scene.scenarioId} · frame ${frame}`)}</title>` +
    `<style>${EXPORT_STYLE}</style>` +
    `<rect width="${width}" height="${height}" fill="${BACKGROUND}"/>` +
    sceneSvgMarkup(scene, frame, 0, 0, width, sceneHeight, false) +
    `<rect y="${sceneHeight}" width="${width}" height="${bandHeight}" fill="${BAND_BACKGROUND}"/>` +
    `<text font-family="${FONT_FAMILY}" font-size="${FONT_PX}" fill="${TEXT_COLOUR}">${text}</text>` +
    `</svg>`
  );
}

// ── what the exporters need from the browser, passed in so that a test can stand in for it ─────────────────────────────────────────────

export interface Canvas2D {
  fillStyle: string | CanvasGradient | CanvasPattern;
  font: string;
  textBaseline: CanvasTextBaseline;
  fillRect(x: number, y: number, w: number, h: number): void;
  drawImage(image: CanvasImageSource, x: number, y: number, w: number, h: number): void;
  fillText(text: string, x: number, y: number): void;
  getImageData(x: number, y: number, w: number, h: number): { data: Uint8ClampedArray };
}

export interface CanvasPort {
  width: number;
  height: number;
  getContext(kind: '2d'): Canvas2D | null;
  toBlob(callback: (blob: Blob | null) => void, type?: string): void;
  captureStream?(fps?: number): MediaStream;
}

export interface RecorderPort {
  ondataavailable: ((event: { data: Blob }) => void) | null;
  onstop: (() => void) | null;
  start(): void;
  stop(): void;
}

export interface RecorderConstructor {
  new (stream: MediaStream, options: { mimeType: string }): RecorderPort;
  isTypeSupported(type: string): boolean;
}

export interface ExportDeps {
  /** An image of an SVG file's text, `width` by `height`. */
  loadImage(svg: string, width: number, height: number): Promise<CanvasImageSource>;
  createCanvas(width: number, height: number): CanvasPort;
  /** Hands the file to the person (a download). */
  save(blob: Blob, filename: string): void;
  MediaRecorder: RecorderConstructor | undefined;
  sleep(ms: number): Promise<void>;
}

export function browserDeps(): ExportDeps {
  return {
    loadImage: (svg, width, height) =>
      new Promise((resolve, reject) => {
        const image = new Image(width, height);
        image.onload = () => resolve(image);
        image.onerror = () => reject(new ExportRefused('Export refused: the browser could not draw the scene.'));
        image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
      }),
    createCanvas: (width, height) => {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      return canvas as unknown as CanvasPort;
    },
    save: (blob, filename) => {
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
    },
    MediaRecorder: (globalThis as { MediaRecorder?: RecorderConstructor }).MediaRecorder,
    sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  };
}

export interface ExportOptions {
  /** The switch's value (waymoDerivedOutputs.ts). Anything but exactly true is refused. */
  enabled: boolean;
  scene: ExportScene;
  width?: number;
  deps?: ExportDeps;
}

export interface FrameExportOptions extends ExportOptions {
  frame: number;
}

export interface ReplayExportOptions extends ExportOptions {
  first: number;
  last: number;
  fps?: number;
}

export interface ExportResult {
  filename: string;
  blob: Blob;
}

function fileStem(scene: ExportScene): string {
  return `${scene.scenarioId.replace(/[^A-Za-z0-9_-]/g, '_')}`;
}

/** The number of frames from `first` to `last` inclusive; ExportRefused when that is not a range. */
function replayCount(first: number, last: number): number {
  if (!Number.isInteger(first) || !Number.isInteger(last) || last < first) throw new ExportRefused(`Export refused: ${first} to ${last} is not a range of frames.`);
  return last - first + 1;
}

const frameNumbers = (first: number, count: number): number[] => Array.from({ length: count }, (_, i) => first + i);

function fpsOf(fps: number | undefined): number {
  const value = fps ?? DEFAULT_FPS;
  if (!Number.isInteger(value) || value < 1 || value > MAX_FPS) throw new ExportRefused(`Export refused: ${value} frames a second is not from 1 to ${MAX_FPS}.`);
  return value;
}

async function drawFrame(canvas: CanvasPort, deps: ExportDeps, scene: ExportScene, frame: number, layout: FrameLayout): Promise<Canvas2D> {
  const ctx = canvas.getContext('2d');
  if (ctx === null) throw new ExportRefused('Export refused: the browser gave no canvas to draw on.');
  const image = await deps.loadImage(sceneSvgMarkup(scene, frame, 0, 0, layout.width, layout.sceneHeight, true), layout.width, layout.sceneHeight);
  ctx.fillStyle = BACKGROUND;
  ctx.fillRect(0, 0, layout.width, layout.height);
  ctx.drawImage(image, 0, 0, layout.width, layout.sceneHeight);
  ctx.fillStyle = BAND_BACKGROUND;
  ctx.fillRect(0, layout.sceneHeight, layout.width, layout.bandHeight);
  ctx.fillStyle = TEXT_COLOUR;
  ctx.font = `${FONT_PX}px ${FONT_FAMILY}`;
  ctx.textBaseline = 'top';
  for (const row of layout.rows) ctx.fillText(row.text, PAD, layout.sceneHeight + row.y);
  return ctx;
}

function widthOf(options: ExportOptions): number {
  return options.width ?? DEFAULT_EXPORT_WIDTH_PX;
}

/** The current frame as an SVG file; hands it to `deps.save` and returns it. */
export async function exportSvg(options: FrameExportOptions): Promise<ExportResult & { svg: string }> {
  requireEnabled(options.enabled);
  const width = widthOf(options);
  const layout = layoutFrame(options.scene, options.frame, width);
  checkLimits(1, width, layout.height);
  const svg = buildExportSvg(options.scene, options.frame, width);
  const blob = new Blob([svg], { type: 'image/svg+xml' });
  const filename = `${fileStem(options.scene)}-frame-${options.frame}.svg`;
  (options.deps ?? browserDeps()).save(blob, filename);
  return { filename, blob, svg };
}

/** The current frame as a PNG: the scene drawn onto a canvas, then the caption drawn as text. */
export async function exportPng(options: FrameExportOptions): Promise<ExportResult> {
  requireEnabled(options.enabled);
  const deps = options.deps ?? browserDeps();
  const width = widthOf(options);
  const layout = layoutFrame(options.scene, options.frame, width);
  checkLimits(1, width, layout.height);
  const canvas = deps.createCanvas(layout.width, layout.height);
  await drawFrame(canvas, deps, options.scene, options.frame, layout);
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
  if (blob === null) throw new ExportRefused('Export refused: the browser could not make a PNG.');
  const filename = `${fileStem(options.scene)}-frame-${options.frame}.png`;
  deps.save(blob, filename);
  return { filename, blob };
}

/** The replay from `first` to `last` as a WebM, recorded with the browser's MediaRecorder from a canvas at `fps` frames a second. */
export async function exportWebm(options: ReplayExportOptions): Promise<ExportResult> {
  requireEnabled(options.enabled);
  const deps = options.deps ?? browserDeps();
  const width = widthOf(options);
  const count = replayCount(options.first, options.last);
  const fps = fpsOf(options.fps);
  const layout = layoutFrame(options.scene, options.first, width);
  checkLimits(count, width, layout.height);
  const frames = frameNumbers(options.first, count);
  const Recorder = deps.MediaRecorder;
  if (Recorder === undefined || !Recorder.isTypeSupported('video/webm')) throw new ExportRefused('Export refused: this browser cannot record WebM video.');
  const canvas = deps.createCanvas(layout.width, layout.height);
  if (canvas.captureStream === undefined) throw new ExportRefused('Export refused: this browser cannot record from a canvas.');
  const chunks: Blob[] = [];
  const recorder = new Recorder(canvas.captureStream(fps), { mimeType: 'video/webm' });
  recorder.ondataavailable = (event) => chunks.push(event.data);
  const stopped = new Promise<void>((resolve) => {
    recorder.onstop = () => resolve();
  });
  recorder.start();
  for (const frame of frames) {
    await drawFrame(canvas, deps, options.scene, frame, layoutFrame(options.scene, frame, width));
    await deps.sleep(1000 / fps);
  }
  recorder.stop();
  await stopped;
  const blob = new Blob(chunks, { type: 'video/webm' });
  const filename = `${fileStem(options.scene)}-frames-${options.first}-${options.last}.webm`;
  deps.save(blob, filename);
  return { filename, blob };
}

/** The replay from `first` to `last` as a looping GIF, written by gif.ts. */
export async function exportGif(options: ReplayExportOptions): Promise<ExportResult> {
  requireEnabled(options.enabled);
  const deps = options.deps ?? browserDeps();
  const width = widthOf(options);
  const count = replayCount(options.first, options.last);
  const fps = fpsOf(options.fps);
  const layout = layoutFrame(options.scene, options.first, width);
  checkLimits(count, width, layout.height);
  if (layout.width * layout.height * count > MAX_GIF_TOTAL_PIXELS) {
    throw new ExportRefused(
      `Export refused: a GIF of ${count} frames at ${layout.width} by ${layout.height} pixels is over the limit of ${MAX_GIF_TOTAL_PIXELS} pixels in all.`,
    );
  }
  const frames = frameNumbers(options.first, count);
  const canvas = deps.createCanvas(layout.width, layout.height);
  const gifFrames: GifFrame[] = [];
  for (const frame of frames) {
    const ctx = await drawFrame(canvas, deps, options.scene, frame, layoutFrame(options.scene, frame, width));
    gifFrames.push({ indexes: quantizeFrame(ctx.getImageData(0, 0, layout.width, layout.height).data) });
  }
  const bytes = encodeGif(gifFrames, { width: layout.width, height: layout.height, delayCentiseconds: Math.max(1, Math.round(100 / fps)) });
  const blob = new Blob([bytes as unknown as BlobPart], { type: 'image/gif' });
  const filename = `${fileStem(options.scene)}-frames-${options.first}-${options.last}.gif`;
  deps.save(blob, filename);
  return { filename, blob };
}
