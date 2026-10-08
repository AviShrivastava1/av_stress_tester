import { memo, useMemo } from 'react';
import { describeAgentType } from '../domain/agentTypes';
import {
  footprint,
  frontCentre,
  largestDimension,
  niceScaleLength,
  observationAt,
  padBounds,
  segments,
  sizeSummary,
  toSvg,
  trackBounds,
  viewBox,
  type Observation,
  type Point,
  type TrackGeometry,
} from './geometry';
import type { DrawnTrack, Role } from './sceneModel';

const ROLE_LABEL: Record<Role, string> = {
  other: 'agent',
  sdc: 'SDC',
  challenger_logged: 'challenger (logged)',
  challenger_perturbed: 'challenger (perturbed)',
};

// Margin around the focused tracks when none of them has usable dimensions.
const DEFAULT_MARGIN_M = 3;

function svgPoints(points: readonly Point[]): string {
  return points.map((p) => toSvg(p).join(',')).join(' ');
}

/**
 * The sizes a track is drawn with, for a title. A track drawn from the scalars keeps its
 * one-decimal size. A track drawn from per-frame sizes gives each dimension as one value only
 * if it never varies, otherwise as the range its boxes take, to two decimals: the differences
 * between frames are a few centimetres, which one decimal would hide.
 */
function describeSize(track: TrackGeometry): string {
  const s = sizeSummary(track);
  if (s.kind === 'none') return 'dimensions unknown';
  if (s.kind === 'scalar') return `${s.length.toFixed(1)} × ${s.width.toFixed(1)} m`;
  const span = (r: { min: number; max: number }) =>
    r.min === r.max ? r.min.toFixed(2) : `${r.min.toFixed(2)}–${r.max.toFixed(2)}`;
  return s.length.min === s.length.max && s.width.min === s.width.max
    ? `${span(s.length)} × ${span(s.width)} m`
    : `length ${span(s.length)} m, width ${span(s.width)} m`;
}

function describeBase(d: DrawnTrack): string {
  const t = d.track;
  return `${ROLE_LABEL[d.role]} · agent ${t.agent_idx} · ${describeAgentType(t.agent_type).label}`;
}

/** The static path title: the sizes the track is drawn with, which may vary by frame. */
function describeTrack(d: DrawnTrack): string {
  return `${describeBase(d)} · ${describeSize(d.track)}`;
}

/** The footprint title: for a per-frame track, the size drawn at THIS frame. */
function describeFootprint(d: DrawnTrack, o: Observation, frame: number): string {
  if (o.source === 'scalar') return describeTrack(d);
  return o.size === null
    ? `${describeBase(d)} · size not recorded at frame ${frame}`
    : `${describeBase(d)} · ${o.size.length.toFixed(2)} × ${o.size.width.toFixed(2)} m at frame ${frame}`;
}

export interface SceneViewProps {
  drawn: DrawnTrack[];
  /** The tracks the view is framed on (all of `drawn`, or the interaction subset). */
  focus: DrawnTrack[];
  /** The frame whose footprints are drawn; null draws paths only. */
  frame: number | null;
  /**
   * The first colliding frame, or null. Only ever set when the perturbed path is
   * drawn; the colliding pair is marked on this frame and no other.
   */
  collisionFrame: number | null;
}

interface Layout {
  vb: { x: number; y: number; width: number; height: number };
  dotRadius: number;
  fontSize: number;
}

function computeLayout(drawn: DrawnTrack[], focus: DrawnTrack[]): Layout | null {
  const raw = trackBounds(focus.map((d) => d.track)) ?? trackBounds(drawn.map((d) => d.track));
  if (raw === null) return null;
  // The largest box any focused track can show at ANY frame: the layout does not change with the frame.
  const largest = Math.max(0, ...focus.map((d) => largestDimension(d.track)));
  const span0 = Math.max(raw.maxX - raw.minX, raw.maxY - raw.minY);
  const vb = viewBox(padBounds(raw, (largest || DEFAULT_MARGIN_M) + 0.05 * span0));
  const span = Math.max(vb.width, vb.height);
  return { vb, dotRadius: span * 0.008, fontSize: span * 0.022 };
}

/**
 * Everything that does not change with the frame: paths and the scale bar. Memoized
 * on the scene and its framing, so playback re-renders only the footprint layer.
 */
const StaticLayer = memo(function StaticLayer({ drawn, layout }: { drawn: DrawnTrack[]; layout: Layout }) {
  const { vb, dotRadius, fontSize } = layout;
  const span = Math.max(vb.width, vb.height);
  const scaleLength = niceScaleLength(vb.width);
  const scaleX = vb.x + span * 0.04;
  const scaleY = vb.y + vb.height - span * 0.04;
  return (
    <>
      {drawn.map((d) => (
        <g
          key={d.key}
          className={`track role-${d.role} kind-${describeAgentType(d.track.agent_type).kind}`}
          data-role={d.role}
          data-agent={d.track.agent_idx}
        >
          <title>{describeTrack(d)}</title>
          {segments(d.track).map((run, i) =>
            run.length === 1 ? (
              <circle key={i} className="path-point" cx={toSvg(run[0]!)[0]} cy={toSvg(run[0]!)[1]} r={dotRadius / 2} />
            ) : (
              <polyline key={i} className="path" points={svgPoints(run)} vectorEffect="non-scaling-stroke" />
            ),
          )}
        </g>
      ))}
      <g className="scale-bar" data-scale-metres={scaleLength}>
        <line x1={scaleX} y1={scaleY} x2={scaleX + scaleLength} y2={scaleY} vectorEffect="non-scaling-stroke" />
        <text x={scaleX} y={scaleY - fontSize * 0.5} fontSize={fontSize}>
          {scaleLength} m
        </text>
      </g>
    </>
  );
});

/** Footprints at one frame: the only part of the scene that changes during playback. */
function FootprintLayer({
  drawn,
  frame,
  collisionFrame,
  dotRadius,
}: {
  drawn: DrawnTrack[];
  frame: number;
  collisionFrame: number | null;
  dotRadius: number;
}) {
  const atCollision = collisionFrame !== null && frame === collisionFrame;
  const hasPerturbed = drawn.some((d) => d.role === 'challenger_perturbed');
  return (
    <>
      {drawn.map((d) => {
        const observation = observationAt(d.track, frame);
        if (observation === null) return null;
        const { pose, size } = observation;
        const colliding = atCollision && (d.role === 'sdc' || d.role === 'challenger_perturbed');
        const ghost = d.role === 'challenger_logged' && hasPerturbed;
        const className = [
          'footprint',
          `role-${d.role}`,
          `kind-${describeAgentType(d.track.agent_type).kind}`,
          colliding ? 'colliding' : '',
          ghost ? 'ghost' : '',
        ]
          .filter(Boolean)
          .join(' ');

        if (size === null) {
          const [cx, cy] = toSvg([pose.x, pose.y]);
          return (
            <circle
              key={`fp-${d.key}`}
              className={className}
              data-footprint={d.role}
              data-shape="dot"
              data-colliding={colliding || undefined}
              cx={cx}
              cy={cy}
              r={dotRadius}
              vectorEffect="non-scaling-stroke"
            >
              <title>{describeFootprint(d, observation, frame)}</title>
            </circle>
          );
        }
        const { length, width } = size;
        return (
          <g
            key={`fp-${d.key}`}
            className={className}
            data-footprint={d.role}
            data-shape="box"
            data-colliding={colliding || undefined}
          >
            <title>{describeFootprint(d, observation, frame)}</title>
            <polygon
              points={svgPoints(footprint(pose, length, width))}
              vectorEffect="non-scaling-stroke"
            />
            <line
              className="heading"
              data-heading-tick=""
              x1={toSvg([pose.x, pose.y])[0]}
              y1={toSvg([pose.x, pose.y])[1]}
              x2={toSvg(frontCentre(pose, length))[0]}
              y2={toSvg(frontCentre(pose, length))[1]}
              vectorEffect="non-scaling-stroke"
            />
          </g>
        );
      })}
    </>
  );
}

/**
 * One scene: every path, plus footprints at one frame. Geometry decisions all live
 * in ./geometry; this only turns them into SVG.
 */
export function SceneView({ drawn, focus, frame, collisionFrame }: SceneViewProps) {
  const layout = useMemo(() => computeLayout(drawn, focus), [drawn, focus]);
  if (layout === null) return null;
  const { vb } = layout;

  return (
    <svg
      className="scene"
      viewBox={`${vb.x} ${vb.y} ${vb.width} ${vb.height}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={frame === null ? 'Scene paths' : `Scene paths, positions at frame ${frame}`}
    >
      <StaticLayer drawn={drawn} layout={layout} />
      {frame !== null && (
        <FootprintLayer drawn={drawn} frame={frame} collisionFrame={collisionFrame} dotRadius={layout.dotRadius} />
      )}
    </svg>
  );
}
