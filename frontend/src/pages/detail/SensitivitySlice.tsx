import { useState } from 'react';
import type { SensitivityResponse, SensitivitySlice } from '../../api/client';
import { RequestError } from '../../components/RequestError';
import { useSensitivitySlices } from '../../api/scenarios';
import { formatMagnitude } from '../../domain/format';

/*
 * The words in "Sensitivity slice of this scenario", and what backs each. Every value is read from GET /scenarios/{id}/slice
 * (src/api/routes.py get_slice; src/api/sensitivity.py is the allow-list; src/api/models.py SensitivitySlice and SensitivityResponse
 * describe each field); none is written here. A value the API sends as null is "Not recorded". The section is closed until opened and
 * asks for nothing before that; its note is drawn once it is opened. It sits below the Provenance panel, not among the panels of the
 * headline figure.
 *
 *   "A slice of a four-dimensional search."
 *       the perturbation has four components (DELTA_LABELS / LINEAR_DELTA_LABELS in src/api/models.py); sensitivity_slices holds two
 *       of them on the axes (axis_x, axis_y) and all four of the stored perturbation (held_delta)
 *   "Two components of the stored perturbation vary over the grid below; the other two are held at the stored values."
 *       src/scoring/sensitivity.py record_sensitivity_slice: every cell is the stored perturbation with the two axis components replaced
 *       by that cell's pair of values; the other two keep held_delta (tests/test_sensitivity.py checks this on the evaluated perturbations)
 *   "Each cell shows whether the replay collides with that pair of values, and the weighted norm of the whole perturbation."
 *       `collides` is the exact oriented-box check (collision_detector.check_collision_trajectory, the check the search used) on the
 *       replay of that cell's perturbation; `weighted_norm` is PerturbationSpace.weighted_norm of it: a weighted norm over components of
 *       mixed units, the same quantity as Min perturbation
 *   "A cell with no collision shows only that this one perturbation did not collide: collisions can lie between the cells and in the
 *    held components, and a finite grid is not a proof."
 *       the grid is finite (2 to 101 values per axis, the table's CHECKs) and two of four components are fixed; nothing about the
 *       cells between or the held components is evaluated. Not a claim about any other perturbation
 *   the legend: "Hatched cell: the replay collides.", "Plain cell: it does not.", "Dotted cell: not recorded.", "Ring: the stored perturbation."
 *       the meaning is in the words and in the pattern of the cell (hatched, plain, dotted); colour is a second cue. The ring is drawn at
 *       (stored_x_index, stored_y_index), found by the API as the index whose value equals the stored component exactly (the offline pass
 *       puts the stored value on the grid)
 *   the axes' names and units
 *       axis_x_label and axis_y_label, derived by the API from that slice's OWN delta_parameterization (DELTA_LABELS_BY_PARAMETERIZATION);
 *       a component is never given a unit its model does not have
 *   the summary of each slice, "{x} against {y}, {n} by {m}"
 *       the two axes named with axis_x_label and axis_y_label, and the grid's size: x_values.length by y_values.length. Each slice is in a
 *       closed details of its own and nothing of it (count, drawing, legend, held components, table) exists until it is opened: a grid of
 *       101 by 101 is 10 201 cells, and the API lists up to twelve slices. A slice under another or an unrecorded model, or with no
 *       grid recorded, keeps one line saying so and is not in a details
 *   "Table of the slice"
 *       every cell as text for a reader who cannot see the drawing: the verdict in words, the weighted norm and, for the stored cell, "stored"
 *   "N of M cells collide"
 *       a count over `collides` (true entries over all entries)
 *   "Held at the stored values:"
 *       held_delta, the two components that are not on an axis, named with delta_labels (all four names, with units, under the slice's
 *       own model; "component N" if they are not the length of held_delta)
 *   "Under a different or unrecorded model; its figures are not shown against the stored result’s."
 *       the slice's delta_parameterization is null or is not the stored result's (stored_delta_parameterization): no number of such a
 *       slice is shown, and none is compared
 *   "Not every recorded slice is listed here; those shown are the first in the order above."
 *       SensitivityResponse.truncated (the API lists at most MAX_SLICES, src/api/sensitivity.py); no number is typed here
 *   "No slice is recorded for this scenario."
 *       slices is empty: none drawn, the table is not in this database, the API role cannot read it, or the slices drawn were around a
 *       result that has since been replaced. Never read as a result
 */

const NOT_RECORDED = 'Not recorded';
const SLICE_CELL = 12;

function comparable(s: SensitivitySlice, stored: string | null | undefined): boolean {
  return s.delta_parameterization !== null && s.delta_parameterization !== undefined && s.delta_parameterization === stored;
}

const fmt = (value: number) => formatMagnitude(value, 4);

function verdictText(collides: boolean | null | undefined): string {
  return collides === true ? 'collision' : collides === false ? 'no collision' : NOT_RECORDED;
}

function replayText(collides: boolean | null | undefined): string {
  return collides === true ? 'collides' : collides === false ? 'does not collide' : 'has no recorded verdict';
}

/**
 * A slice under the stored result's model with its grid recorded: what One has already checked, handed down so that nothing below checks it
 * again (and so there is no text below for a grid that is not there).
 */
interface Recorded {
  s: SensitivitySlice;
  xs: number[];
  ys: number[];
  hits: Array<boolean | null>;
  axisX: number;
  axisY: number;
  xName: string;
  yName: string;
}

function Grid({ r }: { r: Recorded }) {
  const { s, xs, ys, hits, xName, yName } = r;
  const nx = xs.length;
  const ny = ys.length;
  const count = hits.filter((h) => h === true).length;
  const sx = s.stored_x_index;
  const sy = s.stored_y_index;
  const storedKnown = sx !== null && sx !== undefined && sy !== null && sy !== undefined;
  const storedText = storedKnown
    ? `the stored perturbation is at ${xName} ${fmt(xs[sx]!)} and ${yName} ${fmt(ys[sy]!)}, where the replay ${replayText(hits[sy * nx + sx])}`
    : 'the stored value is not one of the grid’s values';
  const name = `Collision verdicts over ${xName} against ${yName}: ${count} of ${hits.length} cells collide; ${storedText}`;
  return (
    <figure className="sens-figure" aria-label={name}>
      <svg className="sens-svg" viewBox={`0 0 ${nx * SLICE_CELL} ${ny * SLICE_CELL}`} preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <pattern id={`sens-hatch-${s.slice_id ?? 'x'}`} width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="4" height="4" className="sens-hit-ground" />
            <line x1="0" y1="0" x2="0" y2="4" className="sens-hit-line" />
          </pattern>
        </defs>
        {hits.map((h, k) => {
          const i = k % nx;
          const j = Math.floor(k / nx);
          const kind = h === true ? 'sens-hit' : h === false ? 'sens-miss' : 'sens-unknown';
          return (
            <rect
              key={k}
              className={`sens-cell ${kind}`}
              x={i * SLICE_CELL}
              y={(ny - 1 - j) * SLICE_CELL}
              width={SLICE_CELL}
              height={SLICE_CELL}
              fill={h === true ? `url(#sens-hatch-${s.slice_id ?? 'x'})` : undefined}
            />
          );
        })}
        {storedKnown && (
          <circle className="sens-stored" cx={sx * SLICE_CELL + SLICE_CELL / 2} cy={(ny - 1 - sy) * SLICE_CELL + SLICE_CELL / 2} r={SLICE_CELL * 0.9} />
        )}
      </svg>
      <figcaption className="sens-scale">
        <span>{xName}: {fmt(xs[0]!)} to {fmt(xs[nx - 1]!)} (left to right)</span>
        <span>{yName}: {fmt(ys[0]!)} to {fmt(ys[ny - 1]!)} (bottom to top)</span>
      </figcaption>
    </figure>
  );
}

function Legend({ storedKnown }: { storedKnown: boolean }) {
  return (
    <ul className="sens-legend">
      <li>Hatched cell: the replay collides.</li>
      <li>Plain cell: it does not.</li>
      <li>Dotted cell: not recorded.</li>
      <li>{storedKnown ? 'Ring: the stored perturbation.' : 'The stored value is not one of the grid’s values, so no cell is ringed.'}</li>
    </ul>
  );
}

function SliceTable({ r }: { r: Recorded }) {
  const [open, setOpen] = useState(false);
  const { s, xs, ys, hits, xName, yName } = r;
  const nx = xs.length;
  return (
    <details className="sens-table" onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>Table of the slice</summary>
      {open && (
        <table className="delta-table">
          <caption>{yName} down the side, {xName} across the top; each cell gives the verdict and the weighted norm.</caption>
          <thead>
            <tr>
              <th scope="col">{yName} \ {xName}</th>
              {xs.map((x, i) => (
                <th scope="col" className="num" key={i} title={String(x)}>{fmt(x)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...ys.keys()].reverse().map((j) => (
              <tr key={j}>
                <th scope="row" title={String(ys[j])}>{fmt(ys[j]!)}</th>
                {xs.map((_, i) => {
                  const k = j * nx + i;
                  const norm = s.weighted_norm?.[k];
                  const stored = i === s.stored_x_index && j === s.stored_y_index;
                  return (
                    <td className="num" key={i}>
                      {verdictText(hits[k])}
                      {norm === null || norm === undefined ? '' : `, ${formatMagnitude(norm)}`}
                      {stored ? ' (stored)' : ''}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </details>
  );
}

function labelOf(s: SensitivitySlice, i: number): string {
  const labels = s.delta_labels;
  return labels !== null && labels !== undefined && labels.length === s.held_delta?.length ? labels[i]! : `component ${i + 1}`;
}

/** What a slice shows once it is opened: the count, the drawing, the legend, the held components and the table. Nothing of it exists while the slice is closed. */
function Drawn({ r }: { r: Recorded }) {
  const { s, hits } = r;
  const held = s.held_delta;
  const total = hits.length;
  const count = hits.filter((h) => h === true).length;
  const storedKnown = s.stored_x_index !== null && s.stored_x_index !== undefined && s.stored_y_index !== null && s.stored_y_index !== undefined;
  return (
    <>
      {total > 0 && <p className="note">{count} of {total} cells collide.</p>}
      <Grid r={r} />
      <Legend storedKnown={storedKnown} />
      {held && (
        <p className="note sens-held">
          Held at the stored values:{' '}
          {held
            .map((value, i) => [i, value] as const)
            .filter(([i]) => i !== r.axisX && i !== r.axisY)
            .map(([i, value]) => `${labelOf(s, i)} = ${fmt(value)}`)
            .join('; ')}
          .
        </p>
      )}
      <SliceTable r={r} />
    </>
  );
}

/** One slice in a closed details of its own: a grid of 101 by 101 is 10 201 cells, and twelve slices are not drawn until a reader opens one. */
function OpenableSlice({ r }: { r: Recorded }) {
  const [open, setOpen] = useState(false);
  return (
    <details className="sens-slice" onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>{`${r.xName} against ${r.yName}, ${r.xs.length} by ${r.ys.length}`}</summary>
      {open && <Drawn r={r} />}
    </details>
  );
}

function One({ s, stored }: { s: SensitivitySlice; stored: string | null | undefined }) {
  const xName = s.axis_x_label ?? (s.axis_x === null || s.axis_x === undefined ? 'a component' : `component ${s.axis_x + 1}`);
  const yName = s.axis_y_label ?? (s.axis_y === null || s.axis_y === undefined ? 'a component' : `component ${s.axis_y + 1}`);
  if (!comparable(s, stored)) {
    return (
      <div className="sens-slice">
        <h4>{xName} against {yName}</h4>
        <p className="note">Under a different or unrecorded model; its figures are not shown against the stored result’s.</p>
      </div>
    );
  }
  const { x_values: xs, y_values: ys, collides: hits, axis_x: axisX, axis_y: axisY } = s;
  if (!xs || !ys || !hits || axisX === null || axisX === undefined || axisY === null || axisY === undefined) {
    return (
      <div className="sens-slice">
        <h4>{xName} against {yName}</h4>
        <p className="note">The grid of this slice is not recorded.</p>
      </div>
    );
  }
  return <OpenableSlice r={{ s, xs, ys, hits, axisX, axisY, xName, yName }} />;
}

function Body({ data }: { data: SensitivityResponse }) {
  const slices = data.slices ?? [];
  if (slices.length === 0) return <p className="note">No slice is recorded for this scenario.</p>;
  return (
    <>
      {slices.map((s, i) => (
        <One key={s.slice_id ?? `slice-${i}`} s={s} stored={data.stored_delta_parameterization} />
      ))}
      {data.truncated === true && (
        <p className="note">Not every recorded slice is listed here; those shown are the first in the order above.</p>
      )}
    </>
  );
}

/** Once opened it stays "opened" for the query's sake: closing and opening again shows what was fetched and asks for nothing. */
export function SensitivitySliceSection({ scenarioId }: { scenarioId: string }) {
  const [opened, setOpened] = useState(false);
  const query = useSensitivitySlices(scenarioId, { enabled: opened });
  return (
    <details className="provenance sensitivity" onToggle={(e) => e.currentTarget.open && setOpened(true)}>
      <summary>Sensitivity slice of this scenario</summary>
      {opened && (
        <p className="note">
          A slice of a four-dimensional search. Two components of the stored perturbation vary over the grid below; the other two are held at
          the stored values. Each cell shows whether the replay collides with that pair of values, and the weighted norm of the whole
          perturbation. A cell with no collision shows only that this one perturbation did not collide: collisions can lie between the cells
          and in the held components, and a finite grid is not a proof.
        </p>
      )}
      {opened && query.isPending && <p className="status" role="status">Loading the slice…</p>}
      {opened && query.isError && <RequestError error={query.error} retry={() => query.refetch()} retrying={query.isFetching} />}
      {opened && query.isSuccess && <Body data={query.data} />}
    </details>
  );
}
