import { useMemo } from 'react';
import { collisionAnatomy, POSITION_PRECISION_M, type Anatomy } from '../../scene/anatomy';
import type { DrawnTrack } from '../../scene/sceneModel';

/*
 * The words under the replay about the frame before first contact. Every figure in them is read from the stored data and
 * computed in the browser (scene/anatomy.ts); no number is written into the copy. What backs each sentence:
 *
 *   title "Gap before first contact"
 *       the shortest distance between the SDC's box and the perturbed challenger's box on the frame before the collision
 *       frame (collisionAnatomy: collisionFrame - 1). The collision frame is the stored collision_timestep, the first frame
 *       at which src/danger/collision_detector.py check_collision_trajectory reported contact
 *   "The boxes were {gap} m apart on frame {n}, the frame before first contact on frame {c}."
 *       boxGap on the two footprints at frame n, each drawn from the stored position, heading and box size for that frame
 *       (observationAt: the per-frame sizes when the export wrote them, otherwise the scalar size at the first observed
 *       frame, exactly as the scene draws them). {gap} is that distance in metres, the units of the stored coordinates
 *   "…less than a millimetre apart…"
 *       the gap is below POSITION_PRECISION_M, the precision to which contact is decided on the stored positions, so it is
 *       not given as a number
 *   "The boxes already touch or overlap on frame {n}, as drawn from the stored sizes, although first contact is reported
 *    on frame {c}."
 *       boxGap is 0 on frame n: the drawn boxes touch or overlap there, while the search's check first reported contact on
 *       frame c. Nothing is said about why; no overlap is drawn or measured
 *   "Worked out in this browser from the stored positions and box sizes of the SDC and the perturbed challenger; the search
 *    does not store it."
 *       the search stores collision_timestep, the perturbation and the perturbed path (src/scoring/db.py scenario_scores,
 *       src/scoring/export_geometry.py), not a gap; this figure is computed by boxGap from the served tracks
 *   "Contact is decided at the precision of the stored positions, up to about a millimetre at the coordinates in this
 *    data, so no finer figure is claimed."
 *       the same sentence the box-size notes carry (pages/detail/ScenePanel.tsx BOX_SIZE_NOTE_*): the corners are float64
 *       and the exported path is written with repr(float(...)), which round-trips binary64 exactly, but the positions are
 *       measurements, so the data supports about a millimetre
 *   "The SDC or the perturbed challenger is not observed before frame {c}, so there is no earlier frame to measure."
 *       one of the two tracks has no timestep before the collision frame (collisionAnatomy: no_earlier_frame), which is how
 *       a collision on the first frame, or on the first frame of one agent, looks in the data
 *   "The SDC and the perturbed challenger were not both observed on frame {n}, so there is no gap to measure there."
 *       one track has no vertex at frame n (a validity gap): the frame before is not replaced by an earlier one
 *   "No box size is stored for both agents on frame {n}, so the gap is not shown."
 *       observationAt returns no size for one of them at frame n (a missing or unusable length or width), and a box is not
 *       drawn from a guessed size (scene/geometry.ts hasDimensions)
 */

/** The sentence saying how the figure was got, shown under it and copied into the text brief (pages/detail/brief.ts). */
export const DERIVED_NOTE =
  'Worked out in this browser from the stored positions and box sizes of the SDC and the perturbed challenger; the search does not store it. ' +
  'Contact is decided at the precision of the stored positions, up to about a millimetre at the coordinates in this data, so no finer figure is claimed.';

export function anatomyWords(anatomy: Anatomy): { derived: boolean; text: string } {
  switch (anatomy.kind) {
    case 'gap': {
      if (anatomy.gap === 0) {
        return {
          derived: true,
          text: `The boxes already touch or overlap on frame ${anatomy.frame}, as drawn from the stored sizes, although first contact is reported on frame ${anatomy.collisionFrame}.`,
        };
      }
      const apart = anatomy.gap < POSITION_PRECISION_M ? 'less than a millimetre' : `${anatomy.gap.toFixed(3)} m`;
      return {
        derived: true,
        text: `The boxes were ${apart} apart on frame ${anatomy.frame}, the frame before first contact on frame ${anatomy.collisionFrame}.`,
      };
    }
    case 'no_earlier_frame':
      return {
        derived: false,
        text: `The SDC or the perturbed challenger is not observed before frame ${anatomy.collisionFrame}, so there is no earlier frame to measure.`,
      };
    case 'not_observed':
      return {
        derived: false,
        text: `The SDC and the perturbed challenger were not both observed on frame ${anatomy.frame}, so there is no gap to measure there.`,
      };
    case 'no_size':
      return {
        derived: false,
        text: `No box size is stored for both agents on frame ${anatomy.frame}, so the gap is not shown.`,
      };
  }
}

export interface CollisionAnatomyProps {
  drawn: DrawnTrack[];
  /** The collision frame of the perturbed run; the caller passes it only when that run is drawn. */
  collisionFrame: number;
}

export function CollisionAnatomy({ drawn, collisionFrame }: CollisionAnatomyProps) {
  const sdc = drawn.find((d) => d.role === 'sdc');
  const challenger = drawn.find((d) => d.role === 'challenger_perturbed');
  const anatomy = useMemo(
    () => (sdc && challenger ? collisionAnatomy(sdc.track, challenger.track, collisionFrame) : null),
    [sdc, challenger, collisionFrame],
  );
  if (anatomy === null) return null;

  const { text, derived } = anatomyWords(anatomy);
  return (
    <section className="anatomy" aria-labelledby="anatomy-title">
      <h3 id="anatomy-title" className="anatomy-title">Gap before first contact</h3>
      <p className="note">{text}</p>
      {derived && <p className="note">{DERIVED_NOTE}</p>}
    </section>
  );
}
