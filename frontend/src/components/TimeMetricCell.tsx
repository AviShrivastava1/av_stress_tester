import { PET_OVERLAP_TITLE, isPetOverlap, readTimeMetric } from '../domain/metrics';

const NO_INTERACTION_MEANING = {
  ttc: 'No finite collision time predicted for an SDC pair by the constant-velocity model. A closing pair can still miss.',
  pet: 'No shared conflict zone with the SDC (999.0 sentinel, not a measurement).',
} as const;

export function TimeMetricCell({ value, metric }: { value: number | null; metric: 'ttc' | 'pet' }) {
  const reading = readTimeMetric(value);
  switch (reading.kind) {
    case 'missing':
      return <span className="muted">—</span>;
    case 'no_interaction':
      return (
        <span className="muted" title={NO_INTERACTION_MEANING[metric]}>
          none
        </span>
      );
    case 'measured':
      return (
        <span>
          {reading.seconds.toFixed(2)} s
          {/* Zero or negative PET is real: both agents were in the conflict zone on a common frame (see PET_OVERLAP_TITLE). */}
          {metric === 'pet' && isPetOverlap(reading.seconds) && (
            <span className="tag" title={PET_OVERLAP_TITLE}>
              overlap
            </span>
          )}
        </span>
      );
  }
}
