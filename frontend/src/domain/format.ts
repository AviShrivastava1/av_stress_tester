/** Preserve small nonzero results instead of presenting them as zero. */
export function formatMagnitude(value: number, digits = 3): string {
  if (value !== 0 && Math.abs(value) < 10 ** -digits) return value.toExponential(2);
  return value.toFixed(digits);
}
