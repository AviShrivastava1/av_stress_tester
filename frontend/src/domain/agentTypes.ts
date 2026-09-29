/**
 * WOMD object types, as the backend's own physics uses them
 * (src/physics/simulator.py: TYPE_VEHICLE = 1, TYPE_PEDESTRIAN = 2,
 * TYPE_CYCLIST = 3).
 *
 * Hand-written for the same reason as outcomes.ts: the schema types `agent_type` as
 * `int | null`, so the codes are not in the generated types. Any other value —
 * 0 (unset), 4 (other), null, or a code this list has never seen — is reported as
 * unknown WITH its code, never mapped to the nearest known type.
 */
export type AgentKind = 'vehicle' | 'pedestrian' | 'cyclist' | 'unknown';

export interface AgentTypeDescription {
  kind: AgentKind;
  label: string;
}

export function describeAgentType(code: number | null | undefined): AgentTypeDescription {
  switch (code) {
    case 1:
      return { kind: 'vehicle', label: 'vehicle' };
    case 2:
      return { kind: 'pedestrian', label: 'pedestrian' };
    case 3:
      return { kind: 'cyclist', label: 'cyclist' };
    default:
      return {
        kind: 'unknown',
        label: code === null || code === undefined ? 'unknown type' : `unknown type (code ${code})`,
      };
  }
}
