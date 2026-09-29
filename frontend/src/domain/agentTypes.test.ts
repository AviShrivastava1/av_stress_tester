import { describe, expect, it } from 'vitest';
import { describeAgentType } from './agentTypes';

describe('describeAgentType', () => {
  it('names the three types the backend physics defines', () => {
    expect(describeAgentType(1)).toEqual({ kind: 'vehicle', label: 'vehicle' });
    expect(describeAgentType(2)).toEqual({ kind: 'pedestrian', label: 'pedestrian' });
    expect(describeAgentType(3)).toEqual({ kind: 'cyclist', label: 'cyclist' });
  });

  it('reports every other code as unknown, keeping the code', () => {
    expect(describeAgentType(0)).toEqual({ kind: 'unknown', label: 'unknown type (code 0)' });
    expect(describeAgentType(4)).toEqual({ kind: 'unknown', label: 'unknown type (code 4)' });
    expect(describeAgentType(7)).toEqual({ kind: 'unknown', label: 'unknown type (code 7)' });
    expect(describeAgentType(null)).toEqual({ kind: 'unknown', label: 'unknown type' });
  });
});
