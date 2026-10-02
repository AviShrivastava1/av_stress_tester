import { describe, expect, it } from 'vitest';
import { describeError } from './errors';
import { RequestTimeoutError } from './request';

describe('describeError', () => {
  it('says a timed-out request is slow, not unreachable', () => {
    const message = describeError(new RequestTimeoutError());
    expect(message).toContain('longer than expected');
    expect(message).not.toContain('Could not reach');
  });
});
