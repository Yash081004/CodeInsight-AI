import { useCallback, useEffect, useRef } from 'react';

/**
 * Returns a runner that ignores stale responses when a newer request starts.
 */
export function useAbortableRequest() {
  const seqRef = useRef(0);

  useEffect(() => () => {
    seqRef.current += 1;
  }, []);

  const run = useCallback(async (asyncFn) => {
    const id = ++seqRef.current;
    const result = await asyncFn();
    if (id !== seqRef.current) {
      return { stale: true, data: null };
    }
    return { stale: false, data: result };
  }, []);

  const cancel = useCallback(() => {
    seqRef.current += 1;
  }, []);

  return { run, cancel };
}
