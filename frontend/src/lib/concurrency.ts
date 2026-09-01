/**
 * Run `task` over every item with at most `limit` tasks in flight at once.
 *
 * Resolves once every item has settled and never rejects: a task that throws
 * is reported through `onError(item, error, index)` and the pool keeps going.
 * Results come back in input order, with `undefined` in the slots of items
 * whose task threw.
 *
 * Used for batch uploads — firing one `fetch` per file with an unbounded
 * `Promise.all` exhausts the browser's request pool (`ERR_INSUFFICIENT_RESOURCES`)
 * and swamps the API long before the batch finishes; a small fixed pool keeps
 * throughput high without either failure mode.
 */
export async function mapWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  task: (item: T, index: number) => Promise<R>,
  onError?: (item: T, error: unknown, index: number) => void,
): Promise<(R | undefined)[]> {
  const results: (R | undefined)[] = new Array(items.length);
  let cursor = 0;

  async function worker(): Promise<void> {
    while (cursor < items.length) {
      const index = cursor++;
      try {
        results[index] = await task(items[index], index);
      } catch (error) {
        onError?.(items[index], error, index);
      }
    }
  }

  const poolSize = Math.max(1, Math.min(limit, items.length));
  await Promise.all(Array.from({ length: poolSize }, worker));
  return results;
}
