import { describe, expect, it } from "vitest";
import { mapWithConcurrency } from "./concurrency";

/** A promise that resolves only when `release()` is called. */
function deferred<T = void>() {
  let release!: (value: T) => void;
  const promise = new Promise<T>((resolve) => {
    release = resolve;
  });
  return { promise, release };
}

describe("mapWithConcurrency", () => {
  it("processes every item and returns results in input order", async () => {
    const results = await mapWithConcurrency([1, 2, 3, 4, 5], 2, async (n) => n * 10);
    expect(results).toEqual([10, 20, 30, 40, 50]);
  });

  it("never runs more than `limit` tasks at once", async () => {
    let inFlight = 0;
    let peak = 0;
    const gates = Array.from({ length: 20 }, () => deferred());

    const run = mapWithConcurrency(gates, 4, async (gate) => {
      inFlight++;
      peak = Math.max(peak, inFlight);
      await gate.promise;
      inFlight--;
    });

    // Let the pool saturate, then release gates one wave at a time.
    for (let i = 0; i < gates.length; i++) {
      await Promise.resolve();
      gates[i].release();
    }
    await run;

    expect(peak).toBe(4);
  });

  it("keeps going after a task throws and reports it through onError", async () => {
    const seen: { value: number; message: string }[] = [];
    const results = await mapWithConcurrency(
      [1, 2, 3, 4],
      2,
      async (n) => {
        if (n % 2 === 0) throw new Error(`boom ${n}`);
        return n;
      },
      (value, error) => seen.push({ value, message: (error as Error).message }),
    );

    expect(results).toEqual([1, undefined, 3, undefined]);
    expect(seen).toEqual([
      { value: 2, message: "boom 2" },
      { value: 4, message: "boom 4" },
    ]);
  });

  it("handles an empty list without invoking the task", async () => {
    let calls = 0;
    const results = await mapWithConcurrency([], 4, async () => {
      calls++;
    });
    expect(results).toEqual([]);
    expect(calls).toBe(0);
  });

  it("tolerates a limit larger than the number of items", async () => {
    const results = await mapWithConcurrency([1, 2], 10, async (n) => n);
    expect(results).toEqual([1, 2]);
  });
});
