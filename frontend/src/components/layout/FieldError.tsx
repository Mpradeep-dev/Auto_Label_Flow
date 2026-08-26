import { ApiError } from "@/services/api";

/** Shared error-message display for a failed mutation. Several pages used
 * to duplicate this exact component locally (and a couple didn't render
 * mutation errors at all — see audit finding FE-06); one shared version
 * here so every form gets the same treatment. */
export function FieldError({ error }: { error: unknown }) {
  if (!error) return null;
  const message = error instanceof ApiError ? error.message : (error as Error).message;
  return <p className="mt-2 text-xs text-accent">{message}</p>;
}
