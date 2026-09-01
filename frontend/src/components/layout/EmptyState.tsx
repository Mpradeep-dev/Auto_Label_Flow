import type { ReactNode } from "react";

export function EmptyState({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children?: ReactNode;
}) {
  return (
    <div className="col-span-full flex flex-col items-start gap-4 border-2 border-dashed border-ink/20 px-8 py-12">
      <div>
        <p className="text-lg font-bold uppercase tracking-tight">{title}</p>
        <p className="mt-1 max-w-md text-sm text-ink/60">{description}</p>
      </div>
      {children && <div className="flex flex-wrap gap-3">{children}</div>}
    </div>
  );
}
