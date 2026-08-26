import { useEffect, useRef } from "react";
import type { AnnotationImage } from "@/types";

interface Props {
  images: AnnotationImage[];
  currentId: string;
  onSelect: (id: string) => void;
}

const STATUS_RING: Record<string, string> = {
  APPROVED: "ring-2 ring-ink",
  REJECTED: "ring-2 ring-accent",
  PENDING: "ring-1 ring-ink/20",
};

/** Bottom filmstrip — shows where the reviewer is in the sequence and lets
 * them jump directly. Matters most for video-frame datasets, where "the
 * next 20 frames" is a meaningful unit of context a flat gallery loses. */
export function Filmstrip({ images, currentId, onSelect }: Props) {
  const currentRef = useRef<HTMLButtonElement>(null);

  // Prev/Next/approve-and-advance move `currentId` without touching this
  // strip's own scroll position — without this, the highlighted thumbnail
  // walks off either edge after a few steps and the reviewer loses track
  // of where they are in the sequence.
  useEffect(() => {
    currentRef.current?.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
  }, [currentId]);

  return (
    <div className="flex h-16 shrink-0 items-center gap-1.5 overflow-x-auto border-t-2 border-ink bg-paper px-2">
      {images.map((img) => (
        <button
          key={img.id}
          ref={img.id === currentId ? currentRef : undefined}
          onClick={() => onSelect(img.id)}
          className={`h-12 w-16 shrink-0 overflow-hidden bg-plate ${
            img.id === currentId ? "ring-2 ring-accent" : STATUS_RING[img.review_status]
          }`}
          title={img.original_filename}
        >
          {/* This strip renders every image in the dataset as a real <button>
              (3500+ for a video-frame dataset) — without lazy loading, all
              of them fire their thumbnail request the instant this mounts,
              flooding the browser's ~6-per-host connection limit so
              whichever thumbnails are actually near the current position
              sit queued behind hundreds of others and only resolve once
              you scroll near them. `loading="lazy"` defers the fetch until
              a thumbnail is actually near the (horizontally scrolling)
              viewport, so scrolling here loads a handful of images at a
              time instead of the whole dataset at once. */}
          <img
            src={img.url}
            alt=""
            loading="lazy"
            decoding="async"
            className="h-full w-full object-cover"
          />
        </button>
      ))}
    </div>
  );
}
