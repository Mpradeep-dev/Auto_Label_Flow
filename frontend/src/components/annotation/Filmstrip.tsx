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
  return (
    <div className="flex h-16 shrink-0 items-center gap-1.5 overflow-x-auto border-t-2 border-ink bg-paper px-2">
      {images.map((img) => (
        <button
          key={img.id}
          onClick={() => onSelect(img.id)}
          className={`h-12 w-16 shrink-0 overflow-hidden bg-plate ${
            img.id === currentId ? "ring-2 ring-accent" : STATUS_RING[img.review_status]
          }`}
          title={img.original_filename}
        >
          <img src={img.url} alt="" className="h-full w-full object-cover" />
        </button>
      ))}
    </div>
  );
}
