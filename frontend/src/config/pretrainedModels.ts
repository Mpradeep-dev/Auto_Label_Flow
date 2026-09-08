/**
 * Curated pretrained weights a user can register in one click instead of
 * hunting down a direct download link themselves. URLs point at Ultralytics'
 * official `ultralytics/assets` GitHub release — the same source the
 * `ultralytics` package itself resolves these filenames to.
 *
 * Pose weights are deliberately absent: this app never fine-tunes a POSE
 * model (see AGENTS.md — it's a fixed auxiliary checkpoint for the
 * anatomical_proximity quality rule), so there's no size/accuracy tradeoff
 * to curate the way there is for a DETECTOR you're about to fine-tune.
 */

const RELEASE_BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0";

export interface PretrainedModel {
  name: string;
  url: string;
  framework: "ultralytics" | "yolo-world";
  hint: string;
}

export interface PretrainedModelFamily {
  family: string;
  models: PretrainedModel[];
}

const SIZE_HINTS = [
  "Nano — fastest, lowest accuracy; quick pipeline tests",
  "Small — good balance for CPU/edge fine-tuning",
  "Medium — solid default for GPU fine-tuning",
  "Large — higher accuracy, slower",
  "Extra-large — best accuracy, slowest",
] as const;

function detectorFamily(family: string, prefix: string): PretrainedModelFamily {
  const sizes = ["n", "s", "m", "l", "x"] as const;
  return {
    family,
    models: sizes.map((size, i) => ({
      name: `${prefix}${size}`,
      url: `${RELEASE_BASE}/${prefix}${size}.pt`,
      framework: "ultralytics",
      hint: SIZE_HINTS[i],
    })),
  };
}

export const PRETRAINED_MODEL_FAMILIES: PretrainedModelFamily[] = [
  detectorFamily("YOLOv8", "yolov8"),
  detectorFamily("YOLO11", "yolo11"),
  detectorFamily("YOLO26", "yolo26"),
  {
    family: "YOLO-World v2 (open-vocab)",
    models: (["s", "m", "l", "x"] as const).map((size, i) => ({
      name: `yolov8${size}-worldv2`,
      url: `${RELEASE_BASE}/yolov8${size}-worldv2.pt`,
      framework: "yolo-world",
      // world models only ship s/m/l/x (no nano) — reuse the s/m/l/x hints
      hint: SIZE_HINTS[i + 1],
    })),
  },
];
