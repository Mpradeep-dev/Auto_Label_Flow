// Mirrors backend/app/schemas/*.py. Kept hand-written rather than
// codegen'd for now — small enough surface that a generator adds more
// ceremony than it saves at this phase.

export interface ClassEntry {
  id: number;
  name: string;
}

export interface Project {
  id: string;
  name: string;
  slug: string;
  description: string | null;
  class_config: ClassEntry[];
  quality_rule_config: Record<string, unknown>;
  primary_model_id: string | null;
  pose_model_id: string | null;
  created_at: string;
  updated_at: string;
}

export type DatasetStatus = "ACTIVE" | "ARCHIVED";

export interface Dataset {
  id: string;
  project_id: string;
  name: string;
  description: string | null;
  status: DatasetStatus;
  created_at: string;
  updated_at: string;
}

export interface DatasetStats {
  total_images: number;
  pending_images: number;
  approved_images: number;
  total_videos: number;
}

export type ImageSourceType = "UPLOAD" | "VIDEO_FRAME";
export type ImageReviewStatus = "PENDING" | "APPROVED" | "REJECTED";

export interface AnnotationImage {
  id: string;
  project_id: string;
  dataset_id: string;
  original_filename: string;
  width: number;
  height: number;
  source_type: ImageSourceType;
  video_id: string | null;
  frame_index: number | null;
  frame_timestamp_s: number | null;
  review_status: ImageReviewStatus;
  difficulty_score: number | null;
  created_at: string;
  url: string;
}

export interface ImageListPage {
  items: AnnotationImage[];
  total: number;
  limit: number;
  offset: number;
}

export type AnnotationSourceType = "AUTO" | "HUMAN" | "CORRECTED";
export type AnnotationReviewStatus = "PENDING" | "APPROVED" | "REJECTED";

export interface Annotation {
  id: string;
  image_id: string;
  class_id: number;
  class_name: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number | null;
  source: AnnotationSourceType;
  review_status: AnnotationReviewStatus;
  revision_seq: number;
  created_at: string;
  updated_at: string;
}

export type ModelKind = "DETECTOR" | "POSE";

export type VideoStatus = "UPLOADED" | "EXTRACTING" | "EXTRACTED" | "FAILED";

export interface VideoRecord {
  id: string;
  project_id: string;
  dataset_id: string;
  original_filename: string;
  width: number | null;
  height: number | null;
  fps: number | null;
  duration_s: number | null;
  total_frames: number | null;
  status: VideoStatus;
  extracted_frame_count: number;
  error: string | null;
  created_at: string;
  url: string;
}

export type JobStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED";

export interface InferenceJob {
  id: string;
  project_id: string;
  dataset_id: string;
  model_id: string;
  status: JobStatus;
  conf: number;
  iou: number;
  total_images: number;
  processed_images: number;
  failed_images: number;
  total_predictions: number;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface JobProgressEvent {
  current: number;
  total: number;
  predictions: number;
  fps: number;
  eta_s: number | null;
  status: string;
  error: string | null;
}

export type DatasetVersionStatus = "DRAFT" | "EXPORTING" | "EXPORTED" | "FAILED";

export interface DatasetVersion {
  id: string;
  dataset_id: string;
  version_number: number;
  status: DatasetVersionStatus;
  train_ratio: number;
  val_ratio: number;
  test_ratio: number;
  split_seed: number;
  used_frame_level_fallback: boolean;
  total_images: number;
  total_annotations: number;
  error: string | null;
  download_url: string | null;
  created_at: string;
}

export type TrainingProviderName = "LOCAL" | "KAGGLE";
export type TrainingJobStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED";

export interface TrainingJob {
  id: string;
  project_id: string;
  dataset_version_id: string;
  base_model_id: string | null;
  result_model_id: string | null;
  provider: TrainingProviderName;
  status: TrainingJobStatus;
  epochs: number;
  batch_size: number;
  image_size: number;
  learning_rate: number | null;
  device: string;
  current_epoch: number;
  metrics: Record<string, number>;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  failed_at: string | null;
}

export interface TrainingJobEpochRow {
  epoch: number;
  box_loss: number | null;
  cls_loss: number | null;
  dfl_loss: number | null;
  precision: number | null;
  recall: number | null;
  map50: number | null;
  map50_95: number | null;
  recorded_at: string;
}

export interface GPUInfo {
  torch_version: string;
  cuda_available: boolean;
  device_name: string | null;
  vram_total_mb: number | null;
  cuda_version: string | null;
}

export interface TrainingProviders {
  available: TrainingProviderName[];
  gpu: GPUInfo;
}

export type FlagType =
  | "CONE_NEAR_PLAYER"
  | "SUSPICIOUS_CONE"
  | "VERY_SMALL_CONE"
  | "LOW_CONFIDENCE"
  | "POSSIBLE_DUPLICATE"
  | "ISOLATED_DETECTION"
  | "TEMPORAL_ANOMALY";

export type FlagResolution = "CONFIRMED_FP" | "CONFIRMED_OK" | "EDITED";

export interface AnnotationFlag {
  id: string;
  annotation_id: string;
  image_id: string;
  flag_type: FlagType;
  severity: number;
  reason: string;
  details: Record<string, unknown>;
  resolution: FlagResolution | null;
  created_at: string;
}

export interface ReviewQueueItem {
  image_id: string;
  dataset_id: string;
  url: string;
  difficulty_score: number | null;
  review_status: string;
  flags: AnnotationFlag[];
}

export interface ReviewQueuePage {
  items: ReviewQueueItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface AutoLabelAcceptance {
  total_auto_predictions: number;
  accepted: number;
  corrected: number;
  rejected: number;
  acceptance_rate: number | null;
}

export interface DatasetStatistics {
  total_images: number;
  reviewed_images: number;
  pending_images: number;
  completion_pct: number;
  total_annotations: number;
  annotations_by_class: Record<string, number>;
  annotations_by_source: Record<string, number>;
  average_confidence: number | null;
  low_confidence_predictions: number;
  suspicious_cones: number;
  auto_label_acceptance: AutoLabelAcceptance;
}

export interface ErrorAnalysis {
  total_categorized_deletions: number;
  by_category: Record<string, number>;
  by_reason: Record<string, number>;
}

export interface MLModel {
  id: string;
  name: string;
  version: string;
  kind: ModelKind;
  framework: string;
  weights_path: string;
  class_config: ClassEntry[];
  metrics: Record<string, unknown>;
  base_model_id: string | null;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}
