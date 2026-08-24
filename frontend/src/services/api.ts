import type {
  Annotation,
  AnnotationFlag,
  AnnotationImage,
  Dataset,
  DatasetStatistics,
  DatasetStats,
  ErrorAnalysis,
  ImageListPage,
  DatasetVersion,
  InferenceJob,
  MLModel,
  Project,
  ReviewQueuePage,
  TrainingJob,
  TrainingJobEpochRow,
  TrainingProviders,
  VideoRecord,
} from "@/types";

const BASE = "/api/v1";

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* body wasn't JSON — keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  // --- Projects ---
  listProjects: () => request<Project[]>("/projects"),
  getProject: (id: string) => request<Project>(`/projects/${id}`),
  createProject: (data: { name: string; description?: string }) =>
    request<Project>("/projects", { method: "POST", body: JSON.stringify(data) }),
  updateProject: (id: string, data: Partial<Pick<Project, "name" | "description" | "class_config">>) =>
    request<Project>(`/projects/${id}`, { method: "PATCH", body: JSON.stringify(data) }),
  deleteProject: (id: string) => request<void>(`/projects/${id}`, { method: "DELETE" }),

  // --- Datasets ---
  listDatasets: (projectId: string) => request<Dataset[]>(`/projects/${projectId}/datasets`),
  createDataset: (projectId: string, data: { name: string; description?: string }) =>
    request<Dataset>(`/projects/${projectId}/datasets`, { method: "POST", body: JSON.stringify(data) }),
  getDataset: (id: string) => request<Dataset>(`/datasets/${id}`),
  getDatasetStats: (id: string) => request<DatasetStats>(`/datasets/${id}/stats`),
  deleteDataset: (id: string) => request<void>(`/datasets/${id}`, { method: "DELETE" }),

  // --- Images ---
  listImages: (datasetId: string, limit = 50, offset = 0) =>
    request<ImageListPage>(`/datasets/${datasetId}/images?limit=${limit}&offset=${offset}`),
  getImage: (id: string) => request<AnnotationImage>(`/images/${id}`),
  uploadImage: (datasetId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<AnnotationImage>(`/datasets/${datasetId}/images`, { method: "POST", body: form });
  },
  deleteImage: (id: string) => request<void>(`/images/${id}`, { method: "DELETE" }),

  // --- Annotations ---
  listAnnotations: (imageId: string) => request<Annotation[]>(`/images/${imageId}/annotations`),
  createAnnotation: (data: {
    image_id: string;
    class_id: number;
    class_name: string;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence?: number | null;
  }) => request<Annotation>("/annotations", { method: "POST", body: JSON.stringify(data) }),
  updateAnnotation: (
    id: string,
    data: Partial<Pick<Annotation, "class_id" | "class_name" | "x1" | "y1" | "x2" | "y2" | "confidence">>,
  ) => request<Annotation>(`/annotations/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteAnnotation: (id: string, data?: { error_category?: string; error_reason?: string }) =>
    request<void>(`/annotations/${id}`, {
      method: "DELETE",
      body: data ? JSON.stringify(data) : undefined,
    }),
  duplicateAnnotation: (id: string) =>
    request<Annotation>(`/annotations/${id}/duplicate`, { method: "POST" }),
  approveImage: (imageId: string) =>
    request<AnnotationImage>(`/images/${imageId}/approve`, { method: "POST" }),
  rejectImage: (imageId: string) =>
    request<AnnotationImage>(`/images/${imageId}/reject`, { method: "POST" }),
  autoAnnotate: (imageId: string, data: { model_id: string; conf?: number; iou?: number; replace_existing?: boolean }) =>
    request<Annotation[]>(`/images/${imageId}/auto-annotate`, { method: "POST", body: JSON.stringify(data) }),

  // --- Models ---
  listModels: () => request<MLModel[]>("/models"),
  registerModel: (data: { name: string; weights_path: string; kind: "DETECTOR" | "POSE"; version?: string }) =>
    request<MLModel>("/models", { method: "POST", body: JSON.stringify(data) }),

  // --- Videos ---
  listVideos: (datasetId: string) => request<VideoRecord[]>(`/datasets/${datasetId}/videos`),
  uploadVideo: (datasetId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<VideoRecord>(`/datasets/${datasetId}/videos`, { method: "POST", body: form });
  },
  extractFrames: (videoId: string, config: { interval?: number; fps?: number }) =>
    request<VideoRecord>(`/videos/${videoId}/extract-frames`, { method: "POST", body: JSON.stringify(config) }),

  // --- Inference jobs ---
  createInferenceJob: (data: { dataset_id: string; model_id: string; conf?: number; iou?: number }) =>
    request<InferenceJob>("/inference/jobs", { method: "POST", body: JSON.stringify(data) }),
  getInferenceJob: (id: string) => request<InferenceJob>(`/inference/jobs/${id}`),

  // --- Dataset versions / export ---
  listDatasetVersions: (datasetId: string) => request<DatasetVersion[]>(`/datasets/${datasetId}/versions`),
  createDatasetVersion: (
    datasetId: string,
    data: { train_ratio?: number; val_ratio?: number; test_ratio?: number; seed?: number },
  ) => request<DatasetVersion>(`/datasets/${datasetId}/versions`, { method: "POST", body: JSON.stringify(data) }),
  exportDatasetVersion: (versionId: string) =>
    request<DatasetVersion>(`/versions/${versionId}/export`, { method: "POST" }),

  // --- Training ---
  getTrainingProviders: () => request<TrainingProviders>("/training/providers"),
  listTrainingJobs: (projectId: string) => request<TrainingJob[]>(`/training/jobs?project_id=${projectId}`),
  getTrainingJob: (id: string) => request<TrainingJob>(`/training/jobs/${id}`),
  getTrainingJobEpochs: (id: string) => request<TrainingJobEpochRow[]>(`/training/jobs/${id}/epochs`),
  createTrainingJob: (data: {
    dataset_version_id: string;
    base_model_id: string;
    provider?: string;
    epochs?: number;
    batch_size?: number;
    image_size?: number;
    learning_rate?: number;
    device?: string;
  }) => request<TrainingJob>("/training/jobs", { method: "POST", body: JSON.stringify(data) }),
  cancelTrainingJob: (id: string) => request<TrainingJob>(`/training/jobs/${id}/cancel`, { method: "POST" }),

  // --- Quality / review ---
  analyzeImageQuality: (imageId: string) =>
    request<AnnotationFlag[]>(`/images/${imageId}/analyze-quality`, { method: "POST" }),
  listImageFlags: (imageId: string) => request<AnnotationFlag[]>(`/images/${imageId}/flags`),
  resolveFlag: (flagId: string, resolution: string) =>
    request<AnnotationFlag>(`/annotation-flags/${flagId}/resolve`, {
      method: "POST",
      body: JSON.stringify({ resolution }),
    }),
  analyzeDatasetQuality: (datasetId: string) =>
    request<{ task_id: string | null }>(`/datasets/${datasetId}/analyze-quality`, { method: "POST" }),
  getReviewQueue: (params: { project_id: string; dataset_id?: string; flag_type?: string; limit?: number; offset?: number }) => {
    const query = new URLSearchParams();
    query.set("project_id", params.project_id);
    if (params.dataset_id) query.set("dataset_id", params.dataset_id);
    if (params.flag_type) query.set("flag_type", params.flag_type);
    if (params.limit) query.set("limit", String(params.limit));
    if (params.offset) query.set("offset", String(params.offset));
    return request<ReviewQueuePage>(`/review/queue?${query.toString()}`);
  },

  // --- Dashboards ---
  getDatasetStatistics: (datasetId: string) => request<DatasetStatistics>(`/datasets/${datasetId}/statistics`),
  getErrorAnalysis: (datasetId: string) => request<ErrorAnalysis>(`/datasets/${datasetId}/error-analysis`),
  updateModelMetrics: (modelId: string, metrics: Record<string, number>) =>
    request<MLModel>(`/models/${modelId}/metrics`, { method: "PUT", body: JSON.stringify({ metrics }) }),
};

export { ApiError };
