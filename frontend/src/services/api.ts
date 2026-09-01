import type {
  Annotation,
  AnnotationFlag,
  AnnotationImage,
  Dataset,
  DatasetStatistics,
  DatasetStats,
  ErrorAnalysis,
  ImageListPage,
  ImageReviewStatus,
  DatasetVersion,
  InferenceJob,
  IntegrationStatus,
  MLModel,
  ModelKind,
  Project,
  ReviewQueuePage,
  RoboflowJob,
  RoboflowProjectSummary,
  RoboflowVersionSummary,
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
  health: () => request<{ status: string; version?: string; task_queue?: string }>("/health"),

  // --- Desktop app: system info + optional add-on packs ---
  systemInfo: () =>
    request<{
      app_version: string;
      schema_version: number | null;
      task_queue: string;
      storage_backend: string;
      data_dir: string | null;
      python_version: string;
      platform: string;
      frozen: boolean;
      gpu_pack_installed: boolean;
      integrations_pack_installed: boolean;
    }>("/system/info"),
  listPacks: () =>
    request<{
      packs: { name: "gpu" | "integrations"; installed: boolean; version: string | null; size_bytes: number | null }[];
    }>("/system/packs"),
  installPack: (name: "gpu" | "integrations") =>
    request<{ pack: string; task_id: string }>(`/system/packs/${name}/install`, { method: "POST" }),
  removePack: (name: "gpu" | "integrations") =>
    request<void>(`/system/packs/${name}`, { method: "DELETE" }),

  // --- Projects ---
  listProjects: () => request<Project[]>("/projects"),
  getProject: (id: string) => request<Project>(`/projects/${id}`),
  createProject: (data: { name: string; description?: string }) =>
    request<Project>("/projects", { method: "POST", body: JSON.stringify(data) }),
  updateProject: (
    id: string,
    data: Partial<Pick<Project, "name" | "description" | "class_config" | "quality_rule_config">>,
  ) => request<Project>(`/projects/${id}`, { method: "PATCH", body: JSON.stringify(data) }),
  deleteProject: (id: string) => request<void>(`/projects/${id}`, { method: "DELETE" }),

  // --- Datasets ---
  listDatasets: (projectId: string) => request<Dataset[]>(`/projects/${projectId}/datasets`),
  createDataset: (projectId: string, data: { name: string; description?: string }) =>
    request<Dataset>(`/projects/${projectId}/datasets`, { method: "POST", body: JSON.stringify(data) }),
  getDataset: (id: string) => request<Dataset>(`/datasets/${id}`),
  getDatasetStats: (id: string) => request<DatasetStats>(`/datasets/${id}/stats`),
  deleteDataset: (id: string) => request<void>(`/datasets/${id}`, { method: "DELETE" }),

  // --- Images ---
  listImages: (datasetId: string, limit = 50, offset = 0, reviewStatus?: ImageReviewStatus) =>
    request<ImageListPage>(
      `/datasets/${datasetId}/images?limit=${limit}&offset=${offset}${reviewStatus ? `&review_status=${reviewStatus}` : ""}`,
    ),
  // The backend caps a single page at 200 regardless of the limit
  // requested — a dataset with more images than that (video frame
  // extraction routinely produces this) needs every page merged for
  // callers that require the complete ordered list, not one page of it
  // (AnnotatePage's prev/next navigation and filmstrip; ImagesPage's own
  // paginated gallery uses listImages directly instead — it only ever
  // needs one page at a time). Optional reviewStatus scopes it to one
  // status-filtered tab, e.g. AnnotatePage browsing a Pending/Approved
  // filter carried over from ImagesPage's own tabs.
  listAllImages: async (datasetId: string, reviewStatus?: ImageReviewStatus): Promise<AnnotationImage[]> => {
    const pageSize = 200;
    const statusQs = reviewStatus ? `&review_status=${reviewStatus}` : "";
    const first = await request<ImageListPage>(`/datasets/${datasetId}/images?limit=${pageSize}&offset=0${statusQs}`);
    const items = [...first.items];
    // Remaining pages don't depend on each other — fetching them in
    // parallel instead of one-at-a-time cuts wall-clock time roughly by the
    // page count on a large dataset (was the dominant cost on a
    // multi-thousand-image dataset: ~18 sequential round trips before this
    // ever resolved).
    const offsets: number[] = [];
    for (let offset = pageSize; offset < first.total; offset += pageSize) offsets.push(offset);
    const rest = await Promise.all(
      offsets.map((offset) => request<ImageListPage>(`/datasets/${datasetId}/images?limit=${pageSize}&offset=${offset}${statusQs}`)),
    );
    for (const page of rest) items.push(...page.items);
    return items;
  },
  getImage: (id: string) => request<AnnotationImage>(`/images/${id}`),
  uploadImage: (datasetId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<AnnotationImage>(`/datasets/${datasetId}/images`, { method: "POST", body: form });
  },
  deleteImage: (id: string) => request<void>(`/images/${id}`, { method: "DELETE" }),

  // --- Annotations ---
  listAnnotations: (imageId: string) => request<Annotation[]>(`/images/${imageId}/annotations`),
  createAnnotation: (
    data:
      | {
          image_id: string;
          class_id: number;
          class_name: string;
          shape_type?: "BBOX";
          x1: number;
          y1: number;
          x2: number;
          y2: number;
          confidence?: number | null;
        }
      | {
          image_id: string;
          class_id: number;
          class_name: string;
          shape_type: "POLYGON";
          points: [number, number][];
          confidence?: number | null;
        },
  ) => request<Annotation>("/annotations", { method: "POST", body: JSON.stringify(data) }),
  updateAnnotation: (
    id: string,
    data: Partial<Pick<Annotation, "class_id" | "class_name" | "x1" | "y1" | "x2" | "y2" | "points" | "confidence">>,
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
  registerModel: (data: {
    name: string;
    weights_path: string;
    kind: ModelKind;
    version?: string;
    framework?: string;
  }) => request<MLModel>("/models", { method: "POST", body: JSON.stringify(data) }),
  downloadModel: (data: {
    name: string;
    url: string;
    kind: ModelKind;
    version?: string;
    framework?: string;
  }) => request<MLModel>("/models/download", { method: "POST", body: JSON.stringify(data) }),
  uploadModel: (
    file: File,
    data: { name: string; kind: ModelKind; version?: string; framework?: string },
  ) => {
    const form = new FormData();
    form.append("file", file);
    form.append("name", data.name);
    form.append("kind", data.kind);
    if (data.version) form.append("version", data.version);
    if (data.framework) form.append("framework", data.framework);
    return request<MLModel>("/models/upload", { method: "POST", body: form });
  },
  renameModel: (id: string, name: string) =>
    request<MLModel>(`/models/${id}`, { method: "PATCH", body: JSON.stringify({ name }) }),
  deleteModel: (id: string) => request<void>(`/models/${id}`, { method: "DELETE" }),

  // --- Videos ---
  listVideos: (datasetId: string) => request<VideoRecord[]>(`/datasets/${datasetId}/videos`),
  uploadVideo: (datasetId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<VideoRecord>(`/datasets/${datasetId}/videos`, { method: "POST", body: form });
  },
  extractFrames: (videoId: string, config: { interval?: number; fps?: number }) =>
    request<VideoRecord>(`/videos/${videoId}/extract-frames`, { method: "POST", body: JSON.stringify(config) }),
  deleteVideo: (videoId: string) => request<void>(`/videos/${videoId}`, { method: "DELETE" }),

  // --- Inference jobs ---
  createInferenceJob: (data: { dataset_id: string; model_id: string; conf?: number; iou?: number }) =>
    request<InferenceJob>("/inference/jobs", { method: "POST", body: JSON.stringify(data) }),
  getInferenceJob: (id: string) => request<InferenceJob>(`/inference/jobs/${id}`),
  // Full run history for a project — backs the Auto Annotation page's
  // history list, so a job that finished while the user was elsewhere
  // doesn't just disappear (unlike the single-dataset /latest reattach).
  listInferenceJobs: (projectId: string) => request<InferenceJob[]>(`/inference/jobs?project_id=${projectId}`),
  cancelInferenceJob: (id: string) => request<InferenceJob>(`/inference/jobs/${id}/cancel`, { method: "POST" }),
  // Lets the Auto Annotation page reattach to a job it kicked off before a
  // navigation away or a reload dropped its local state.
  getLatestInferenceJob: (datasetId: string) =>
    request<InferenceJob | null>(`/inference/jobs/latest?dataset_id=${datasetId}`),

  // --- Dataset versions / export ---
  listDatasetVersions: (datasetId: string) => request<DatasetVersion[]>(`/datasets/${datasetId}/versions`),
  createDatasetVersion: (
    datasetId: string,
    data: { train_ratio?: number; val_ratio?: number; test_ratio?: number; seed?: number },
  ) => request<DatasetVersion>(`/datasets/${datasetId}/versions`, { method: "POST", body: JSON.stringify(data) }),
  exportDatasetVersion: (versionId: string) =>
    request<DatasetVersion>(`/versions/${versionId}/export`, { method: "POST" }),
  exportVersionCoco: (versionId: string) =>
    request<DatasetVersion>(`/versions/${versionId}/export/coco`, { method: "POST" }),
  exportVersionCvat: (versionId: string) =>
    request<DatasetVersion>(`/versions/${versionId}/export/cvat`, { method: "POST" }),

  // --- COCO / CVAT-XML import (the CVAT round trip — see export_coco.py /
  // export_cvat.py's docstrings for why these two formats specifically) ---
  importCocoDataset: (projectId: string, file: File, datasetName?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (datasetName) form.append("dataset_name", datasetName);
    return request<Dataset>(`/projects/${projectId}/import/coco`, { method: "POST", body: form });
  },
  importCvatDataset: (projectId: string, file: File, datasetName?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (datasetName) form.append("dataset_name", datasetName);
    return request<Dataset>(`/projects/${projectId}/import/cvat`, { method: "POST", body: form });
  },

  // --- Training ---
  getTrainingProviders: () => request<TrainingProviders>("/training/providers"),
  listTrainingJobs: (projectId: string) => request<TrainingJob[]>(`/training/jobs?project_id=${projectId}`),
  getTrainingJob: (id: string) => request<TrainingJob>(`/training/jobs/${id}`),
  getTrainingJobEpochs: (id: string) => request<TrainingJobEpochRow[]>(`/training/jobs/${id}/epochs`),
  createTrainingJob: (data: {
    dataset_version_id: string;
    base_model_id: string;
    result_model_name?: string;
    provider?: string;
    epochs?: number;
    batch_size?: number;
    image_size?: number;
    learning_rate?: number;
    device?: string;
    enable_gpu?: boolean;
    extra_args?: Record<string, unknown>;
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
  getReviewQueue: (params: {
    project_id: string;
    dataset_id?: string;
    flag_type?: string;
    review_status?: "PENDING" | "APPROVED" | "REJECTED";
    limit?: number;
    offset?: number;
  }) => {
    const query = new URLSearchParams();
    query.set("project_id", params.project_id);
    if (params.dataset_id) query.set("dataset_id", params.dataset_id);
    if (params.flag_type) query.set("flag_type", params.flag_type);
    if (params.review_status) query.set("review_status", params.review_status);
    if (params.limit) query.set("limit", String(params.limit));
    if (params.offset) query.set("offset", String(params.offset));
    return request<ReviewQueuePage>(`/review/queue?${query.toString()}`);
  },
  // Mirrors listAllImages above: when AnnotatePage is entered from the
  // Review Queue, its prev/next/filmstrip need the *complete*
  // difficulty-ordered id list for that queue, not one 60-item page of it —
  // otherwise stepping past the page boundary would silently fall back to
  // plain dataset upload order, which is the "review order changes once you
  // open an image" bug this exists to avoid.
  listAllReviewQueueImageIds: async (params: {
    project_id: string;
    dataset_id?: string;
    flag_type?: string;
    review_status?: "PENDING" | "APPROVED" | "REJECTED";
  }): Promise<string[]> => {
    const pageSize = 200;
    const first = await api.getReviewQueue({ ...params, limit: pageSize, offset: 0 });
    const ids = first.items.map((item) => item.image_id);
    // See listAllImages above — independent pages fetched in parallel
    // instead of sequentially.
    const offsets: number[] = [];
    for (let offset = pageSize; offset < first.total; offset += pageSize) offsets.push(offset);
    const rest = await Promise.all(offsets.map((offset) => api.getReviewQueue({ ...params, limit: pageSize, offset })));
    for (const page of rest) ids.push(...page.items.map((item) => item.image_id));
    return ids;
  },

  // --- Dashboards ---
  getDatasetStatistics: (datasetId: string) => request<DatasetStatistics>(`/datasets/${datasetId}/statistics`),
  getErrorAnalysis: (datasetId: string) => request<ErrorAnalysis>(`/datasets/${datasetId}/error-analysis`),
  updateModelMetrics: (modelId: string, metrics: Record<string, number>) =>
    request<MLModel>(`/models/${modelId}/metrics`, { method: "PUT", body: JSON.stringify({ metrics }) }),

  // --- Integrations (Settings page: Kaggle + Modal + Roboflow connect) ---
  listIntegrations: () => request<IntegrationStatus[]>("/integrations"),
  connectKaggle: (data: { username: string; key: string }) =>
    request<IntegrationStatus>("/integrations/kaggle", { method: "POST", body: JSON.stringify(data) }),
  disconnectKaggle: () => request<void>("/integrations/kaggle", { method: "DELETE" }),
  connectModal: (data: { token_id: string; token_secret: string }) =>
    request<IntegrationStatus>("/integrations/modal", { method: "POST", body: JSON.stringify(data) }),
  disconnectModal: () => request<void>("/integrations/modal", { method: "DELETE" }),
  connectRoboflow: (data: { api_key: string; default_workspace?: string }) =>
    request<IntegrationStatus>("/integrations/roboflow", { method: "POST", body: JSON.stringify(data) }),
  disconnectRoboflow: () => request<void>("/integrations/roboflow", { method: "DELETE" }),
  exportVersionToRoboflow: (versionId: string, data: { workspace: string; project: string }) =>
    request<RoboflowJob>(`/versions/${versionId}/export/roboflow`, {
      method: "POST",
      body: JSON.stringify(data),
    }),
  importRoboflowDataset: (
    projectId: string,
    // `version: undefined` pulls the project's raw uploaded images instead
    // of a generated Version — see RoboflowImportSection in DatasetsPage.
    // `unannotated_only` only matters for that raw path.
    data: {
      workspace: string;
      project: string;
      version?: number;
      dataset_name?: string;
      unannotated_only?: boolean;
    },
  ) => request<RoboflowJob>(`/projects/${projectId}/import/roboflow`, { method: "POST", body: JSON.stringify(data) }),
  listRoboflowProjects: (workspace?: string) =>
    request<RoboflowProjectSummary[]>(
      `/integrations/roboflow/projects${workspace ? `?workspace=${encodeURIComponent(workspace)}` : ""}`,
    ),
  listRoboflowVersions: (workspace: string, project: string) =>
    request<RoboflowVersionSummary[]>(
      `/integrations/roboflow/projects/${encodeURIComponent(workspace)}/${encodeURIComponent(project)}/versions`,
    ),
  getRoboflowJob: (id: string) => request<RoboflowJob>(`/integrations/roboflow/jobs/${id}`),
  // Lets a page reattach to a job it kicked off before a navigation away
  // or a reload dropped its local state — see RoboflowImportSection /
  // RoboflowExportControls, which poll this on mount.
  getLatestRoboflowJob: (params: { kind: "IMPORT"; project_id: string } | { kind: "EXPORT"; dataset_version_id: string }) => {
    const query = new URLSearchParams({ kind: params.kind });
    if ("project_id" in params) query.set("project_id", params.project_id);
    if ("dataset_version_id" in params) query.set("dataset_version_id", params.dataset_version_id);
    return request<RoboflowJob | null>(`/integrations/roboflow/jobs/latest?${query.toString()}`);
  },
  cancelRoboflowJob: (id: string) =>
    request<RoboflowJob>(`/integrations/roboflow/jobs/${id}/cancel`, { method: "POST" }),
};

export { ApiError };
