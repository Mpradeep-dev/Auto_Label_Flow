import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { ProjectsPage } from "@/pages/ProjectsPage";
import { PipelineHome } from "@/pages/PipelineHome";
import { DatasetsPage } from "@/pages/DatasetsPage";
import { ImagesPage } from "@/pages/ImagesPage";
import { AnnotatePage } from "@/pages/AnnotatePage";
import { ModelsPage } from "@/pages/ModelsPage";
import { VideosPage } from "@/pages/VideosPage";
import { AutoAnnotationPage } from "@/pages/AutoAnnotationPage";
import { ExportPage } from "@/pages/ExportPage";
import { TrainingRunsPage } from "@/pages/TrainingRunsPage";
import { ReviewQueuePage } from "@/pages/ReviewQueuePage";
import { DatasetStatisticsPage } from "@/pages/DatasetStatisticsPage";
import { SettingsPage } from "@/pages/SettingsPage";
import { IntegrationsPage } from "@/pages/IntegrationsPage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<Navigate to="/projects" replace />} />
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/settings" element={<IntegrationsPage />} />
        <Route path="/projects/:projectId" element={<PipelineHome />} />
        <Route path="/projects/:projectId/datasets" element={<DatasetsPage />} />
        <Route
          path="/projects/:projectId/datasets/:datasetId/images"
          element={<ImagesPage />}
        />
        <Route
          path="/projects/:projectId/datasets/:datasetId/images/:imageId/annotate"
          element={<AnnotatePage />}
        />
        <Route
          path="/projects/:projectId/datasets/:datasetId/statistics"
          element={<DatasetStatisticsPage />}
        />
        <Route path="/projects/:projectId/images" element={<ImagesPage />} />
        <Route path="/projects/:projectId/videos" element={<VideosPage />} />
        <Route path="/projects/:projectId/auto-annotation" element={<AutoAnnotationPage />} />
        <Route path="/projects/:projectId/review" element={<ReviewQueuePage />} />
        <Route path="/projects/:projectId/models" element={<ModelsPage />} />
        <Route path="/projects/:projectId/training" element={<TrainingRunsPage />} />
        <Route path="/projects/:projectId/export" element={<ExportPage />} />
        <Route path="/projects/:projectId/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/projects" replace />} />
      </Route>
    </Routes>
  );
}
