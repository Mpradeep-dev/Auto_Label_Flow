import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { ProjectGuard } from "@/components/layout/ProjectGuard";
import { LandingPage } from "@/pages/LandingPage";
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
import { HelpPage } from "@/pages/HelpPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route element={<AppShell />}>
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/settings" element={<IntegrationsPage />} />
        <Route path="/help" element={<HelpPage />} />
        {/* Every child here can assume the project exists — ProjectGuard
            fetches it once at this boundary and shows a real not-found/error
            state instead of each page independently ignoring a 404
            (audit finding FE-03). */}
        <Route path="/projects/:projectId" element={<ProjectGuard />}>
          <Route index element={<PipelineHome />} />
          <Route path="datasets" element={<DatasetsPage />} />
          <Route path="datasets/:datasetId/images" element={<ImagesPage />} />
          <Route path="datasets/:datasetId/images/:imageId/annotate" element={<AnnotatePage />} />
          <Route path="datasets/:datasetId/statistics" element={<DatasetStatisticsPage />} />
          <Route path="images" element={<ImagesPage />} />
          <Route path="videos" element={<VideosPage />} />
          <Route path="auto-annotation" element={<AutoAnnotationPage />} />
          <Route path="review" element={<ReviewQueuePage />} />
          <Route path="models" element={<ModelsPage />} />
          <Route path="training" element={<TrainingRunsPage />} />
          <Route path="export" element={<ExportPage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/projects" replace />} />
      </Route>
    </Routes>
  );
}
