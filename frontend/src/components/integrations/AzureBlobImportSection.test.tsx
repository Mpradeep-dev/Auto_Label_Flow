import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { AzureBlobImportSection } from "./AzureBlobImportSection";
import { api } from "@/services/api";

vi.mock("@/services/api", () => ({
  api: {
    getLatestBlobImportJob: vi.fn().mockResolvedValue(null),
    importAzureBlobDataset: vi.fn().mockResolvedValue({
      id: "job-1",
      project_id: "p1",
      status: "COMPLETED",
      prefix: "prod-batch-1/",
      label_format: "auto",
      dataset_name: null,
      total_items: 0,
      processed_items: 0,
      result_dataset_id: "d1",
      error: null,
      created_at: "2026-09-02T00:00:00Z",
    }),
    cancelBlobImportJob: vi.fn(),
  },
}));

function renderSection() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <AzureBlobImportSection projectId="p1" />
    </QueryClientProvider>,
  );
}

describe("AzureBlobImportSection", () => {
  beforeEach(() => vi.clearAllMocks());

  it("disables Import until a prefix is entered", () => {
    renderSection();
    expect(screen.getByRole("button", { name: /^import$/i })).toBeDisabled();
  });

  it("submits the entered prefix and label format", async () => {
    const user = userEvent.setup();
    renderSection();

    await user.type(screen.getByLabelText(/blob prefix/i), "prod-batch-1/");
    await user.click(screen.getByLabelText(/yolo/i));
    await user.click(screen.getByRole("button", { name: /^import$/i }));

    await waitFor(() =>
      expect(api.importAzureBlobDataset).toHaveBeenCalledWith("p1", {
        prefix: "prod-batch-1/",
        label_format: "yolo",
        dataset_name: undefined,
      }),
    );
  });
});
