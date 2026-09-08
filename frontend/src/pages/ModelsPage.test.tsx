import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ModelsPage } from "./ModelsPage";

function renderWithProviders(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("ModelsPage — pretrained model picker", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => [],
      }),
    );
  });

  it("registers a pretrained model in one click via the download-by-URL endpoint", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ModelsPage />);

    await user.click(await screen.findByRole("button", { name: /pretrained/i }));
    await user.click(await screen.findByRole("button", { name: /register yolov8n/i }));

    await waitFor(() => {
      const downloadCall = (fetch as ReturnType<typeof vi.fn>).mock.calls.find(
        ([url]) => url === "/api/v1/models/download",
      );
      expect(downloadCall).toBeDefined();
      const body = JSON.parse(downloadCall![1].body as string);
      expect(body).toMatchObject({
        name: "yolov8n",
        url: "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
        kind: "DETECTOR",
        framework: "ultralytics",
      });
    });
  });

  it("registers a YOLO-World pretrained model with the yolo-world framework", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ModelsPage />);

    await user.click(await screen.findByRole("button", { name: /pretrained/i }));
    await user.click(await screen.findByRole("button", { name: /register yolov8s-worldv2/i }));

    await waitFor(() => {
      const downloadCall = (fetch as ReturnType<typeof vi.fn>).mock.calls.find(
        ([url]) => url === "/api/v1/models/download",
      );
      expect(downloadCall).toBeDefined();
      const body = JSON.parse(downloadCall![1].body as string);
      expect(body).toMatchObject({ name: "yolov8s-worldv2", framework: "yolo-world" });
    });
  });
});
