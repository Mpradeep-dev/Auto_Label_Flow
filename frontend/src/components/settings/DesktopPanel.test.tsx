import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { DesktopPanel } from "./DesktopPanel";
import { api } from "@/services/api";

vi.mock("@/services/api", () => ({
  api: {
    systemInfo: vi.fn().mockResolvedValue({ app_version: "1.0.0" }),
    listPacks: vi.fn().mockResolvedValue({ packs: [] }),
    listSamModels: vi.fn().mockResolvedValue([
      { name: "sam-lite", label: "SAM Lite (MobileSAM)", blurb: "~40 MB, usable on CPU.", installed: false, size_bytes: null },
      { name: "sam-full", label: "SAM Full (SAM2)", blurb: "~150 MB+, best mask quality.", installed: true, size_bytes: 150_000_000 },
    ]),
    installSamModel: vi.fn().mockResolvedValue({ variant: "sam-lite", task_id: "t1" }),
    removeSamModel: vi.fn().mockResolvedValue(undefined),
  },
  // FieldError (rendered by both PackRow and SamModelRow) checks
  // `error instanceof ApiError` — needs a real export, not just a mocked
  // `api` object, or importing it throws.
  ApiError: class ApiError extends Error {
    status: number;
    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
}));

vi.mock("@/services/desktop", () => ({
  desktop: {
    onUpdateAvailable: () => () => {},
    onDownloadProgress: () => () => {},
    onUpdateDownloaded: () => () => {},
    onUpdateError: () => () => {},
  },
}));

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <DesktopPanel />
    </QueryClientProvider>,
  );
}

describe("DesktopPanel — SAM model rows", () => {
  beforeEach(() => vi.clearAllMocks());

  function cardFor(label: string): HTMLElement {
    return screen.getByText(label).closest("div")!.parentElement!;
  }

  it("renders both SAM variants with their installed state", async () => {
    renderPanel();
    await waitFor(() => expect(screen.getByText("SAM Lite (MobileSAM)")).toBeInTheDocument());
    expect(screen.getByText("SAM Full (SAM2)")).toBeInTheDocument();
    expect(within(cardFor("SAM Lite (MobileSAM)")).getByText("Not installed")).toBeInTheDocument();
    expect(within(cardFor("SAM Full (SAM2)")).getByText("Installed")).toBeInTheDocument();
  });

  it("downloading an uninstalled variant calls installSamModel", async () => {
    const user = userEvent.setup();
    renderPanel();
    await waitFor(() => expect(screen.getByText("SAM Lite (MobileSAM)")).toBeInTheDocument());

    const downloadButton = within(cardFor("SAM Lite (MobileSAM)")).getByRole("button", { name: /download/i });
    await user.click(downloadButton);

    await waitFor(() => expect(api.installSamModel).toHaveBeenCalledWith("sam-lite"));
  });

  it("an installed variant shows Remove instead of Download", async () => {
    renderPanel();
    await waitFor(() => expect(screen.getByText("SAM Full (SAM2)")).toBeInTheDocument());
    expect(within(cardFor("SAM Full (SAM2)")).getByRole("button", { name: /remove/i })).toBeInTheDocument();
  });
});
