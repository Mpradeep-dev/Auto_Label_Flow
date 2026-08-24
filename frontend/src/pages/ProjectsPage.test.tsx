import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ProjectsPage } from "./ProjectsPage";

function renderWithProviders(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("ProjectsPage", () => {
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

  it("renders the empty state once projects load", async () => {
    renderWithProviders(<ProjectsPage />);
    expect(screen.getByRole("heading", { name: "Projects", level: 1 })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText(/no projects yet/i)).toBeInTheDocument());
  });

  it("disables create until a name is entered", () => {
    renderWithProviders(<ProjectsPage />);
    const button = screen.getByRole("button", { name: /create/i });
    expect(button).toBeDisabled();
  });
});
