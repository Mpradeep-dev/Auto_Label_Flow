import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { useKeyboardShortcuts, type ShortcutHandlers } from "./useKeyboardShortcuts";

function TestHarness({ handlers, enabled = true }: { handlers: ShortcutHandlers; enabled?: boolean }) {
  useKeyboardShortcuts(handlers, enabled);
  return <input aria-label="text-field" />;
}

function makeHandlers(overrides: Partial<ShortcutHandlers> = {}): ShortcutHandlers {
  return {
    add: vi.fn(),
    delete: vi.fn(),
    edit: vi.fn(),
    prev: vi.fn(),
    next: vi.fn(),
    approve: vi.fn(),
    save: vi.fn(),
    zoom: vi.fn(),
    fit: vi.fn(),
    setClassByIndex: vi.fn(),
    ...overrides,
  };
}

describe("useKeyboardShortcuts", () => {
  it("dispatches A to add", async () => {
    const handlers = makeHandlers();
    render(<TestHarness handlers={handlers} />);
    await userEvent.keyboard("a");
    expect(handlers.add).toHaveBeenCalledTimes(1);
  });

  it("dispatches D to delete and arrow keys to prev/next", async () => {
    const handlers = makeHandlers();
    render(<TestHarness handlers={handlers} />);
    await userEvent.keyboard("d");
    await userEvent.keyboard("{ArrowLeft}");
    await userEvent.keyboard("{ArrowRight}");
    expect(handlers.delete).toHaveBeenCalledTimes(1);
    expect(handlers.prev).toHaveBeenCalledTimes(1);
    expect(handlers.next).toHaveBeenCalledTimes(1);
  });

  it("dispatches Space to approve", async () => {
    const handlers = makeHandlers();
    render(<TestHarness handlers={handlers} />);
    await userEvent.keyboard(" ");
    expect(handlers.approve).toHaveBeenCalledTimes(1);
  });

  it("maps number keys 1-9 to setClassByIndex(0-8)", async () => {
    const handlers = makeHandlers();
    render(<TestHarness handlers={handlers} />);
    await userEvent.keyboard("1");
    await userEvent.keyboard("9");
    expect(handlers.setClassByIndex).toHaveBeenNthCalledWith(1, 0);
    expect(handlers.setClassByIndex).toHaveBeenNthCalledWith(2, 8);
  });

  it("does not dispatch shortcuts while a text input is focused", async () => {
    const handlers = makeHandlers();
    render(<TestHarness handlers={handlers} />);
    const input = screen.getByLabelText("text-field");
    input.focus();
    await userEvent.keyboard("a");
    expect(handlers.add).not.toHaveBeenCalled();
  });

  it("does nothing when disabled", async () => {
    const handlers = makeHandlers();
    render(<TestHarness handlers={handlers} enabled={false} />);
    await userEvent.keyboard("a");
    expect(handlers.add).not.toHaveBeenCalled();
  });
});
