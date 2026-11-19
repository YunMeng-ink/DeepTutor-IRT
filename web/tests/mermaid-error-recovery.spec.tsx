import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Mermaid } from "@/components/Mermaid";

const renderer = vi.hoisted(() => ({ render: vi.fn(), initialize: vi.fn() }));
vi.mock("mermaid", () => ({ default: renderer }));

// Keep ``t`` module-scoped: the component's render effect lists it as a
// dependency, so an inline arrow would hand React a new identity on every
// render and re-trigger the effect an unpredictable number of times.
const translate = (value: string) => value;
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: translate }) }));

describe("Mermaid rendering recovery", () => {
  it("renders a corrected chart after a renderer error", async () => {
    // A persistent implementation, not chained ``Once`` values: the component
    // may re-render the corrected chart more than once, and an exhausted mock
    // returns ``undefined``, which the component then destructures and reports
    // as a fresh renderer error.
    renderer.render.mockReset().mockImplementation(async (_id: string, chart: string) => {
      if (chart.includes("A[")) throw new Error("Parse error");
      return { svg: '<svg aria-label="corrected diagram"></svg>' };
    });

    const { rerender } = render(<Mermaid chart="flowchart TD; A[" />);
    await screen.findByText("Diagram rendering error", {}, { timeout: 2500 });
    rerender(<Mermaid chart="flowchart TD; A-->B" />);

    // Assert the observable recovery rather than an exact call count: the
    // corrected chart legitimately reaches the renderer once per effect pass
    // (debounce settle plus any re-render), so pinning "exactly 2" fails
    // whenever React coalesces or splits those passes differently.
    await waitFor(
      () => expect(renderer.render).toHaveBeenCalledWith(expect.any(String), "flowchart TD; A-->B"),
      { timeout: 2500 },
    );
    await waitFor(
      () => expect(screen.getByLabelText("corrected diagram")).toBeInTheDocument(),
      { timeout: 2500 },
    );
    expect(screen.queryByText("Diagram rendering error")).not.toBeInTheDocument();
  });
});
