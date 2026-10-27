// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { MilestoneTimeline } from "./MilestoneTimeline";
afterEach(cleanup);
it("requires a separate saved event for each stage", () => {
    render(<MilestoneTimeline events={[{ kind: "checkpoint", step: 3 }, { kind: "generation_verification", step: 3, passed: false }]} />);
    const rows = screen.getAllByTestId("milestone-step");
    expect(rows[0].textContent).toContain("No observation");
    expect(rows[1].textContent).toContain("No observation");
    expect(rows[2].textContent).toContain("Recorded at step 3");
    expect(rows[3].textContent).toContain("Recorded at step 3");
    expect(screen.queryByText(/Passed|Complete/)).toBeNull();
});
it("shows no fabricated progress before runtime evidence exists", () => {
    render(<MilestoneTimeline />);
    expect(screen.getAllByText("No observation")).toHaveLength(6);
    expect(screen.queryByRole("button")).toBeNull();
});
