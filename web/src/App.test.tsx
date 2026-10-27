// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import App from "./App";

const cfg = { model: { target_model_path: "local-target" }, training: { strategy: "dspark" }, data: {} };
function bootstrap(extra = {}) {
    return { token: "test", engine: { package: "draftfit", support_scope: "Current engine" },
        algorithms: [{ algorithm: "dspark" }], recipes: [{ name: "current-recipe", config: cfg }],
        jobs: [], datasets: [], runs: [{ id: "run", name: "Saved", config: cfg }],
        allow_local_execution: false, ...extra };
}
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
it("validates an imported recipe and runs the actual plan endpoint while training is disabled", async () => {
    const fetcher = vi.fn(async (url: string, init?: RequestInit) => {
        if (url.endsWith("/bootstrap")) return new Response(JSON.stringify(bootstrap()));
        if (url.endsWith("/config/validate")) return new Response(JSON.stringify({ valid: true }));
        if (url.endsWith("/execute")) {
            expect(JSON.parse(String(init?.body)).action).toBe("plan");
            return new Response(JSON.stringify({ id: "job" }));
        }
        return new Response("");
    });
    vi.stubGlobal("fetch", fetcher);
    render(<App />);
    await screen.findByRole("option", { name: "current-recipe" });
    fireEvent.change(screen.getByLabelText("Engine recipe"), { target: { value: "current-recipe" } });
    fireEvent.click(screen.getByRole("button", { name: "Validate config" }));
    await screen.findByText("Configuration is valid. GPU and serving checks have not run.");
    fireEvent.click(screen.getByRole("button", { name: "Experiments" }));
    fireEvent.change(screen.getByLabelText("Experiment"), { target: { value: "run" } });
    expect((screen.getByRole("button", { name: "Train with current engine" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Run plan" }));
    await waitFor(() => expect(fetcher.mock.calls.some(([url]) => url === "/api/runs/run/execute")).toBe(true));
});
it("shows current controller and queue evidence without inferring a speedup", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify(
        url.includes("/status?") ? { state: { status: "failed", total_steps: 2 }, queue: { backpressure: 3 },
            events: [{ kind: "capture", step: 2 }] } : bootstrap()
    ))));
    render(<App />);
    await screen.findByRole("button", { name: "Unattended" });
    fireEvent.click(screen.getByRole("button", { name: "Unattended" }));
    fireEvent.change(screen.getByLabelText("Controller config path"), { target: { value: "controller.json" } });
    fireEvent.click(screen.getByRole("button", { name: "Read saved state" }));
    await screen.findByText("failed");
    expect(screen.getByLabelText("Queue observation").textContent).toContain('"backpressure": 3');
    expect(screen.getAllByTestId("milestone-step")[1].textContent).toContain("No observation");
});
