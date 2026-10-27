import { Check, Circle } from "lucide-react";

export type EngineEvent = { kind: string; step?: number; arm?: string; [key: string]: unknown };
const stages = [
    ["capture", "Shared feature capture"],
    ["train", "Joint optimizer updates"],
    ["checkpoint", "Checkpoint recorded"],
    ["generation_verification", "Native generation verification"],
    ["milestone_observer", "Controller observation"],
    ["queue_closed", "Feature queue closed"],
] as const;

// Ported from the experiment UI. Each row requires its own saved engine event;
// neither terminal process status nor later events imply earlier observations.
export function MilestoneTimeline({ events = [] }: { events?: EngineEvent[] }) {
    return <section className="card" aria-label="Saved engine observations">
        <span className="eyebrow">CHECKPOINT VALIDATION</span>
        <h3>Saved engine observations</h3>
        <p className="muted">Observations show recorded stages. Verification details and failures remain in the raw evidence.</p>
        <ol className="milestone-steps">
            {stages.map(([kind, title]) => {
                const event = [...events].reverse().find(row => row.kind === kind);
                return <li key={kind} data-testid="milestone-step" className={event ? "observed" : "unobserved"}>
                    {event ? <Check size={16} /> : <Circle size={12} />}
                    <span>{title}</span><small>{event ? `Recorded at step ${event.step ?? "unknown"}` : "No observation"}</small>
                </li>;
            })}
        </ol>
        {!events.length && <p>No saved events are available for the current attempt.</p>}
    </section>;
}
