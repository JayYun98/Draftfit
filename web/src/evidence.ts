type Job = {
    status: string;
    created_at?: number;
    metadata: { run_id?: string; action?: string };
};
export function latestStage(
    jobs: Record<string, any>[],
    runId: string | undefined,
    actions: string[],
): Job | null {
    if (!runId) return null;
    return (
        jobs
            .filter(
                (job) =>
                    job.metadata.run_id === runId &&
                    actions.includes(job.metadata.action || ""),
            )
            .map(job => ({status: String(job.status), created_at: Number(job.created_at || 0), metadata: {run_id: String(job.metadata.run_id), action: String(job.metadata.action)}}))
    .sort((a, b) => (b.created_at || 0) - (a.created_at || 0))[0] ||
        null
    );
}
export function validationOutcome(log: string) {
    const passed = log.match(/"passed"\s*:\s*(true|false)/);
    const mismatch = log.match(/"first_mismatch"\s*:\s*(null|[^,\n}]+)/);
    return {
        passed: passed ? passed[1] === "true" : null,
        mismatch: mismatch?.[1] || null,
    };
}
