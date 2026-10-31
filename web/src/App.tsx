import { useCallback, useEffect, useRef, useState } from "react";
import { Activity, Database, Layers3, LoaderCircle, Play, RefreshCw, ShieldCheck, Terminal } from "lucide-react";
import { api, downloadJSON, setToken } from "./api";
import { Dropzone } from "./Dropzone";
import { MilestoneTimeline } from "./MilestoneTimeline";
import type { EngineEvent } from "./MilestoneTimeline";
import { latestStage, validationOutcome } from "./evidence";

type Json = Record<string, any>;
type View = "workflow" | "data" | "runs" | "unattended";
const navigation = [
    ["workflow", "Configure", Layers3], ["data", "Datasets", Database],
    ["runs", "Experiments", Activity], ["unattended", "Unattended", ShieldCheck],
] as const;
const active = (job: Json) => ["queued", "running", "cancelling"].includes(job.status);

export default function App() {
    const [bootstrap, setBootstrap] = useState<Json | null>(null);
    const [view, setView] = useState<View>("workflow");
    const [config, setConfig] = useState("");
    const [name, setName] = useState("Local experiment");
    const [dataset, setDataset] = useState("");
    const [selected, setSelected] = useState("");
    const [message, setMessage] = useState("");
    const [error, setError] = useState("");
    const [busy, setBusy] = useState(false);
    const [preview, setPreview] = useState<Json | null>(null);
    const [jobId, setJobId] = useState("");
    const [log, setLog] = useState("");
    const [controller, setController] = useState("");
    const [observation, setObservation] = useState<Json | null>(null);
    const [expected, setExpected] = useState("");
    const [actual, setActual] = useState("");
    const selection = useRef(0);

    const refresh = useCallback(async () => {
        const value = await api<Json>("/bootstrap");
        setToken(value.token);
        setBootstrap(value);
    }, []);
    useEffect(() => { void refresh().catch(err => setError(String(err.message))); }, [refresh]);
    useEffect(() => {
        if (!bootstrap?.jobs.some(active)) return;
        const timer = window.setInterval(() => {
            void api<Json[]>("/jobs").then(jobs => setBootstrap(previous => previous && ({ ...previous, jobs })))
                .catch(err => setError(String(err.message)));
        }, 1200);
        return () => window.clearInterval(timer);
    }, [bootstrap?.jobs]);
    useEffect(() => {
        const version = ++selection.current;
        setLog("");
        if (!jobId) return;
        let disposed = false;
        const load = () => fetch(`/api/jobs/${jobId}/log`).then(response => {
            if (!response.ok) throw new Error("Could not read the job log.");
            return response.text();
        }).then(value => { if (!disposed && selection.current === version) setLog(value); })
          .catch(err => { if (!disposed) setError(String(err.message)); });
        void load();
        const timer = window.setInterval(() => { void load(); }, 1200);
        return () => { disposed = true; window.clearInterval(timer); };
    }, [jobId]);

    const perform = async (work: () => Promise<void>) => {
        setBusy(true); setError(""); setMessage("");
        try { await work(); } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
        finally { setBusy(false); }
    };
    const loadConfig = async (file: File) => {
        const form = new FormData(); form.append("file", file);
        const result = await api<Json>("/config/import", form);
        setConfig(JSON.stringify(result.config, null, 2)); setPreview(null);
        setMessage("Configuration is valid. GPU and serving checks have not run.");
    };
    const save = async () => {
        const run = await api<Json>("/runs", { name, config: JSON.parse(config), dataset_id: dataset || null });
        await refresh(); setSelected(run.id); setView("runs"); setPreview(null);
        setMessage("Experiment saved. Relative paths are resolved from the project root.");
    };
    const request = async (action: string, execute = false, unattended = false) => {
        const params = unattended ? { config_path: controller } : action === "validate_tokens" ? { expected, actual } : {};
        const path = unattended ? "/unattended" : `/runs/${selected}`;
        const result = await api<Json>(`${path}/${execute ? "execute" : "preview"}`, { action, params });
        if (execute) { await refresh(); setJobId(result.id); setMessage(`Started ${action}.`); }
        else setPreview(result);
    };
    const readObservation = async () => {
        const result = await api<Json>(`/unattended/status?config_path=${encodeURIComponent(controller)}`);
        setObservation(result);
    };
    const run = bootstrap?.runs.find((item: Json) => item.id === selected);
    const job = bootstrap?.jobs.find((item: Json) => item.id === jobId);
    const evidence = job?.metadata.action === "validate_tokens" ? validationOutcome(log) : null;
    const latest = latestStage(bootstrap?.jobs || [], run?.id, ["train"]);
    const blocked = busy || Boolean(bootstrap?.jobs.some(active));
    const commands = preview?.commands as string[][] | undefined;

    return <div className="app-shell">
        <aside className="sidebar">
            <div className="brand"><Layers3 size={24} /><strong>Draftfit</strong></div>
            <span className="sidebar-label">LOCAL WORKSPACE</span>
            <nav aria-label="Workspace">{navigation.map(([id, title, Icon]) =>
                <button key={id} className={view === id ? "selected" : ""} aria-current={view === id ? "page" : undefined}
                    onClick={() => { setView(id); setPreview(null); }}><Icon size={18} />{title}</button>)}</nav>
            <p className="sidebar-note">Current Draftfit engine<br />Local execution and saved evidence</p>
        </aside>
        <main>
            <header className="page-head"><div><span className="eyebrow">EXPERIMENT WORKSPACE</span>
                <h1>{navigation.find(([id]) => id === view)?.[1]}</h1></div>
                <button className="button secondary" disabled={busy} onClick={() => void perform(refresh)}><RefreshCw size={15} />Refresh</button></header>
            {error && <div className="notice error" role="alert">{error}</div>}
            {message && <div className="notice" role="status">{message}</div>}
            {!bootstrap ? <p>Connecting to the local engine…</p> : <>
                {view === "workflow" && <>
                    <section className="card intro"><h2>Configure an experiment</h2>
                        <p>Import a training config or use a current engine recipe. Validate it before saving.</p>
                        <div className="algorithm-list">{bootstrap.algorithms.map((item: Json) => <span key={item.algorithm} className="tag">{item.algorithm}</span>)}</div>
                    </section>
                    <section className="card form-grid">
                        <label>Experiment name<input value={name} onChange={event => setName(event.target.value)} /></label>
                        <label>Engine recipe<select aria-label="Engine recipe" value="" onChange={event => {
                            const recipe = bootstrap.recipes.find((item: Json) => item.name === event.target.value);
                            if (recipe) { setConfig(JSON.stringify(recipe.config, null, 2)); setPreview(null); }
                        }}><option value="">Choose a recipe</option>{bootstrap.recipes.map((recipe: Json) =>
                            <option value={recipe.name} key={recipe.name}>{recipe.name}</option>)}</select></label>
                        <div className="full"><Dropzone label="Import JSON / YAML config" accept=".json,.yaml,.yml" disabled={busy} onFile={file => void perform(() => loadConfig(file))} /></div>
                        <label className="full">Training config<textarea aria-label="Training config" rows={18} spellCheck={false} value={config}
                            placeholder="Import a config or choose an engine recipe" onChange={event => { setConfig(event.target.value); setPreview(null); }} /></label>
                        <label className="full">Prepared dataset (online configurations only)<select value={dataset} onChange={event => setDataset(event.target.value)}>
                            <option value="">Keep the config's existing data paths</option>{bootstrap.datasets.map((item: Json) => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label>
                        <div className="actions full"><button className="button secondary" disabled={busy || !config} onClick={() => void perform(async () => {
                            await api("/config/validate", JSON.parse(config)); setMessage("Configuration is valid. GPU and serving checks have not run.");
                        })}>Validate config</button><button className="button" disabled={busy || !config || !name.trim()} onClick={() => void perform(save)}>Save experiment</button>
                            <button className="button secondary" disabled={!config} onClick={() => void perform(async () => downloadJSON(JSON.parse(config), "train.json"))}>Download config</button></div>
                    </section>
                </>}
                {view === "data" && <section className="card"><h2>Prepare conversation data</h2>
                    <p>JSON / JSONL, up to 64 MiB. The engine prepares a grouped holdout for separate capture or verification. Single-run online evaluation is unavailable; offline runs require captured features.</p>
                    <Dropzone label="Upload conversations" accept=".json,.jsonl" disabled={busy} onFile={file => void perform(async () => {
                        const form = new FormData(); form.append("file", file); form.append("split_eval", "true");
                        const result = await api<Json>("/datasets", form); await refresh(); setDataset(result.id);
                        setMessage(`Prepared ${result.train_rows} training rows and ${result.eval_rows} holdout rows.`);
                    })} />
                    <div className="table-wrap"><table><thead><tr><th>Dataset</th><th>Train</th><th>Holdout</th></tr></thead><tbody>
                        {bootstrap.datasets.map((item: Json) => <tr key={item.id}><td>{item.name}<small>{item.sha256}</small></td><td>{item.train_rows}</td><td>{item.eval_rows}</td></tr>)}</tbody></table></div>
                </section>}
                {view === "runs" && <section className="card"><h2>Saved experiments</h2>
                    <label>Experiment<select value={selected} onChange={event => { setSelected(event.target.value); setPreview(null); }}>
                        <option value="">Choose an experiment</option>{bootstrap.runs.map((item: Json) => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label>
                    {run && <><p className="muted">{run.config.training.strategy} · {run.config.model.target_model_path} · Latest training process: {latest?.status || "not run"}</p>
                        <div className="actions"><button className="button secondary" disabled={busy} onClick={() => void perform(() => request("plan"))}>Preview plan command</button>
                            <button className="button" disabled={blocked} onClick={() => void perform(() => request("plan", true))}>Run plan</button>
                            <button className="button secondary" disabled={blocked || !bootstrap.allow_local_execution} onClick={() => void perform(() => request("train", true))}><Play size={14} />Train with current engine</button></div>
                        <h3>Token parity</h3><p className="muted">Compare existing token-ID JSON arrays. A passing comparison covers only these inputs.</p>
                        <div className="form-grid"><label>Expected JSON path<input value={expected} onChange={event => setExpected(event.target.value)} /></label>
                            <label>Actual JSON path<input value={actual} onChange={event => setActual(event.target.value)} /></label></div>
                        <button className="button secondary" disabled={blocked || !expected || !actual} onClick={() => void perform(() => request("validate_tokens", true))}>Compare token IDs</button></>}
                </section>}
                {view === "unattended" && <>
                    <section className="card"><h2>Continue the existing controller</h2>
                        <p>Select an existing controller config inside this project. Its bounded queue, budget admission, checkpoint recovery and verification policy remain engine-owned.</p>
                        <label>Controller config path<input aria-label="Controller config path" disabled={busy} value={controller} placeholder="outputs/campaign/controller.json" onChange={event => {
                            setController(event.target.value); setObservation(null); setPreview(null);
                        }} /></label>
                        <div className="actions"><button className="button secondary" disabled={busy || !controller} onClick={() => void perform(() => request("unattended_plan", false, true))}>Preview plan command</button>
                            <button className="button" disabled={blocked || !controller} onClick={() => void perform(() => request("unattended_plan", true, true))}>Run unattended plan</button>
                            <button className="button secondary" disabled={busy || !controller} onClick={() => void perform(readObservation)}>Read saved state</button>
                            {(["unattended", "unattended_resume"] as const).map(action => <button key={action} className="button secondary"
                                disabled={blocked || !controller || !bootstrap.allow_local_execution} onClick={() => void perform(() => request(action, true, true))}>
                                {action === "unattended" ? "Start controller" : "Resume controller"}</button>)}</div>
                        {observation && <><p>Controller status: <strong>{observation.state?.status || "not started"}</strong> · Completed updates: {observation.state?.total_steps ?? "unknown"}</p>
                            <h3>Latest saved queue observation</h3><pre aria-label="Queue observation">{observation.queue ? JSON.stringify(observation.queue, null, 2) : "No queue observation"}</pre>
                            <details><summary>Raw controller state and evidence</summary><pre>{JSON.stringify(observation, null, 2)}</pre></details></>}
                    </section>
                    <MilestoneTimeline events={(observation?.events || []) as EngineEvent[]} />
                </>}
                {!bootstrap.allow_local_execution && (view === "runs" || view === "unattended") && <p className="muted">Training controls require the server's --allow-local-execution option and a provisioned trainer interpreter.</p>}
                {commands && <section className="card"><h3>Command preview</h3><pre>{commands.map(command => command.map(arg => JSON.stringify(arg)).join(" ")).join("\n")}</pre>
                    <p className="muted">Working directory: {preview?.cwd}. Previewing a command does not execute or validate the plan.</p></section>}
                {(view === "runs" || view === "unattended") && <section className="card"><h3><Terminal size={17} /> Workspace processes</h3>
                    {!bootstrap.jobs.length && <p>No workspace jobs yet.</p>}
                    <div className="job-list">{bootstrap.jobs.map((item: Json) => <button className={`job ${jobId === item.id ? "selected" : ""}`} key={item.id} onClick={() => setJobId(item.id)}>
                        <span>{item.metadata.action}</span><span>{item.status}</span></button>)}</div>
                    {job && <><div className="actions"><span>Process: {job.status} · Exit: {job.exit_code ?? "pending"}</span>
                        {active(job) && <button className="button secondary" disabled={busy} onClick={() => void perform(async () => { await api(`/jobs/${job.id}/cancel`, {}); await refresh(); })}>Stop process</button>}</div>
                        <pre aria-label="Job output">{log || "Waiting for output…"}</pre>
                        {evidence && <p>Token comparison: {evidence.passed === null ? "No result recorded" : evidence.passed ? "Passed for supplied inputs" : "Failed"}</p>}
                        <p className="muted">A successful process is not evidence of GPU correctness, compatible serving or a speedup.</p></>}
                </section>}
                <footer>Engine: {bootstrap.engine.package} · {bootstrap.engine.support_scope}<br />Remote rental, publishing, research adapters and arbitrary commands are unavailable in this workspace.</footer>
            </>}
            {busy && <span className="working" role="status"><LoaderCircle className="spin" size={15} />Working…</span>}
        </main>
    </div>;
}
