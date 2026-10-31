import { useRef, useState } from "react";
import { UploadCloud } from "lucide-react";

export function Dropzone({ label, accept, onFile, disabled = false }: {
    label: string; accept: string; onFile: (file: File) => void; disabled?: boolean;
}) {
    const input = useRef<HTMLInputElement>(null);
    const [dragging, setDragging] = useState(false);
    return <div className={`dropzone ${dragging ? "dragging" : ""}`}
        onDragOver={event => { event.preventDefault(); if (!disabled) setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={event => {
            event.preventDefault(); setDragging(false);
            if (!disabled && event.dataTransfer.files[0]) onFile(event.dataTransfer.files[0]);
        }}>
        <input ref={input} type="file" accept={accept} aria-label={label} disabled={disabled}
            onChange={event => { const file = event.target.files?.[0]; if (file) onFile(file); event.target.value = ""; }} />
        <button className="button secondary" disabled={disabled} onClick={() => input.current?.click()}>
            <UploadCloud size={18} />{label}
        </button>
        <small>Drop a file here or choose one</small>
    </div>;
}
