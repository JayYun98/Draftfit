// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { Dropzone } from "./Dropzone";
afterEach(cleanup);
it("accepts the same file again after an upload and respects the disabled state", () => {
    const onFile = vi.fn();
    const { rerender } = render(<Dropzone label="Upload conversations" accept=".jsonl" onFile={onFile} />);
    const file = new File(["{}"], "data.jsonl");
    const input = screen.getByLabelText("Upload conversations") as HTMLInputElement;
    fireEvent.change(input, { target: { files: [file] } });
    fireEvent.change(input, { target: { files: [file] } });
    expect(onFile).toHaveBeenCalledTimes(2);
    rerender(<Dropzone label="Upload conversations" accept=".jsonl" onFile={onFile} disabled />);
    fireEvent.drop(input.parentElement!, { dataTransfer: { files: [file] } });
    expect(onFile).toHaveBeenCalledTimes(2);
});
