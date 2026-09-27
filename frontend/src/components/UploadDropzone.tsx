import { useRef, useState } from "react";
import type { DocumentInfo } from "../api";

const ACCEPT = ".pdf,.txt,.md,.docx";

interface Props {
  open: boolean;
  documents: DocumentInfo[];
  uploading: boolean;
  error: string | null;
  onClose: () => void;
  onUpload: (files: File[]) => void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function UploadDropzone({ open, documents, uploading, error, onClose, onUpload }: Props) {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4" onClick={onClose}>
      <div
        className="w-full max-w-lg rounded-2xl bg-white p-6 shadow-xl dark:bg-neutral-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Documents</h2>
          <button
            onClick={onClose}
            className="text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-200"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragActive(false);
            onUpload(Array.from(e.dataTransfer.files));
          }}
          onClick={() => inputRef.current?.click()}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-8 text-center transition ${
            dragActive
              ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-500/10"
              : "border-neutral-300 hover:border-indigo-400 dark:border-neutral-700"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPT}
            className="hidden"
            onChange={(e) => e.target.files && onUpload(Array.from(e.target.files))}
          />
          <p className="text-sm text-neutral-600 dark:text-neutral-300">
            {uploading ? "Indexing…" : "Drag & drop files here, or click to browse"}
          </p>
          <p className="mt-1 text-xs text-neutral-400">PDF, TXT, MD, DOCX</p>
        </div>

        {error && <p className="mt-3 text-sm text-red-500">{error}</p>}

        {documents.length > 0 && (
          <ul className="scroll-thin mt-4 max-h-40 space-y-1 overflow-y-auto">
            {documents.map((doc) => (
              <li
                key={doc.name}
                className="flex items-center justify-between rounded-lg px-2 py-1.5 text-sm text-neutral-600 dark:text-neutral-300"
              >
                <span className="truncate">{doc.name}</span>
                <span className="ml-2 shrink-0 text-xs text-neutral-400">{formatSize(doc.size_bytes)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
