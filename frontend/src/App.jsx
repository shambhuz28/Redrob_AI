import {
  BriefcaseBusiness,
  CheckCircle2,
  FileText,
  Loader2,
  Search,
  UploadCloud,
  Users,
} from "lucide-react";
import React from "react";
import { useMemo, useRef, useState } from "react";


const API_URL = (
  import.meta.env.VITE_API_URL || "https://redrob-ai-backend.onrender.com"
).replace(/\/$/, "");
const allowedTypes = [".pdf", ".docx"];
const POLL_INTERVAL_MS = 2000;

export default function App() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [jobId, setJobId] = useState("");
  const [jobStatus, setJobStatus] = useState("");
  const [jobProgress, setJobProgress] = useState(0);
  const activePollRef = useRef("");

  const canUpload = Boolean(file) && !isUploading;
  const candidateCount = result?.rows?.records?.length || 0;
  const fileLabel = useMemo(() => {
    if (!file) return "Select job description";
    return `${file.name} (${formatBytes(file.size)})`;
  }, [file]);

  function handleFileChange(event) {
    const selected = event.target.files?.[0];
    setResult(null);
    setError("");
    setJobId("");
    setJobStatus("");
    setJobProgress(0);
    activePollRef.current = "";

    if (!selected) {
      setFile(null);
      return;
    }

    const lowerName = selected.name.toLowerCase();
    if (!allowedTypes.some((ext) => lowerName.endsWith(ext))) {
      setFile(null);
      setError("Only PDF and DOCX job descriptions are allowed.");
      return;
    }

    setFile(selected);
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file) return;

    setIsUploading(true);
    setError("");
    setResult(null);
    setJobId("");
    setJobStatus("queued");
    setJobProgress(0);

    try {
      const formData = new FormData();
      formData.append("resume", file);

      const response = await fetch(`${API_URL}/api/match`, {
        method: "POST",
        body: formData,
      });

      const payload = await readResponse(response);
      if (!response.ok) {
        throw new Error(payload.error || "Candidate matching failed.");
      }

      if (!payload.jobId) {
        throw new Error("Backend did not return a job ID.");
      }

      const pollToken = `${payload.jobId}-${Date.now()}`;
      activePollRef.current = pollToken;
      setJobId(payload.jobId);

      const completedJob = await pollJob(payload.jobId, pollToken);
      setResult(completedJob);
    } catch (err) {
      if (err.message !== "Candidate matching was cancelled.") {
        setJobStatus("failed");
        setError(err.message || "Candidate matching failed.");
      }
    } finally {
      setIsUploading(false);
    }
  }


  async function pollJob(nextJobId, pollToken) {
    while (activePollRef.current === pollToken) {
      await sleep(POLL_INTERVAL_MS);

      const response = await fetch(`${API_URL}/api/job/${nextJobId}`);
      const payload = await readResponse(response);

      if (!response.ok) {
        throw new Error(payload.error || "Could not read matching job status.");
      }

      setJobStatus(payload.status);
      setJobProgress(payload.progress || 0);

      if (payload.status === "completed") {
        return payload;
      }

      if (payload.status === "failed") {
        throw new Error(payload.error || "Candidate matching failed.");
      }
    }

    throw new Error("Candidate matching was cancelled.");
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Recruiter Candidate Matching</p>
            <h1>Upload a job description to identify top candidates</h1>
          </div>
          <div className="status-pill">
            <CheckCircle2 size={18} aria-hidden="true" />
            Matching pipeline ready
          </div>
        </header>

        <section className="summary-grid" aria-label="Matching summary">
          <SummaryMetric
            icon={<BriefcaseBusiness size={20} aria-hidden="true" />}
            label="Input"
            value={file ? "Job description selected" : "Awaiting job description"}
          />
          <SummaryMetric
            icon={<Search size={20} aria-hidden="true" />}
            label="Status"
            value={getStatusLabel(jobStatus, result, isUploading)}
          />
          <SummaryMetric
            icon={<Users size={20} aria-hidden="true" />}
            label="Candidates"
            value={result ? `${candidateCount} ranked` : "Not generated"}
          />
        </section>

        <form className="upload-panel" onSubmit={handleSubmit}>
          <label className="dropzone" htmlFor="job-description-upload">
            <UploadCloud aria-hidden="true" size={34} />
            <span>{fileLabel}</span>
            <small>Upload a PDF or DOCX job description to generate ranked candidates.</small>
          </label>
          <input
            ref={inputRef}
            id="job-description-upload"
            type="file"
            accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            onChange={handleFileChange}
          />

          <div className="actions">
            <button
              type="button"
              className="secondary-button"
              onClick={() => inputRef.current?.click()}
            >
              <FileText size={18} aria-hidden="true" />
              Choose JD
            </button>
            <button type="submit" className="primary-button" disabled={!canUpload}>
              {isUploading ? (
                <Loader2 className="spin" size={18} aria-hidden="true" />
              ) : (
                <Search size={18} aria-hidden="true" />
              )}
              {isUploading ? "Matching" : "Find Candidates"}
            </button>
          </div>

          {jobStatus && !error && !result && (
            <p className="message info">
              {getStatusLabel(jobStatus, result, isUploading)}
              {jobId ? ` Job ID: ${jobId}` : ""}
              {jobProgress ? ` (${jobProgress}%)` : ""}
            </p>
          )}
          {error && <p className="message error">{error}</p>}
          {result && (
            <p className="message success">
              Completed. Candidate list generated for {result.fileName}. Source CSV: {result.csvPath}.
            </p>
          )}
        </form>

        <CsvTable data={result?.rows} />
      </section>
    </main>
  );
}

function SummaryMetric({ icon, label, value }) {
  return (
    <div className="metric">
      <div className="metric-icon">{icon}</div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function CsvTable({ data }) {
  if (!data) {
    return (
      <section className="table-empty">
        <FileText size={24} aria-hidden="true" />
        <span>Ranked candidates will appear here after matching.</span>
      </section>
    );
  }

  if (!data.headers.length) {
    return (
      <section className="table-empty">
        <FileText size={24} aria-hidden="true" />
        <span>No candidates were returned for this job description.</span>
      </section>
    );
  }

  return (
    <section className="table-panel">
      <div className="table-header">
        <div>
          <h2>Top Candidate Matches</h2>
          <p>Sorted by the ranking produced by the matching pipeline.</p>
        </div>
        <span>{data.records.length} candidates</span>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {data.headers.map((header, index) => (
                <th key={`${header}-${index}`}>{formatHeader(header, index)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.records.map((row, rowIndex) => (
              <tr key={`row-${rowIndex}`}>
                {data.headers.map((_header, cellIndex) => (
                  <td key={`cell-${rowIndex}-${cellIndex}`}>
                    {row[cellIndex] || ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function formatHeader(header, index) {
  if (!header) return `Column ${index + 1}`;
  return header
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1
  );
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}


function getStatusLabel(status, result, isUploading) {
  if (result) return "Completed";
  if (status === "queued") return "Queued...";
  if (status === "running") return "Running...";
  if (status === "completed") return "Completed";
  if (status === "failed") return "Failed";
  return isUploading ? "Matching candidates" : "Ready";
}

function sleep(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

async function readResponse(response) {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    return response.json();
  }

  return { error: await response.text() };
}
