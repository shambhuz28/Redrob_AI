const cors = require("cors");
const dotenv = require("dotenv");
const express = require("express");
const fs = require("fs");
const multer = require("multer");
const path = require("path");
const { spawn } = require("child_process");
const { randomUUID } = require("crypto");

dotenv.config({ path: path.join(__dirname, "..", ".env") });

const app = express();
const PORT = Number(process.env.PORT || 5000);
const FRONTEND_ORIGINS = parseOrigins(process.env.FRONTEND_ORIGIN);
const BACKEND_DIR = path.resolve(__dirname, "..");
const UPLOAD_DIR = path.join(BACKEND_DIR, "uploaded");
const JOB_RESULTS_DIR = path.join(BACKEND_DIR, "job-results");
const DEFAULT_OUTPUT_CSV = path.join(BACKEND_DIR, "ranked_candidates.csv");
const JOB_TTL_MS = readPositiveNumber(process.env.JOB_TTL_MS, 30 * 60 * 1000);
const JOB_CLEANUP_INTERVAL_MS = readPositiveNumber(
  process.env.JOB_CLEANUP_INTERVAL_MS,
  5 * 60 * 1000
);
const jobs = new Map();

fs.mkdirSync(UPLOAD_DIR, { recursive: true });
fs.mkdirSync(JOB_RESULTS_DIR, { recursive: true });

app.use(cors({ origin: corsOrigin }));
app.use(express.json());

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const base = path
      .basename(file.originalname, ext)
      .replace(/[^a-z0-9-_]+/gi, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 80);
    cb(null, `${Date.now()}-${base || "upload"}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: 20 * 1024 * 1024,
  },
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (ext === ".pdf" || ext === ".docx") {
      cb(null, true);
      return;
    }
    cb(new Error("Only PDF and DOCX files are allowed."));
  },
});

app.get("/api/health", (_req, res) => {
  res.json({
    status: "ok",
    pythonScript: process.env.PYTHON_SCRIPT || "redrob_backend.py",
    activeJobs: jobs.size,
  });
});

app.post("/api/match", upload.single("resume"), createMatchJob);

// Backward-compatible alias. It now returns a jobId instead of waiting for Python.
app.post("/api/uploaded", upload.single("resume"), createMatchJob);

app.get("/api/job/:jobId", (req, res) => {
  const job = jobs.get(req.params.jobId);

  if (!job) {
    res.status(404).json({ error: "Job not found.", code: "job_not_found" });
    return;
  }

  res.json(serializeJob(job));
});

app.use((error, _req, res, _next) => {
  if (error instanceof multer.MulterError) {
    res.status(400).json({ error: error.message });
    return;
  }

  res.status(error.status || 500).json({
    error: error.message || "Something went wrong.",
    code: error.code || "backend_error",
  });
});

app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});

setInterval(cleanupJobs, JOB_CLEANUP_INTERVAL_MS).unref();


function parseOrigins(value) {
  const fallback = "http://localhost:5173,https://redrob-ai-frontend-tau.vercel.app";
  return (value || fallback)
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean);
}

function corsOrigin(origin, callback) {
  if (!origin || FRONTEND_ORIGINS.includes("*") || FRONTEND_ORIGINS.includes(origin)) {
    callback(null, true);
    return;
  }

  callback(new Error(`CORS blocked origin: ${origin}`));
}

function readPositiveNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

function createMatchJob(req, res, next) {
  try {
    if (!req.file) {
      res.status(400).json({ error: "Upload a PDF or DOCX file." });
      return;
    }

    const jobId = randomUUID();
    const uploadedPath = path.resolve(req.file.path);
    const outputCsvPath = path.join(JOB_RESULTS_DIR, `${jobId}.csv`);
    const job = {
      jobId,
      status: "queued",
      progress: 0,
      result: null,
      error: null,
      fileName: req.file.originalname,
      storedPath: uploadedPath,
      outputCsvPath,
      stdout: "",
      stderr: "",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      completedAt: null,
      child: null,
    };

    jobs.set(jobId, job);
    setImmediate(() => startPythonJob(job));
    res.status(202).json({ jobId });
  } catch (error) {
    next(error);
  }
}

function startPythonJob(job) {
  let child;

  try {
    const scriptPath = resolvePythonScript();
    const pythonCommand = process.env.PYTHON_COMMAND || "python3";
    const timeoutMs = readPositiveNumber(process.env.PYTHON_TIMEOUT_MS, 300000);

    updateJob(job, { status: "running", progress: 10 });

    child = spawn(pythonCommand, [scriptPath, job.storedPath], {
      cwd: BACKEND_DIR,
      env: {
        ...process.env,
        CSV_OUTPUT_PATH: job.outputCsvPath,
      },
      shell: false,
    });

    job.child = child;

    const timeout = setTimeout(() => {
      if (job.status === "running") {
        child.kill("SIGTERM");
        setTimeout(() => child.kill("SIGKILL"), 5000).unref();
        failJob(job, `Candidate matching timed out after ${Math.round(timeoutMs / 1000)} seconds.`);
      }
    }, timeoutMs);

    child.stdout.on("data", (data) => {
      const text = data.toString();
      job.stdout += text;
      updateProgressFromOutput(job, text);
    });

    child.stderr.on("data", (data) => {
      job.stderr += data.toString();
      touchJob(job);
    });

    child.on("error", (error) => {
      clearTimeout(timeout);
      failJob(job, error.message || "Failed to start Python process.");
    });

    child.on("close", async (code) => {
      clearTimeout(timeout);

      if (job.status === "failed") {
        return;
      }

      if (code !== 0) {
        failJob(
          job,
          `Python script failed with exit code ${code}.${job.stderr ? ` ${job.stderr}` : ""}`
        );
        return;
      }

      try {
        updateJob(job, { progress: 90 });
        const csvResult = await resolveCsvResult(job.stdout, job.outputCsvPath);

        completeJob(job, {
          fileName: job.fileName,
          storedPath: job.storedPath,
          csvPath: csvResult.path,
          rows: parseCsv(csvResult.content),
          python: {
            stdout: job.stdout.trim(),
            stderr: job.stderr.trim(),
          },
        });
      } catch (error) {
        failJob(job, error.message || "Candidate matching failed.");
      }
    });
  } catch (error) {
    if (child) {
      child.kill("SIGTERM");
    }
    failJob(job, error.message || "Candidate matching failed.");
  }
}

function resolvePythonScript() {
  const configuredScript = process.env.PYTHON_SCRIPT || "redrob_backend.py";
  const scriptPath = path.resolve(BACKEND_DIR, configuredScript);

  if (!scriptPath.startsWith(BACKEND_DIR + path.sep) || !scriptPath.endsWith(".py")) {
    const error = new Error("PYTHON_SCRIPT must point to a .py file inside backend.");
    error.status = 500;
    throw error;
  }

  if (!fs.existsSync(scriptPath)) {
    const error = new Error(`Python script not found: ${scriptPath}`);
    error.status = 500;
    throw error;
  }

  return scriptPath;
}

function updateProgressFromOutput(job, text) {
  if (/JD ingestion/i.test(text)) {
    updateJob(job, { progress: Math.max(job.progress, 25) });
    return;
  }

  if (/Semantic search/i.test(text)) {
    updateJob(job, { progress: Math.max(job.progress, 45) });
    return;
  }

  if (/Ranked candidates/i.test(text)) {
    updateJob(job, { progress: Math.max(job.progress, 70) });
    return;
  }

  if (/Saved .* candidates/i.test(text)) {
    updateJob(job, { progress: Math.max(job.progress, 85) });
    return;
  }

  touchJob(job);
}

function updateJob(job, updates) {
  Object.assign(job, updates);
  touchJob(job);
}

function touchJob(job) {
  job.updatedAt = Date.now();
}

function completeJob(job, result) {
  updateJob(job, {
    status: "completed",
    progress: 100,
    result,
    error: null,
    child: null,
    completedAt: Date.now(),
  });
}

function failJob(job, error) {
  updateJob(job, {
    status: "failed",
    progress: 100,
    result: null,
    error,
    child: null,
    completedAt: Date.now(),
  });
}

function serializeJob(job) {
  const response = {
    jobId: job.jobId,
    status: job.status,
    progress: job.progress,
  };

  if (job.status === "completed" && job.result) {
    return { ...response, ...job.result };
  }

  if (job.status === "failed") {
    return { ...response, error: job.error || "Candidate matching failed." };
  }

  return response;
}

function cleanupJobs() {
  const now = Date.now();

  for (const [jobId, job] of jobs.entries()) {
    if (!job.completedAt) {
      continue;
    }

    if (now - job.completedAt > JOB_TTL_MS) {
      jobs.delete(jobId);
      deleteJobFiles(job);
    }
  }
}

function deleteJobFiles(job) {
  for (const filePath of [job.storedPath, job.outputCsvPath]) {
    if (!filePath || !isPathInside(BACKEND_DIR, filePath)) {
      continue;
    }

    fs.promises.unlink(filePath).catch(() => {});
  }
}

function isPathInside(parent, child) {
  const relative = path.relative(parent, child);
  return relative && !relative.startsWith("..") && !path.isAbsolute(relative);
}

async function resolveCsvResult(stdout, preferredCsvPath) {
  for (const csvPath of resolveCsvPaths(stdout, preferredCsvPath)) {
    if (csvPath && fs.existsSync(csvPath)) {
      return {
        path: csvPath,
        content: await fs.promises.readFile(csvPath, "utf8"),
      };
    }
  }

  if (looksLikeCsv(stdout)) {
    return {
      path: "stdout",
      content: stdout,
    };
  }

  const error = new Error(
    "Python script finished, but no CSV file was found. Return/print a .csv path, set CSV_OUTPUT_PATH, or print CSV content."
  );
  error.status = 500;
  throw error;
}

function resolveCsvPaths(stdout, preferredCsvPath) {
  const paths = [
    preferredCsvPath,
    process.env.CSV_OUTPUT_PATH,
    extractCsvPath(stdout),
    DEFAULT_OUTPUT_CSV,
  ].filter(Boolean);

  return [...new Set(paths.map(normalizeBackendPath))];
}

function extractCsvPath(stdout) {
  const lines = stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .reverse();

  for (const line of lines) {
    const absoluteMatches = line.match(/(?:[A-Za-z]:[\\/]|\/)[^\r\n"<>|]+?\.csv/gi);
    if (absoluteMatches?.length) {
      return absoluteMatches[absoluteMatches.length - 1].trim();
    }
  }

  for (const line of lines) {
    const tokenMatches = line.match(/[^\s"<>|]+\.csv/gi);
    if (tokenMatches?.length) {
      return tokenMatches[tokenMatches.length - 1].trim();
    }
  }

  return null;
}

function looksLikeCsv(stdout) {
  const lines = stdout
    .trim()
    .split(/\r?\n/)
    .filter(Boolean);
  return lines.length > 0 && lines[0].includes(",");
}

function normalizeBackendPath(filePath) {
  return path.isAbsolute(filePath)
    ? path.resolve(filePath)
    : path.resolve(BACKEND_DIR, filePath);
}

function parseCsv(csv) {
  const rows = [];
  let row = [];
  let value = "";
  let inQuotes = false;

  for (let i = 0; i < csv.length; i += 1) {
    const char = csv[i];
    const next = csv[i + 1];

    if (char === '"' && inQuotes && next === '"') {
      value += '"';
      i += 1;
      continue;
    }

    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }

    if (char === "," && !inQuotes) {
      row.push(value);
      value = "";
      continue;
    }

    if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") {
        i += 1;
      }
      row.push(value);
      if (row.some((cell) => cell.length > 0)) {
        rows.push(row);
      }
      row = [];
      value = "";
      continue;
    }

    value += char;
  }

  row.push(value);
  if (row.some((cell) => cell.length > 0)) {
    rows.push(row);
  }

  if (rows.length === 0) {
    return { headers: [], records: [] };
  }

  const [headers, ...records] = rows;
  return { headers, records };
}
