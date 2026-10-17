# MotionSense AI — React Frontend

Modern, high-performance machine learning experimentation dashboard for **MotionSense AI**, built with React 19, Vite, TypeScript, Tailwind CSS, Recharts, and Lucide Icons. It connects directly to the FastAPI REST backend service running at `http://127.0.0.1:8000`.

---

## Prerequisites

- **Node.js**: `v20+` or `v24+` (tested on `v24.15.0`)
- **npm**: `v10+` or `v11+` (tested on `11.12.1`)
- **FastAPI Backend**: Running at `http://127.0.0.1:8000` (see [Backend Startup](#backend-startup))

---

## Installation

```bash
# 1. Navigate to the frontend directory
cd D:\HAR\frontend

# 2. Install all dependencies
npm install
```

---

## Environment Variables

The frontend connects to the backend via a single environment variable configured in `.env`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

A template file is provided at `.env.example`. In development mode, Vite also provides a development proxy forwarding `/api/*` to `http://127.0.0.1:8000`.

---

## Running the Application

### 1. Backend Startup

Ensure the Python virtual environment is activated and start the FastAPI service:

```powershell
# From D:\HAR (repository root)
.\venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Verify backend health at:
- Swagger Docs: `http://127.0.0.1:8000/docs`
- Health Endpoint: `http://127.0.0.1:8000/api/health`

### 2. Frontend Development Server

Start Vite in development mode:

```bash
cd D:\HAR\frontend
npm run dev
```

The web application will be available at:
`http://127.0.0.1:5173`

---

## Production Build & Quality Verification

```bash
# Type-check and build production distribution
npm run build

# Run automated Vitest test suite
npm test
```

Production assets are emitted to `dist/`.

---

## Application Routes & Structure

The mini-project interface is consolidated into three focused, responsive views:

| Route | View | Description | Key API Integration |
|---|---|---|---|
| `/` | **Dashboard** | Real-time system monitoring, model benchmark cards (1D-CNN: 94.20%, RandomForest: 92.87%, LSTM: 91.72%, GRU: 91.58%), Recharts model accuracy comparison chart, and 6-activity class legend. | `GET /api/health`<br>`GET /api/models` |
| `/analyzer` | **Activity Analyzer** | Main demonstration screen. **Mode 1**: Select verified benchmark samples (`Walking`, `Sitting`, `Walking Upstairs`). **Mode 2**: Drag-and-drop custom CSV uploads with configurable sliding stride (64 or 128 samples). | `GET /api/samples`<br>`POST /api/predict/sample/{id}`<br>`POST /api/predict/csv` |
| `/results` | **Results & Evaluation** | Combined prediction results and model evaluation. Dominant activity, model agreement rate, timeline chart, window-level prediction table, CSV export, architecture comparison table, and interactive confusion matrix heatmap (normalized % / raw counts). | `GET /api/confusion-matrices`<br>`GET /api/models`<br>Prediction response session state |

---

## Troubleshooting & FAQ

### 1. "Backend Offline (Port 8000)" in Navbar
- Confirm the FastAPI backend is running: `http://127.0.0.1:8000/api/health`
- Check that the port is not blocked: `netstat -ano | findstr 8000`
- Check `VITE_API_BASE_URL` in `D:\HAR\frontend\.env`.

### 2. CORS Errors in Browser Console
- The FastAPI backend includes `CORSMiddleware` permitting `http://localhost:5173` and `http://127.0.0.1:5173`.
- If accessing via a custom hostname, add the origin to `allow_origins` in `backend/main.py`.

### 3. CSV Upload Validation Failures
- The backend requires standard comma-separated text files (`.csv`).
- Ensure recordings have at least **128 rows** (2.56 seconds at 50 Hz).
- Supported schemas:
  - **Option A (9-channel):** `body_acc_x,y,z`, `body_gyro_x,y,z`, `total_acc_x,y,z`
  - **Option B (6-channel raw IMU):** `total_acc_x,y,z`, `body_gyro_x,y,z` (processed automatically with 0.3 Hz Butterworth gravity separation).
