<p align="center">
  <img src="https://img.shields.io/badge/Dharura-AI-red?style=for-the-badge&logo=mapbox&logoColor=white" alt="Dharura AI" />
</p>

<h1 align="center">🚨 Dharura AI</h1>

<p align="center">
  <strong>Real-time AI-powered emergency intelligence system for Kenya.</strong><br/>
  <em>"Dharura" means <strong>Emergency</strong> in Swahili.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NestJS-E0234E?style=flat&logo=nestjs&logoColor=white" />
  <img src="https://img.shields.io/badge/Next.js-000000?style=flat&logo=nextdotjs&logoColor=white" />
  <img src="https://img.shields.io/badge/Prisma-2D3748?style=flat&logo=prisma&logoColor=white" />
  <img src="https://img.shields.io/badge/Supabase-3ECF8E?style=flat&logo=supabase&logoColor=white" />
  <img src="https://img.shields.io/badge/Socket.io-010101?style=flat&logo=socketdotio&logoColor=white" />
  <img src="https://img.shields.io/badge/TypeScript-3178C6?style=flat&logo=typescript&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
</p>

---

## 📖 Overview

Dharura AI transforms citizens into **geo-sensors** — enabling anyone to drop a geotagged emergency pin on a live map, describe the incident, and have it instantly classified by an AI risk engine. Severity alerts are then dispatched in real-time to a responder command center via WebSockets.

Built for the unique emergency response landscape of Kenya, Dharura AI bridges the gap between citizens witnessing emergencies and first responders who need actionable, prioritized intelligence.

---

## ✨ Features

- **🗺️ Interactive Live Map** — React-Leaflet powered map centered on Kenya with real-time incident pins
- **🤖 AI Risk Engine** — Scores emergencies based on keyword triggers, incident type, and spatial-temporal clustering (+15 confidence per nearby event within 2km/10 minutes)
- **⚡ Real-time WebSocket Dispatch** — Socket.IO gateway broadcasts new reports instantly to all connected responders
- **🎯 Severity Classification** — Automatically categorizes incidents as `LOW`, `WARNING`, or `CRITICAL`
- **🧑‍💼 Responder Command Center** — JWT-secured dashboard at `/responder` for acknowledging, routing, and resolving incidents
- **🌙 Light / Dark Mode** — Full theme support via `next-themes` with glassmorphism UI
- **🔐 Auth System** — Hand-rolled JWT + Passport + bcrypt authentication for responders

---

## 🏗️ Architecture

```
Dharura_AI/
├── backend/          # NestJS API + WebSocket Gateway
└── frontend/         # Next.js App Router + React-Leaflet Map
```

### Backend — NestJS (Port 3001)

| Module | Description |
|--------|-------------|
| **Auth** | JWT-based authentication with Passport local strategy and bcrypt password hashing |
| **Reports** | REST API for creating and managing emergency reports, houses the AI Risk Engine |
| **Events Gateway** | Socket.IO gateway that emits `newReport` events to all connected clients in real-time |

### Frontend — Next.js App Router (Port 3000)

| Component | Description |
|-----------|-------------|
| **MapClient** | Dynamically loaded (SSR disabled) Leaflet map with real-time report pins and confidence score popups |
| **ReportModal** | Citizen-facing incident submission form triggered by clicking anywhere on the map |
| **Responder Dashboard** | `/responder` — JWT-secured command center for managing and resolving incoming reports |

### Database — Supabase PostgreSQL via Prisma

- Schema migrations via `prisma db push` on port **5432** (direct connection)
- App runtime connections via **port 6543** (IPv4 Transaction Pooler)

---

## 🧠 AI Risk Engine

The AI Risk Engine scores each incoming report using a weighted algorithm:

- **Base score** from incident category (e.g. `FIRE`, `MEDICAL`, `FLOOD`)
- **Keyword multiplier** — descriptions containing words like `flames`, `burning`, `unconscious`, `collapsed` boost the score
- **Spatial-temporal clustering** — +15 confidence points for each existing report within **2km** and **10 minutes** of the new report

Final scores map to severity bands:

| Score | Severity |
|-------|----------|
| 0–39 | `LOW` |
| 40–69 | `WARNING` |
| 70–100 | `CRITICAL` |

> 💡 **Tip:** Submit multiple emergencies at the same location within 10 minutes to watch the AI Confidence Score organically climb to `CRITICAL`!

---

## 🚀 Getting Started

### Prerequisites

- Node.js v18+
- npm v9+
- A Supabase project with PostgreSQL enabled

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/dharura-ai.git
cd dharura-ai
```

### 2. Configure Environment Variables

**Backend** — create `backend/.env`:

```env
DATABASE_URL="postgresql://USER:PASSWORD@HOST:6543/postgres?pgbouncer=true"
DIRECT_URL="postgresql://USER:PASSWORD@HOST:5432/postgres"
JWT_SECRET="your-super-secret-jwt-key"
PORT=3001
```

**Frontend** — create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:3001
NEXT_PUBLIC_WS_URL=http://localhost:3001
```

### 3. Start the Backend

```bash
cd backend
npm install
npx prisma db push
npm run start:dev
```

### 4. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## 🖥️ Usage

### Public Map — `http://localhost:3000`

1. The map loads centered on Kenya with dynamic light/dark theme support
2. Click anywhere on the map to open the incident report modal
3. Select a category (e.g. `FIRE`, `MEDICAL`, `FLOOD`, `SECURITY`) and describe the incident
4. Submit — your pin appears on the map instantly for all connected users

### Responder Command Center — `http://localhost:3000/responder`

1. Register a responder account via the API:
   ```bash
   POST http://localhost:3001/auth/register
   { "email": "responder@dharura.ke", "password": "securepassword" }
   ```
2. Log in through the dashboard UI
3. View incoming reports sorted by severity and AI confidence score
4. Update report status: `PENDING` → `ACKNOWLEDGED` → `EN_ROUTE` → `RESOLVED`

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend Framework | NestJS with TypeScript |
| Real-time | Socket.IO Gateway |
| Authentication | Passport.js + JWT + bcrypt |
| ORM | Prisma v5 |
| Database | Supabase PostgreSQL |
| Frontend Framework | Next.js 15 (App Router) |
| Map | React-Leaflet |
| Styling | Tailwind CSS v4 + Glassmorphism |
| Theming | next-themes |
| Font | Poppins (Google Fonts) |

---

## 📡 API Reference

### Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/register` | Register a new responder |
| `POST` | `/auth/login` | Login and receive JWT token |

### Reports

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/reports` | Fetch all reports |
| `POST` | `/reports` | Submit a new emergency report |
| `PATCH` | `/reports/:id/status` | Update report status (auth required) |

### WebSocket Events

| Event | Direction | Payload |
|-------|-----------|---------|
| `newReport` | Server → Client | Full report object with AI score |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](../LICENSE) file for details.

---

## 👤 Author

**Melckzedek Kaisha**

Built with 🔥 for Kenya's emergency response future.
