# Nya AI — Production Deployment Guide (Cloudflare & Docker)

This guide walks you through deploying Nya AI to production using **Cloudflare** and **Docker**.

---

## Architecture Overview

Since Nya AI uses FastAPI with PyTorch & SentenceTransformers:
* FastAPI mounts the `frontend/` directory directly at `/`, serving **both** the backend API and frontend UI on the same origin (no CORS issues).
* You can deploy the complete container and route your traffic through Cloudflare.

---

## Deployment Option 1: Docker + Cloudflare Tunnel (Recommended ⭐)

This is the cleanest, easiest setup. You run the container on any Linux VPS (DigitalOcean, Hetzner, AWS EC2, or your server), and Cloudflare Tunnel securely connects it to your domain without opening any firewall ports or managing SSL certs.

### Step 1: Create a Cloudflare Tunnel
1. Go to the [Cloudflare Zero Trust Dashboard](https://one.dash.cloudflare.com/).
2. Navigate to **Networks** → **Tunnels** → Click **Create a tunnel**.
3. Choose **Cloudflared** and name your tunnel (e.g., `nyaai-prod`).
4. Copy your **Tunnel Token** (the long alphanumeric string).

### Step 2: Configure Public Hostname
In the Cloudflare tunnel settings:
* **Subdomain / Domain:** e.g., `nya.yourdomain.com`
* **Type:** `HTTP`
* **URL:** `nyaai:8000` (or `localhost:8000` if running without docker-compose network)

### Step 3: Run with Docker Compose
On your server or machine:
1. Copy your `.env` file into `Major/backend/.env`.
2. Edit `docker-compose.yml` and uncomment the `tunnel` service:
   ```yaml
   tunnel:
     image: cloudflare/cloudflared:latest
     container_name: nyaai_tunnel
     restart: unless-stopped
     command: tunnel run
     environment:
       - TUNNEL_TOKEN=your_copied_token_here
     depends_on:
       - nyaai
   ```
3. Start the application:
   ```bash
   docker compose up -d --build
   ```
4. Done! Your app is now live at `https://nya.yourdomain.com` with free Cloudflare SSL, DDoS protection, and CDN caching.

---

## Deployment Option 2: Split Deployment (Cloudflare Pages + Render / Fly.io)

If you prefer serverless hosting for the frontend:

### 1. Deploy the Backend (Docker)
1. Push your repository to GitHub.
2. In [Render.com](https://render.com) or [Fly.io](https://fly.io) or [Railway.app](https://railway.app):
   * Create a new **Web Service**.
   * Select your GitHub repository.
   * Choose **Docker** as the environment.
   * Add a persistent disk mounted to `/app/backend/data` (to keep your SQLite database and embeddings safe across deploys).
   * Note down your backend URL: e.g. `https://nyaai-backend.onrender.com`.

### 2. Deploy the Frontend (Cloudflare Pages)
1. Go to **Cloudflare Dashboard** → **Workers & Pages** → **Create application** → **Pages**.
2. Connect your Git repository.
3. Configure build settings:
   * **Root directory:** `frontend`
   * **Build command:** *(leave empty)*
   * **Build output directory:** `.`
4. In `frontend/_redirects`, set your backend URL:
   ```
   /cases/*            https://nyaai-backend.onrender.com/cases/:splat            200
   /historical-cases*  https://nyaai-backend.onrender.com/historical-cases:splat  200
   /api/*              https://nyaai-backend.onrender.com/api/:splat              200
   /*                  /index.html                                                200
   ```
5. Deploy! Cloudflare Pages will serve your frontend globally and automatically proxy API calls to your backend container.

---

## Useful Docker Commands

```bash
# Build the Docker image locally
docker build -t nyaai:latest .

# Run container with persistent data
docker run -d -p 8000:8000 \
  -v "$(pwd)/backend/data:/app/backend/data" \
  --name nyaai_app \
  nyaai:latest

# Check container logs
docker logs -f nyaai_app

# Stop container
docker stop nyaai_app
```
