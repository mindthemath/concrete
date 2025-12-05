# use the official Bun image
# see all versions at https://hub.docker.com/r/oven/bun/tags
FROM oven/bun:1.3.3 AS base
WORKDIR /usr/src/app

# install dependencies into temp directory
# this will cache them and speed up future builds
FROM base AS install
RUN mkdir -p /temp/dev
COPY frontend/package.json frontend/bun.lock /temp/dev/
RUN cd /temp/dev && bun install --frozen-lockfile

# copy node_modules from temp directory
# then copy all (non-ignored) project files into the image
FROM base AS build
COPY --from=install /temp/dev/node_modules node_modules
COPY frontend .

# build the static site
ENV NODE_ENV=production
ENV VITE_API_ENDPOINT=/predict
RUN bun run build

# final stage: Python base with nginx for static files + API backend
FROM python:3.13-slim AS release

# install nginx
RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx gettext-base && \
    rm -rf /var/lib/apt/lists/*

# create nginx directories and copy static files
RUN mkdir -p /usr/share/nginx/html
COPY --from=build /usr/src/app/dist /usr/share/nginx/html

# copy backend API files and install dependencies
WORKDIR /app
COPY backend/api/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# nginx configuration for SPA with API proxy
RUN rm -f /etc/nginx/sites-enabled/default && \
    echo 'server { \
    listen ${PORT}; \
    server_name _; \
    root /usr/share/nginx/html; \
    index index.html; \
    access_log /dev/stdout; \
    error_log /dev/stderr; \
    location /predict { \
        proxy_pass http://localhost:9600/predict; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
    } \
    location ~ ^/docs(/.*)?$ { \
        proxy_pass http://localhost:9600/docs$1; \
        proxy_set_header Host $http_host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
        proxy_redirect http://localhost:9600/ http://$http_host/; \
        proxy_redirect http://$http_host:9600/ http://$http_host/; \
    } \
    location /openapi.json { \
        proxy_pass http://localhost:9600/openapi.json; \
        proxy_set_header Host $http_host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
    } \
    location /health { \
        proxy_pass http://localhost:9600/health; \
        proxy_set_header Host $http_host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
    } \
    location /info { \
        proxy_pass http://localhost:9600/info; \
        proxy_set_header Host $http_host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
    } \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf.template

# Copy files required for backend API
COPY backend/api/server.py ./
COPY backend/api/data ./data

# copy and set up entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY frontend/data ./data

# ENV DATA_DIR=/app/data

# Port is configurable via PORT env var (Cloud Run sets this automatically)
EXPOSE 3030/tcp
CMD ["/entrypoint.sh"]
