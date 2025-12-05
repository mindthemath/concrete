# use the official Bun image
# see all versions at https://hub.docker.com/r/oven/bun/tags
FROM oven/bun:1.3.3 AS base
WORKDIR /usr/src/app

# install dependencies into temp directory
# this will cache them and speed up future builds
FROM base AS install
RUN mkdir -p /temp/dev
COPY package.json bun.lock /temp/dev/
RUN cd /temp/dev && bun install --frozen-lockfile

# copy node_modules from temp directory
# then copy all (non-ignored) project files into the image
FROM base AS build
COPY --from=install /temp/dev/node_modules node_modules
COPY . .

# build the static site
ENV NODE_ENV=production
RUN bun run build

# final stage: serve static files with nginx
FROM nginx:alpine AS release

# copy built static files
COPY --from=build /usr/src/app/dist /usr/share/nginx/html

# nginx configuration for SPA (all routes serve index.html)
RUN echo 'server { \
    listen 3000; \
    server_name _; \
    root /usr/share/nginx/html; \
    index index.html; \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 3000/tcp
CMD [ "nginx", "-g", "daemon off;" ]
