# Build multi-stage : compile le frontend Vite, puis le sert avec nginx qui
# reverse-proxy /api/ vers le service "backend" (résolu par DNS Docker interne
# au réseau docker-compose) — le conteneur est autonome, le reverse proxy de
# l'hôte n'a besoin de savoir que celui-ci existe, pas le backend.

FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
