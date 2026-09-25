# Дашборд — статические файлы, сборка не нужна. nginx слушает 80,
# наружу пробрасывается как 3000 (см. docker-compose.yml: "3000:80").
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html /usr/share/nginx/html/
COPY css /usr/share/nginx/html/css
COPY js /usr/share/nginx/html/js
EXPOSE 80
