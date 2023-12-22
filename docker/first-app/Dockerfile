FROM node:14

RUN echo node image created

WORKDIR /app

COPY package.json .

RUN npm install

COPY . .

EXPOSE 80

CMD ["node","server.js"]

