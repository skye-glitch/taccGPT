FROM node:20-alpine
WORKDIR /etc/frontend

COPY package.json .
RUN npm i --save react-select
RUN yarn install
#COPY . .
EXPOSE 3000
RUN npm i @types/react
# RUN yarn build
RUN npm run build

CMD ["npm", "start"]