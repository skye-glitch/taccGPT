FROM node:20-alpine
WORKDIR /etc/frontend

COPY package.json .
RUN npm i --save react-select
# RUN yarn install --ignore-engines
RUN yarn install
COPY . .
EXPOSE 3000
RUN yarn build

CMD ["npm", "start"]