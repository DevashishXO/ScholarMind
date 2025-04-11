import express, { Request, Response } from "express";
import dotenv from 'dotenv';
import mainRouter from "./routes/mainRouter";

// Load env variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;
app.use(express.json());

app.use("/api/v1", mainRouter);

app.get('/', (req:Request, res:Response) => {
  res.send('Hello from TypeScript + Express + dotenv!');
});

app.listen(PORT, () => {
  console.log(`Server Running on PORT Number: ${PORT}`);
});