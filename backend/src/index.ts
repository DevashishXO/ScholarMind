import express, { Request, Response } from "express";
import dotenv from 'dotenv';

// Load env variables
dotenv.config();

const app = express();
app.use(express.json());

app.use("/api/v1", (req: Request, res: Response) => {
  res.send("Scholar Mind backend");
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server Running on PORT Number: ${PORT}`);
});

module.exports = app;
