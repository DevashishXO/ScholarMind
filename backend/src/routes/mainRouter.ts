import { Router } from "express";
import { createQuery } from "../controllers/queryController";

const mainRouter = Router();

mainRouter.post("/query", createQuery);

export default mainRouter;
