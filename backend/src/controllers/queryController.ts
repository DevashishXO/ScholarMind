import { Request, Response } from "express";

export const createQuery = async (req: Request, res: Response): Promise<void> => {
  try {
    const { query } = req.body;

    if (!query) {
      res.status(400).json({ error: "Query is required" });
      return;
    }

    res.json({
      message: "Query recieved successfully",
      Query: query,
    });
  } catch (error) {
    res.status(500).json({ error: "Internal server error" });
  }
};


