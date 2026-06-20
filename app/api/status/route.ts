import { NextResponse } from "next/server";
import { getDB, hasData } from "@/lib/vectorStore";

export async function GET() {
  const loaded = hasData();
  if (!loaded) {
    return NextResponse.json({ loaded: false });
  }
  const db = getDB();

  return NextResponse.json({
    loaded: true,
    scholarName: db.scholarName,
    totalChunks: db.chunks.length,
    uploadedAt: db.uploadedAt,
  });
}
