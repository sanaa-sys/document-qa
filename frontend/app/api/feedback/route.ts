import { type NextRequest, NextResponse } from 'next/server'

function getBackendBaseUrl(): string | null {
  const raw =
    process.env.FASTAPI_URL ||
    process.env.BACKEND_URL ||
    process.env.RAILWAY_API_URL ||
    process.env.NEXT_PUBLIC_FASTAPI_URL ||
    ''
  const cleaned = raw.trim().replace(/^['"]|['"]$/g, '').replace(/\/$/, '')
  return cleaned || null
}

export async function POST(req: NextRequest) {
  let body: {
    rating?: number
    accuracy?: number
    categories?: string[]
    comments?: string
  }
  try {
    body = await req.json()
  } catch {
    return NextResponse.json({ error: 'Invalid request body.' }, { status: 400 })
  }

  if (!body.comments?.trim()) {
    return NextResponse.json({ error: 'Feedback comments are required.' }, { status: 400 })
  }

  const payload = {
    rating: Math.round(body.rating ?? 0),
    accuracy: Math.round(body.accuracy ?? 0),
    categories: Array.isArray(body.categories) ? body.categories : [],
    comments: body.comments.trim(),
  }

  const FASTAPI_URL = getBackendBaseUrl()
  if (FASTAPI_URL) {
    try {
      await fetch(`${FASTAPI_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
    } catch (err) {
      console.error('[feedback] backend forward failed:', err)
    }
  }

  return NextResponse.json({ ok: true })
}
