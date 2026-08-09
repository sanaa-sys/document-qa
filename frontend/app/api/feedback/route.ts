import { type NextRequest, NextResponse } from 'next/server'

const FASTAPI_URL = process.env.FASTAPI_URL

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

  // Matches main.py's FeedbackRequest: { rating, accuracy, categories, comments }.
  const payload = {
    rating: Math.round(body.rating ?? 0),
    accuracy: Math.round(body.accuracy ?? 0),
    categories: Array.isArray(body.categories) ? body.categories : [],
    comments: body.comments.trim(),
  }

  // If a backend is configured, forward the feedback to FastAPI (main.py).
  // Otherwise just log it so the form still works during development.
  if (!FASTAPI_URL) {
    console.log('[v0] Feedback received (no FASTAPI_URL set):', payload)
    return NextResponse.json({ ok: true })
  }

  try {
    const upstream = await fetch(`${FASTAPI_URL.replace(/\/$/, '')}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    if (!upstream.ok) {
      const detail = await upstream.text().catch(() => '')
      console.log('[v0] Feedback backend error:', upstream.status, detail)
      return NextResponse.json(
        { error: 'Could not save your feedback right now. Please try again.' },
        { status: 502 },
      )
    }

    const data = await upstream.json().catch(() => null)
    return NextResponse.json({ ok: true, count: data?.count ?? null })
  } catch (err) {
    console.log('[v0] Feedback proxy failure:', err)
    return NextResponse.json(
      { error: 'Could not reach the feedback backend. Please try again shortly.' },
      { status: 502 },
    )
  }
}
