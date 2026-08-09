import { type NextRequest, NextResponse } from 'next/server'

/**
 * Resolve the FastAPI base URL from common env names and strip
 * accidental quotes/whitespace from the Vercel dashboard.
 */
function getBackendBaseUrl(): string | null {
  const raw = process.env.FASTAPI_URL

  const cleaned = raw.trim().replace(/^['"]|['"]$/g, '').replace(/\/$/, '')
  return cleaned || null
}

export async function POST(req: NextRequest) {
    const FASTAPI_URL = getBackendBaseUrl()

  if (!FASTAPI_URL) {
    return NextResponse.json(
      {
        error:
          'The chatbot is not connected yet. Set FASTAPI_URL in Vercel to your Railway URL (e.g. https://document-qa-production-c8a0.up.railway.app).',
      },
      { status: 503 },
    )
  }

  let body: { message?: string }
  try {
    body = await req.json()
  } catch {
    return NextResponse.json({ error: 'Invalid request body.' }, { status: 400 })
  }

  const message = body.message?.trim()
  if (!message) {
    return NextResponse.json({ error: 'A message is required.' }, { status: 400 })
  }

  const chatUrl = `${FASTAPI_URL}/chat`

  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 60_000)

    const upstream = await fetch(chatUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
      signal: controller.signal,
      // Avoid Next.js caching of POST results
      cache: 'no-store',
    })

    clearTimeout(timeout)

    if (!upstream.ok) {
      const detail = await upstream.text().catch(() => '')
      console.error('[chat] FastAPI error:', upstream.status, detail.slice(0, 500))
      return NextResponse.json(
        {
          error: `The backend responded with an error (${upstream.status}).`,
          detail: detail.slice(0, 300) || undefined,
        },
        { status: 502 },
      )
    }

    const data = await upstream.json().catch(() => null)

    const answer =
      data?.answer ?? data?.response ?? data?.reply ?? data?.message ?? ''
    const sources = data?.sources ?? data?.citations ?? []
    const confidence = typeof data?.confidence === 'number' ? data.confidence : null

    // Surface backend Groq/runtime failures that still return HTTP 200
    if (
      typeof answer === 'string' &&
      (answer.startsWith('Error generating answer:') ||
        answer.startsWith('No Groq API key found.'))
    ) {
      return NextResponse.json(
        { error: answer, answer, sources, confidence },
        { status: 502 },
      )
    }

    return NextResponse.json({ answer, sources, confidence })
  } catch (err) {
    const aborted = err instanceof Error && err.name === 'AbortError'
    const cause = err instanceof Error ? err.message : String(err)
    console.error('[chat] proxy failure:', cause, 'url=', chatUrl)

    return NextResponse.json(
      {
        error: aborted
          ? 'The backend took too long to respond. Please try again.'
          : 'Could not reach the chatbot backend. Please try again shortly.',
        detail: aborted
          ? undefined
          : `Failed to reach ${chatUrl}. Check FASTAPI_URL on Vercel (${cause}).`,
      },
      { status: aborted ? 504 : 502 },
    )
  }
}
