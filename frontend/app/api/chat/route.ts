import { type NextRequest, NextResponse } from 'next/server'
import { stripThinkTags } from '@/lib/strip-think'

export const dynamic = 'force-dynamic'
export const maxDuration = 60

/**
 * Resolve the FastAPI base URL from common env names and strip
 * accidental quotes/whitespace from the Vercel dashboard.
 */
function getBackendBaseUrl(): string | null {
  const raw = process.env.FASTAPI_URL || ' '

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

  const streamUrl = `${FASTAPI_URL}/chat/stream`
  const jsonUrl = `${FASTAPI_URL}/chat`

  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 90_000)

    const upstream = await fetch(streamUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify({ message }),
      signal: controller.signal,
      cache: 'no-store',
    })

    const contentType = upstream.headers.get('content-type') || ''

    if (upstream.ok && contentType.includes('text/event-stream') && upstream.body) {
      // Keep the abort timer until the stream ends so a hung backend still stops.
      const bodyStream = upstream.body
      const wrapped = new ReadableStream({
        async start(controllerOut) {
          const reader = bodyStream.getReader()
          try {
            while (true) {
              const { done, value } = await reader.read()
              if (done) break
              controllerOut.enqueue(value)
            }
            controllerOut.close()
          } catch (err) {
            controllerOut.error(err)
          } finally {
            clearTimeout(timeout)
          }
        },
        cancel() {
          clearTimeout(timeout)
          controller.abort()
        },
      })

      return new Response(wrapped, {
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream; charset=utf-8',
          'Cache-Control': 'no-cache, no-transform',
          Connection: 'keep-alive',
          'X-Accel-Buffering': 'no',
        },
      })
    }

    clearTimeout(timeout)

    // Older backends without /chat/stream — fall back to JSON /chat.
    if (upstream.status === 404 || upstream.status === 405) {
      return proxyJsonChat(jsonUrl, message)
    }

    if (!upstream.ok) {
      const detail = await upstream.text().catch(() => '')
      console.error('[chat] FastAPI stream error:', upstream.status, detail.slice(0, 500))
      return NextResponse.json(
        {
          error: `The backend responded with an error (${upstream.status}).`,
          detail: detail.slice(0, 300) || undefined,
        },
        { status: 502 },
      )
    }

    return jsonFromUpstream(upstream)
  } catch (err) {
    const aborted = err instanceof Error && err.name === 'AbortError'
    const cause = err instanceof Error ? err.message : String(err)
    console.error('[chat] proxy failure:', cause, 'url=', streamUrl)

    return NextResponse.json(
      {
        error: aborted
          ? 'The backend took too long to respond. Please try again.'
          : 'Could not reach the chatbot backend. Please try again shortly.',
        detail: aborted
          ? undefined
          : `Failed to reach ${streamUrl}. Check FASTAPI_URL on Vercel (${cause}).`,
      },
      { status: aborted ? 504 : 502 },
    )
  }
}

async function proxyJsonChat(chatUrl: string, message: string) {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 90_000)
  try {
    const upstream = await fetch(chatUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
      signal: controller.signal,
      cache: 'no-store',
    })
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
    return jsonFromUpstream(upstream)
  } catch (err) {
    const aborted = err instanceof Error && err.name === 'AbortError'
    const cause = err instanceof Error ? err.message : String(err)
    return NextResponse.json(
      {
        error: aborted
          ? 'The backend took too long to respond. Please try again.'
          : 'Could not reach the chatbot backend. Please try again shortly.',
        detail: aborted ? undefined : cause,
      },
      { status: aborted ? 504 : 502 },
    )
  } finally {
    clearTimeout(timeout)
  }
}

async function jsonFromUpstream(upstream: Response) {
  const data = await upstream.json().catch(() => null)

  const answer = stripThinkTags(
    String(data?.answer ?? data?.response ?? data?.reply ?? data?.message ?? ''),
  )
  const sources = data?.sources ?? data?.citations ?? []
  const confidence = typeof data?.confidence === 'number' ? data.confidence : null

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
}
