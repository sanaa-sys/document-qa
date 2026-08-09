import { type NextRequest, NextResponse } from 'next/server'

// Your FastAPI backend base URL, e.g. https://my-rag-api.onrender.com
// Set this in Project Settings ? Environment Variables.
const FASTAPI_URL = process.env.FASTAPI_URL

export async function POST(req: NextRequest) {
    if (!FASTAPI_URL) {
        return NextResponse.json(
            {
                error:
                    'The chatbot is not connected yet. Add a FASTAPI_URL environment variable pointing to your FastAPI (main.py) backend.',
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

    try {
        const controller = new AbortController()
        const timeout = setTimeout(() => controller.abort(), 60_000)

        // main.py's ChatRequest only expects `{ message }`.
        const upstream = await fetch(`${FASTAPI_URL.replace(/\/$/, '')}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
            signal: controller.signal,
        })

        clearTimeout(timeout)

        if (!upstream.ok) {
            const detail = await upstream.text().catch(() => '')
            console.log('[v0] FastAPI error:', upstream.status, detail)
            return NextResponse.json(
                { error: `The backend responded with an error (${upstream.status}).` },
                { status: 502 },
            )
        }

        const data = await upstream.json().catch(() => null)

        // main.py returns { answer, sources: [{ page, source, score }], confidence }.
        const answer =
            data?.answer ?? data?.response ?? data?.reply ?? data?.message ?? ''
        const sources = data?.sources ?? data?.citations ?? []
        const confidence = typeof data?.confidence === 'number' ? data.confidence : null

        return NextResponse.json({ answer, sources, confidence })
    } catch (err) {
        const aborted = err instanceof Error && err.name === 'AbortError'
        console.log('[v0] Chat proxy failure:', err)
        return NextResponse.json(
            {
                error: aborted
                    ? 'The backend took too long to respond. Please try again.'
                    : 'Could not reach the chatbot backend. Please try again shortly.',
            },
            { status: aborted ? 504 : 502 },
        )
    }
}
