import { type NextRequest, NextResponse } from 'next/server'

const EMAILJS_SERVICE_ID =
  process.env.EMAILJS_SERVICE_ID?.trim() || 'service_mvmk66y'
const EMAILJS_TEMPLATE_ID =
  process.env.EMAILJS_TEMPLATE_ID?.trim() || 'template_jvw0i06'
const EMAILJS_PUBLIC_KEY =
  process.env.EMAILJS_PUBLIC_KEY?.trim() || '9hZIx33OKAeoBcUaB'
const EMAILJS_PRIVATE_KEY = process.env.EMAILJS_PRIVATE_KEY?.trim() || ''

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

async function sendFeedbackEmail(payload: {
  rating: number
  accuracy: number
  categories: string[]
  comments: string
}) {
  const stars = (n: number) =>
    n > 0 ? `${'★'.repeat(n)}${'☆'.repeat(5 - n)} (${n}/5)` : 'Not rated'
  const categories =
    payload.categories.length > 0 ? payload.categories.join(', ') : 'None selected'

  const message = [
    'New feedback from ONE-THIRD',
    '',
    `Overall rating: ${stars(payload.rating)}`,
    `Accuracy: ${stars(payload.accuracy)}`,
    `Categories: ${categories}`,
    '',
    'Comments:',
    payload.comments,
  ].join('\n')

  const templateParams = {
    from_name: 'ONE-THIRD Feedback',
    subject: `ONE-THIRD feedback — rating ${payload.rating || 0}/5`,
    rating: stars(payload.rating),
    accuracy: stars(payload.accuracy),
    categories,
    comments: payload.comments,
    message,
  }

  const body: Record<string, unknown> = {
    service_id: EMAILJS_SERVICE_ID,
    template_id: EMAILJS_TEMPLATE_ID,
    user_id: EMAILJS_PUBLIC_KEY,
    template_params: templateParams,
  }
  if (EMAILJS_PRIVATE_KEY) {
    body.accessToken = EMAILJS_PRIVATE_KEY
  }

  const res = await fetch('https://api.emailjs.com/api/v1.0/email/send', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    console.error('[feedback] EmailJS error:', res.status, detail.slice(0, 500))
    const lowered = detail.toLowerCase()
    const needsNonBrowserAccess =
      lowered.includes('non-browser') ||
      lowered.includes('private key') ||
      lowered.includes('access token')
    return {
      ok: false as const,
      error: needsNonBrowserAccess
        ? 'EmailJS blocked this server request. In EmailJS go to Account → Security and enable “Allow EmailJS API for non-browser applications” (or add EMAILJS_PRIVATE_KEY).'
        : 'Could not send the feedback email right now. Please try again.',
    }
  }

  return { ok: true as const }
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

  const emailResult = await sendFeedbackEmail(payload)
  if (!emailResult.ok) {
    return NextResponse.json({ error: emailResult.error }, { status: 503 })
  }

  // Best-effort: also store on the FastAPI backend when configured.
  const FASTAPI_URL = getBackendBaseUrl()
  if (FASTAPI_URL) {
    try {
      await fetch(`${FASTAPI_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
    } catch (err) {
      console.error('[feedback] backend forward failed (email already sent):', err)
    }
  }

  return NextResponse.json({ ok: true })
}

