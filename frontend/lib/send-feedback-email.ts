import emailjs from '@emailjs/browser'

const SERVICE_ID =
  process.env.NEXT_PUBLIC_EMAILJS_SERVICE_ID?.trim() || 'service_mvmk66y'
const TEMPLATE_ID =
  process.env.NEXT_PUBLIC_EMAILJS_TEMPLATE_ID?.trim() || 'template_jvw0i06'
const PUBLIC_KEY =
  process.env.NEXT_PUBLIC_EMAILJS_PUBLIC_KEY?.trim() || '9hZIx33OKAeoBcUaB'
const TO_EMAIL =
  process.env.NEXT_PUBLIC_EMAILJS_TO_EMAIL?.trim() || 'onethird073@gmail.com'

function stars(n: number) {
  return n > 0 ? `${'★'.repeat(n)}${'☆'.repeat(5 - n)} (${n}/5)` : 'Not rated'
}

export async function sendFeedbackEmail(payload: {
  rating: number
  accuracy: number
  categories: string[]
  comments: string
}) {
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

  await emailjs.send(
    SERVICE_ID,
    TEMPLATE_ID,
    {
      to_email: TO_EMAIL,
      email: TO_EMAIL,
      from_name: 'ONE-THIRD Feedback',
      subject: `ONE-THIRD feedback — rating ${payload.rating || 0}/5`,
      rating: stars(payload.rating),
      accuracy: stars(payload.accuracy),
      categories,
      comments: payload.comments,
      message,
    },
    { publicKey: PUBLIC_KEY },
  )
}
