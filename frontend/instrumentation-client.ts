import posthog from 'posthog-js'

const key = process.env.NEXT_PUBLIC_POSTHOG_KEY
const host = process.env.NEXT_PUBLIC_POSTHOG_HOST || 'https://us.i.posthog.com'

if (typeof window !== 'undefined' && key) {
  posthog.init(key, {
    api_host: host,
    defaults: '2026-05-30',
    capture_pageview: true,
    capture_pageleave: true,
    person_profiles: 'identified_only',
  })
}
