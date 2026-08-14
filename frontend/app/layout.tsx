import { Analytics } from '@vercel/analytics/next'
import type { Metadata, Viewport } from 'next'
import { Nunito, Poppins } from 'next/font/google'
import { Navbar } from '@/components/navbar'
import { SiteFooter } from '@/components/site-footer'
import './globals.css'

const nunito = Nunito({
    subsets: ['latin'],
    variable: '--font-nunito',
    display: 'swap',
})

const poppins = Poppins({
    subsets: ['latin'],
    weight: ['500', '600', '700'],
    variable: '--font-poppins',
    display: 'swap',
})

export const metadata: Metadata = {
    title: 'ONE-THIRD — Your caring medical assistant',
    description:
        'ONE-THIRD is a warm, approachable medical RAG chatbot that answers your health questions with grounded, trustworthy information.',
    icons: {
        icon: [{ url: '/favicon.png', type: 'image/png' }],
        apple: '/favicon.png',
    },
}

export const viewport: Viewport = {
    colorScheme: 'light dark',
    themeColor: [
        { media: '(prefers-color-scheme: light)', color: '#f3fbfd' },
        { media: '(prefers-color-scheme: dark)', color: '#06181d' },
    ],
}

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode
}>) {
    return (
        <html lang="en" className={`bg-background ${nunito.variable} ${poppins.variable}`}>
            <body className="font-sans antialiased">
                <div className="flex min-h-dvh flex-col">
                    <Navbar />
                    <div className="flex-1">{children}</div>
                    <SiteFooter />
                </div>
                {process.env.NODE_ENV === 'production' && <Analytics />}
            </body>
        </html>
    )
}