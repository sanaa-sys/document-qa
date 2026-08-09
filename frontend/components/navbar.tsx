'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Info, MessageSquare, ClipboardList, Brain } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

interface NavItem {
  href: string
  label: string
  icon: React.ComponentType<{ className?: string }>
}

const navItems: NavItem[] = [
  { href: '/about', label: 'About', icon: Info },
  { href: '/chatbot', label: 'Chatbot', icon: MessageSquare },
  { href: '/feedback', label: 'Feedback', icon: ClipboardList },
]

export function Navbar() {
  const pathname = usePathname()

  return (
    <nav className="sticky top-0 z-50 w-full border-b bg-primary/95 backdrop-blur supports-[backdrop-filter]:bg-primary/60">
      <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-4">
        <Link
          href="/about"
          className="flex items-center gap-2 text-lg font-bold text-primary-foreground"
        >
          <Brain className="h-6 w-6" />
          <span>One Third</span>
        </Link>

        <div className="flex items-center gap-2">
          {navItems.map(({ href, label, icon: Icon }) => {
            const isActive = pathname === href
            return (
              <Link key={href} href={href}>
                <Button
                  variant={isActive ? 'default' : 'ghost'}
                  size="sm"
                  className={cn(
                    'gap-2',
                    isActive &&
                      'bg-secondary-foreground text-secondary hover:bg-secondary-foreground/90',
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{label}</span>
                </Button>
              </Link>
            )
          })}
        </div>
      </div>
    </nav>
  )
}
