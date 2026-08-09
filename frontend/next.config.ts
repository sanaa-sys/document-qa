import type { NextConfig } from 'next';
import path from 'path';

const nextConfig: NextConfig = {
    images: {
        unoptimized: true,
    },
    // Disable Turbopack, force webpack
    experimental: {
        webpackBuildWorker: true,
    },
    // Keep webpack config for path aliases
    webpack: (config, { isServer }) => {
        config.resolve.alias['@'] = path.join(__dirname);
        return config;
    },
    async rewrites() {
        const backendUrl = process.env.FAST_API_URL || 'http://localhost:8000';
        return [
            {
                source: '/api/:path*',
                destination: `${backendUrl}/:path*`,
            },
        ];
    },
};

export default nextConfig;
