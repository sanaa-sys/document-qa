import type { NextConfig } from 'next';
import path from 'path';

const nextConfig: NextConfig = {
    images: {
        unoptimized: true,
    },
    webpack: (config, { isServer }) => {
        // Add @ alias for paths
        config.resolve.alias['@'] = path.join(__dirname);
        return config;
    },
};

export default nextConfig;
