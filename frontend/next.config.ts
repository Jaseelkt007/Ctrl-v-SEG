import type { NextConfig } from "next";

const BACKEND_URL = process.env.BACKEND_URL || "http://129.69.32.93:8000";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "http",
        hostname: "**",
      },
    ],
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
      {
        source: "/ws/:path*",
        destination: `${BACKEND_URL}/ws/:path*`,
      },
    ];
  },
};

export default nextConfig;
