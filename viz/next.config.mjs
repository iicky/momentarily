/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Native NAPI module — leave it as a runtime require, don't bundle the .node.
  serverExternalPackages: ["@iicky/murk-secrets"],
};

export default nextConfig;
