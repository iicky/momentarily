import path from "node:path";
import { fileURLToPath } from "node:url";

// viz/ is a sub-package with its own lockfile; the shared model-core sources it
// imports live one level up in ../shared. Point Turbopack's root at the repo
// root so those cross-package `../../shared/*.ts` imports resolve during build.
const repoRoot = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Native NAPI module — leave it as a runtime require, don't bundle the .node.
  serverExternalPackages: ["@interrupted/murk-secrets"],
  turbopack: { root: repoRoot },
};

export default nextConfig;
