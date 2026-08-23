import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Momentarily",
  description: "Live NYC MTA service status + HMM inference.",
  icons: {
    icon: [
      { url: "/brand/favicon.svg", type: "image/svg+xml" },
      { url: "/brand/icon-192.png", sizes: "192x192", type: "image/png" },
      { url: "/brand/icon-512.png", sizes: "512x512", type: "image/png" },
    ],
    apple: [{ url: "/brand/apple-icon.png", sizes: "180x180" }],
  },
  openGraph: {
    title: "Momentarily",
    description: "Live NYC MTA service status + HMM inference.",
    images: [{ url: "/brand/og.png", width: 1200, height: 630 }],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      {/* Browser extensions (ColorZilla, Grammarly, etc.) mutate <body>
          attributes before hydration; suppress that one-level diff. */}
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
