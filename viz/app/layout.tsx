import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Momentarily — local dashboard",
  description: "Inspect the Momentarily MTA feed and HMM model.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
