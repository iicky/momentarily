"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Status" },
  { href: "/lines", label: "Lines" },
  { href: "/map", label: "Map" },
  { href: "/trip", label: "Trip" },
  { href: "/commutes", label: "Commutes" },
  { href: "/models", label: "Models" },
];

function isActive(path: string, href: string): boolean {
  if (href === "/") return path === "/";
  return path === href || path.startsWith(`${href}/`);
}

export default function Nav() {
  const path = usePathname();
  return (
    <nav className="nav">
      {LINKS.map((l) => (
        <Link
          key={l.href}
          href={l.href}
          className={isActive(path, l.href) ? "active" : ""}
        >
          {l.label}
        </Link>
      ))}
    </nav>
  );
}
