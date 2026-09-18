# Momentarily brand guide

The single source of truth for Momentarily's mark, palette, and voice. Every
asset in this directory is derived from the rules below; change the rules here
before touching any SVG or PNG.

## The mark

Two rounded rectangles seen from above as train cars — a 24-unit grid, w=5
h=18 rx=2.5. The gap and bar height together carry the route state:

- **gap** (analog, nominal 4): encodes delay magnitude.  
  `gap = 4 + 7 × (1 − exp(−r/12))` where r is predicted recovery minutes.  
  Sensitive across 0–15 min, compressed in the tail. Saturates at 11 so the
  mark cannot break its 24-unit frame.

- **height** (binary, 18 or 10): encodes service existence.  
  Full height = trains running. Half height = no service (suspended).  
  This is the channel that must survive 20 px monochrome.

Bar width is fixed at 5. A gap change reads as a total footprint change —
far more legible in an aligned column than an internal gap alone.

Nominal gap 4 is a **bidirectional neutral**, not a floor: bunching (trains
stacked) reads as gap approaching 0; gapping (trains far apart) as gap rising.
The cars never touch so gap 0 stays reserved for the stacked-trains reading.

At nominal (gap 4, h=18) the silhouette is the pause glyph (0.80:1 aspect).
By ten minutes of delay it is past 1.59:1 and no longer reads as pause. That
distinction is intentional: the mark starts as a pause and delay deforms it
into something uniquely itself.

The mark stays two cars. N-car strips are derived by repetition, never by
changing bar count in this base asset.

**Rejected encodings** (verified in mark-study.html):
- gap-width alone as a 3-state encoding: fails the 20 px monochrome test; colour does all the work.
- a tick inside the gap: vanishes by 32 px.
- unequal bar widths: reads as a text cursor beside a block.
- the rotated (dimensionally truer) version: collapses into an equals sign by 32 px.
- circles and diamonds: MTA route bullets are circles (local) and diamonds (express), and murk already owns the diamond.

## Asset variants

Three base variants, chosen by context:

| File | Fill | Use |
|---|---|---|
| `assets/mark.svg` | `currentColor` | Inline SVG only — inherits colour from CSS |
| `assets/mark-mono.svg` | `#c8ccd4` (bone) | `<img>` or `background-image` on dark surfaces |
| `assets/mark-ink.svg` | `#16181d` (ground) | `<img>` or `background-image` on light surfaces |

`currentColor` resolves only when the SVG is **inlined**. Loaded via `<img>` or
`background-image` the SVG is an isolated document and `currentColor` falls
back to black. Use the named variant for those contexts.

State mark assets encode the fixed per-state geometry and colour:

| File | Gap | Height | Fill |
|---|---|---|---|
| `assets/mark-normal.svg` | 4 (nominal) | 18 | `#98c379` |
| `assets/mark-delayed.svg` | ~7.96 (~10 min) | 18 | `#e5c07b` |
| `assets/mark-suspended.svg` | 11 (saturated) | 10 | `#e06c75` |

## State palette

One Dark semantic three. State is always **shape AND hue** — never hue alone.

| State | Role | Hex |
|---|---|---|
| Normal | Full-height mark, gap 4 | `#98c379` |
| Delayed / disrupted | Full-height mark, wider gap | `#e5c07b` |
| Suspended | Half-height mark, saturated gap | `#e06c75` |
| Neutral / bone | Mark on dark surfaces | `#c8ccd4` |
| Ground / ink | Dark surfaces, mark fill on light | `#16181d` |

The quiet-normal state (service running but below judgment threshold) uses the
muted tone (`#949aa6`), not the normal green. A page full of quiet marks
overnight must not read as an all-clear.

## Accent hue

Momentarily's per-product accent is **`#00BCA8`** — a fixed teal-cyan.

This is distinct from the state palette above, which belongs to route status.
The accent is for UI chrome: links, focus rings, hover states, interactive
affordances.

**Why teal:** The runtime route colours are fixed in `worker/src/derive.ts`
`SUBWAY_ROUTE_META` and supplied to the dashboard at runtime via GTFS static.
Every accent candidate must be visually distinct from all of them, and from
murk's violet (`#8760FF`).

Runtime route → hex (source of truth; do not diverge from `derive.ts`):

| Routes | Hex |
|---|---|
| 1, 2, 3 | `#EE352E` |
| 4, 5, 6 | `#00933C` |
| 7 | `#B933AD` |
| A, C, E | `#2850AD` |
| B, D, F, M | `#FF6319` |
| G | `#6CBE45` |
| J, Z | `#996633` |
| L | `#A7A9AC` |
| N, Q, R, W | `#FCCC0A` |
| GS, FS, H (S shuttles) | `#808183` |
| SI | `#1F4F9F` |
| murk (org product) | `#8760FF` |

The occupied hues are: red, orange, yellow, green, lime, brown, blue, navy,
magenta-purple, and two greys. Teal-cyan is the only remaining band with
sufficient distinctness from all twelve entries above.

**Contrast:**

| Background | Ratio |
|---|---|
| Ground `#16181d` | 7.41:1 |

Clears both the 4.5:1 bar (small text) and the 3:1 bar (graphics and marks).

**Rule note:** The warm-vs-cool product rule on interrupted.sh is a
misstatement of the actual invariant. What separates org chrome from product
identity is *rerolled* (AccentReroll picks a random accent per page load)
versus *fixed* (product themes are scoped and stable). A fixed teal is correct
product behaviour; the warmth of the hue is irrelevant to that distinction.

## Contrast summary

Key pairs, all measured against ground `#16181d`:

| Color | Role | Ratio |
|---|---|---|
| `#c8ccd4` text / bone | Body text, neutral mark | 11.03:1 |
| `#98c379` normal green | State mark, badge | 8.81:1 |
| `#e5c07b` disrupted amber | State mark, badge | 10.28:1 |
| `#e06c75` suspended red | State mark, badge | 5.56:1 |
| `#00bca8` accent teal | Links, focus, interactive | 7.41:1 |

All clear 4.5:1. Use state colours as marks and accent only — never for small
body text on a surface darker than the ground.

## Icons and favicons

| File | Use |
|---|---|
| `assets/favicon.svg` | Browser tab, scalable (rounded ground tile + bone mark) |
| `assets/favicon-16.png` | Legacy 16×16 |
| `assets/favicon-32.png` | Legacy 32×32 |
| `assets/icon-192.png` | Web app manifest, 192×192 |
| `assets/icon-512.png` | Web app manifest, 512×512 |
| `assets/icon-512-maskable.png` | Maskable manifest icon |
| `assets/apple-icon.png` | Apple touch icon, 180×180 |
| `assets/og.png` | Open Graph card, 1200×630 |

The favicon tile uses `rx=22` matching the interrupted.sh icon convention.
The Apple touch icon is full-bleed (no rounding — iOS applies its own mask).
The maskable icon places the mark inside the central 80 % safe zone.

## Typography

Inter (400) for UI text. Meslo (400, 700) for monospaced / numeric output.
Both self-hosted in `assets/fonts/` with the font licences alongside.

## Voice

Momentarily is direct and un-dramatic. Status is a fact, not a crisis.

Tagline source: "We are being held momentarily by the train's dispatcher."
That sentence is the product in one line.

**Plain voice rules:**
- State a fact: "Trains are moving normally." Not "Service is excellent."
- A disruption is a disruption, not a "situation". Name the state.
- Recovery estimates are approximate; say so with a tilde or "~".
- The feed infers from movement; say "inferred from train movement", not
  "AI-powered" or "machine-learning-based".

## Positioning copy

Reusable copy, plain voice. Pull from these for titles, descriptions, and
launch materials so the claim stays consistent.

**Tagline:** Live NYC subway status, inferred from how trains are actually
moving.

**One-liner:** Real-time subway state from movement, not from the MTA's own
reports.

**Two-sentence pitch:** Momentarily watches where trains actually are and
infers whether a line is running normally, slowing, or suspended — without
waiting for an official alert to arrive. The state mark changes shape as
delays compound, so you read the condition from the silhouette, not just the
colour.

**Honest caveat:** The inference is probabilistic. The model reads movement
patterns; it does not have access to dispatcher instructions or track
conditions directly.
