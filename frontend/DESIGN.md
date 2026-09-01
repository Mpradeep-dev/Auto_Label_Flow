# Design system — Swiss International

Full rationale lives in the project plan; this is the working reference for
anyone writing a component. Tokens are defined once in `tailwind.config.ts` —
never re-declare a hex value or a border width in a component file.

## Palette

| Token | Value | Use |
|---|---|---|
| `paper` | `#FFFFFF` | Every page/panel/table background |
| `ink` | `#000000` | Text, borders, primary fills |
| `muted` | `#F2F2F2` | Secondary surfaces, editorial texture backgrounds |
| `accent` | `#FF3000` | **Reserved exclusively** for "needs attention" — suspicious-flag borders, alerts. Never a class colour, never decorative, never a hover affordance. |
| `accent.ink` | `#C42200` | The same red darkened to clear WCAG AA on white (~5.9:1). Use for accent-coloured **text** at `text-sm` and below — error/validation copy, small alert labels (`text-accent-ink`). Not for fills or borders. |
| `orange` | `#FFB000` | Amber — the interactive-affordance colour: hover fills (`hover:bg-orange`), hover borders (`hover:border-orange`), and the "pending / in-progress" state (a drawn-but-unclassified shape). Deliberately far from `accent` red so a hovered control never reads as an alert. Pair fills with `text-ink` — black on amber ≈ 11:1; white on amber fails contrast. |
| `plate` | `#000000` | The image-viewport ground (see "The one exception" below) |

`accent` at full strength on white is ~3.7:1 — fine as a fill/border/icon
colour paired with black or white text, or at large/bold sizes, but **not**
as small red-on-white text. For that, reach for `accent.ink`.

Grey text: use `text-ink/60` as the lightest step for anything a user is
meant to read (~4.8:1). `text-ink/50` and below is for disabled or purely
decorative text only.

## The one exception to the white ground

Every surface is `paper`/`ink`/`accent` except the annotation image viewport,
which is `plate` (pure black) framed by a 4px `ink` border — a photograph
mounted on a white page, the way Swiss print treats photography. This is
deliberate and singular: don't introduce a second exception elsewhere.

## Structure

- **Radius: 0px, always.** No rounded corners anywhere.
- **Borders are thick and visible**: `border-2` (2px) for internal structure,
  `border-4` (4px) for major section boundaries (sidebar, image plate).
- **Typography**: Inter, loaded via Google Fonts in `index.html`. Headings and
  labels are uppercase. Editorial headings use `font-black` (900) at
  `tracking-tightest`; instrument labels use `font-semibold`/`font-bold` at
  `tracking-widest`.

## Two density registers, one token set

- **Editorial** (pipeline home, projects, datasets, dashboards, training,
  export): massive responsive numerals as content, `NN. SECTION` numbered
  labels in `accent` (see `SectionLabel`), generous `p-12`–`p-20` padding,
  `.swiss-grid-pattern`/`.swiss-dots` texture on `muted` surfaces only —
  never on `plate` or `accent` fills.
- **Instrument** (the annotation workspace, built in Phase 3): compact rows,
  `text-xs`/`text-sm` uppercase labels, `.tabular` (tabular-nums) on every
  numeric field so coordinate columns align, minimal vertical rhythm — the
  image stays the largest thing on screen.

Same colours, borders, and radius in both registers. Only the scale changes.

## Box colour language (Phase 3)

Classes are told apart by a small flat-hue palette assigned by class *index*,
defined in `src/config/classColors.ts` once that file exists — never by class
*name*, since names come from whatever model is loaded. `accent` (red) is
reserved for a suspicious-flag border, applied on top of a box regardless of
its class colour. The two channels — identity, attention — never collide.

## Motion

`150ms ease-out` default (see `tailwind.config.ts` `transitionDuration`/
`transitionTimingFunction` defaults). Colour inversions, not fades. No
spring/elastic easing. Kept minimal in the instrument register — a box drag
must never feel like it's animating.

`prefers-reduced-motion: reduce` is enforced globally in `src/index.css` — it
collapses every CSS transition/animation to instant. JS-driven motion that CSS
can't reach (the landing-page Ballpit) is gated in-component via
`usePrefersReducedMotion()` and must fall back to a static equivalent, never
just freeze. The landing page is the only place with motion beyond colour
inversions; it has no custom cursor (removed — accessibility/perf-hostile).

## Focus

Every interactive element gets one visible focus ring: a 2px `ink` outline at
`outline-offset: 2px`, applied via `:focus-visible` in `src/index.css` (base
layer). Components may add `focus:border-accent` on inputs as extra signal,
but must not `outline-none` without it. Keyboard/AT users only — not on mouse
click.
