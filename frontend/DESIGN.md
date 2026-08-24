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
| `accent` | `#FF3000` | **Reserved exclusively** for "needs attention" — suspicious-flag borders, alerts. Never a class colour, never decorative. |
| `plate` | `#000000` | The image-viewport ground (see "The one exception" below) |

`accent` on white fails WCAG AA at small sizes (~4:1, needs 4.5:1). Use it as
a fill/border/icon colour paired with black or white text, or at large/bold
sizes — never as small red-on-white body text.

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
spring/elastic easing. Respect `prefers-reduced-motion`. Kept minimal in the
instrument register — a box drag must never feel like it's animating.
