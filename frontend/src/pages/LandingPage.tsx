import { lazy, Suspense } from "react";
import { Link } from "react-router-dom";
import { usePrefersReducedMotion } from "@/lib/usePrefersReducedMotion";
import logoFull from "@/assets/logo-full.png";

// Code-split: Ballpit pulls in three.js + @react-three/fiber (~150KB gzipped)
// plus a render loop, and has no business in the critical path for a page
// whose job is one link. Loads after first paint, and only when motion is
// allowed.
const Ballpit = lazy(() => import("@/components/landing/Ballpit.jsx"));

export function LandingPage() {
  const reducedMotion = usePrefersReducedMotion();

  return (
    <div className="relative flex min-h-[100dvh] flex-col overflow-hidden bg-paper text-ink">
      {/* Signature element: a physically-simulated field of glossy orange/black
          instances, floating (gravity 0) so it fills the whole hero. Sits
          behind the copy. Under prefers-reduced-motion it's dropped entirely
          and the static Swiss grid ground stands in for it. Orange here is
          the same `orange` token as everywhere else in the app (tailwind.config.ts),
          not a one-off amber pulled in just for this component. */}
      {reducedMotion ? (
        <div className="swiss-grid-pattern absolute inset-0 z-0 bg-muted" aria-hidden="true" />
      ) : (
        <div className="absolute inset-0 z-0">
          <Suspense fallback={<div className="swiss-grid-pattern h-full w-full bg-muted" />}>
            <Ballpit
              className="h-full w-full"
              count={110}
              colors={["#FF4500", "#000000", "#FF4500", "#000000", "#FF4500", "#000000", "#FF4500", "#000000"]}
              ambientColor={0xffffff}
              ambientIntensity={0.55}
              lightIntensity={340}
              minSize={0.3}
              maxSize={0.65}
              size0={0.8}
              gravity={0}
              friction={0.99}
              wallBounce={0.75}
              maxVelocity={0.14}
              wander={0.18}
              followCursor={false}
            />
          </Suspense>
        </div>
      )}

      <div className="pointer-events-none relative z-10 flex min-h-[100dvh] flex-col items-center justify-center px-8 py-16 sm:px-16">
        <div className="max-w-2xl border-2 border-ink/10 bg-paper/85 px-8 py-10 text-center sm:px-12 sm:py-14">
          <img
            src={logoFull}
            alt="AutoLabelFlow — Annotate. Verify. Train. Improve."
            className="mx-auto w-full max-w-xl"
            width={1024}
            height={280}
            fetchPriority="high"
            decoding="async"
          />
          <p className="mx-auto mt-6 max-w-[52ch] text-lg text-ink/70 sm:text-xl">
            Point a detection model at new footage, correct what it gets wrong, and turn every fix
            into the next model.
          </p>
          <Link
            to="/projects"
            className="pointer-events-auto mt-8 inline-block border-2 border-ink px-6 py-3 text-xs font-bold uppercase tracking-widest transition-colors duration-150 hover:border-orange hover:bg-orange hover:text-ink"
          >
            Enter workspace →
          </Link>
        </div>
      </div>
    </div>
  );
}
