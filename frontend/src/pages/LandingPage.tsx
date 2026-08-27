import { useRef } from "react";
import { Link } from "react-router-dom";
import Ballpit from "@/components/landing/Ballpit.jsx";
import Crosshair from "@/components/landing/Crosshair.jsx";
import logoFull from "@/assets/logo-full.png";

export function LandingPage() {
  const heroRef = useRef<HTMLDivElement>(null);

  return (
    <div ref={heroRef} className="relative flex min-h-screen flex-col overflow-hidden bg-paper text-ink">
      {/* Custom crosshair cursor, scoped to the hero — it auto-wires an
          enter/leave "glitch" distortion on every <a> inside this container,
          so it fires when the pointer reaches the Enter workspace link. */}
      <Crosshair containerRef={heroRef} color="#000000" />
      {/* Signature element: a physically-simulated field of glossy orange/black
          instances, floating (gravity 0 — no floor to collapse onto) so it
          fills the whole hero instead of piling into a thin band at the
          bottom. Sits behind the copy; content above opts back into pointer
          events per-element so the pit still reacts to cursor movement. */}
      <div className="absolute inset-0 z-0">
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
      </div>

      <div className="pointer-events-none relative z-10 flex min-h-screen flex-col items-center justify-center px-8 py-16 sm:px-16">
        <div className="max-w-2xl border-2 border-ink/10 bg-paper/80 px-8 py-10 text-center sm:px-12 sm:py-14">
          <div className="mb-4 flex items-baseline justify-center gap-3">
            <span className="font-mono text-sm font-bold text-accent">00.</span>
            <h2 className="text-sm font-bold uppercase tracking-widest text-ink">Welcome</h2>
          </div>
          <img
            src={logoFull}
            alt="AutoLabelFlow — Annotate. Verify. Train. Improve."
            className="mx-auto w-full max-w-xl"
          />
          <p className="mt-6 text-lg text-ink/70 sm:text-xl">
            A human-in-the-loop computer-vision annotation platform. Point it at a
            detection model, run it over new footage, correct what it gets wrong,
            and turn every correction into the next model.
          </p>
          <Link
            to="/projects"
            className="pointer-events-auto mt-8 inline-block border-2 border-ink px-6 py-3 text-xs font-bold uppercase tracking-widest transition-colors duration-150 hover:border-orange hover:bg-orange hover:text-paper"
          >
            Enter workspace →
          </Link>
        </div>
      </div>
    </div>
  );
}
