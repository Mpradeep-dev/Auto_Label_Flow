import { Link } from "react-router-dom";
import Antigravity from "@/components/landing/Antigravity.jsx";
import logoMark from "@/assets/logo-mark.png";
import logoFull from "@/assets/logo-full.png";
import { SectionLabel } from "@/components/layout/SectionLabel";

const LOOP = [
  { label: "Auto-annotate", detail: "the registered model runs over new footage" },
  { label: "Correct", detail: "a human reviews and fixes what it got wrong" },
  { label: "Version", detail: "corrected images become a versioned dataset" },
  { label: "Train", detail: "a new model trains on that dataset" },
  { label: "Register", detail: "it replaces the old model as the annotator" },
];

const STACK = [
  { label: "Backend", detail: "FastAPI · SQLAlchemy · Celery · Ultralytics YOLO" },
  { label: "Frontend", detail: "React · TypeScript · Vite" },
  { label: "Storage", detail: "MinIO (S3-compatible), local in dev" },
  { label: "Training", detail: "local GPU or Kaggle" },
];

export function LandingPage() {
  return (
    <div className="min-h-screen bg-paper text-ink">
      <section className="relative overflow-hidden bg-paper text-ink">
        <div className="absolute inset-0 z-0">
          <Antigravity
            count={300}
            magnetRadius={6}
            ringRadius={7}
            waveSpeed={0.4}
            waveAmplitude={1}
            particleSize={1.5}
            lerpSpeed={0.05}
            color="#FF4500"
            autoAnimate
            particleVariance={1}
          />
        </div>

        <div className="pointer-events-none relative z-10 flex min-h-screen flex-col px-8 sm:px-16">
          <header className="flex items-center justify-between border-b-2 border-ink/20 py-6">
            <Link to="/" className="pointer-events-auto flex items-center gap-2">
              <img src={logoMark} alt="" className="h-7 w-7" />
              <span className="text-sm font-bold uppercase tracking-widest">
                Auto <span className="text-orange">Label</span> Flow
              </span>
            </Link>
            <Link
              to="/projects"
              className="pointer-events-auto border-2 border-ink px-4 py-2 text-xs font-bold uppercase tracking-widest transition-colors duration-150 hover:border-orange hover:bg-orange hover:text-paper"
            >
              Enter workspace →
            </Link>
          </header>

          <div className="flex flex-1 flex-col justify-center py-16">
            <div className="mb-4 flex items-baseline gap-3">
              <span className="font-mono text-sm font-bold text-accent">00.</span>
              <h2 className="text-sm font-bold uppercase tracking-widest text-ink">Welcome</h2>
            </div>
            <img src={logoFull} alt="AutoLabelFlow — Annotate. Verify. Train. Improve." className="w-full max-w-2xl" />
            <p className="mt-6 max-w-2xl text-lg text-ink/70 sm:text-xl">
              A human-in-the-loop computer-vision annotation platform. Point it at a
              detection model, run it over new footage, correct what it gets wrong,
              and turn every correction into the next model.
            </p>
          </div>
        </div>
      </section>

      <main className="px-8 py-12 sm:px-16 sm:py-20">
        <div className="grid gap-16 lg:grid-cols-[1.2fr_1fr]">
          <section>
            <SectionLabel index={1}>The loop</SectionLabel>
            <div className="max-w-2xl">
              {LOOP.map((stage, i) => (
                <div key={stage.label} className="flex gap-4">
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center border-2 border-ink text-xs font-bold">
                    {i + 1}
                  </span>
                  <div className="flex-1 border-b-2 border-ink/10 py-6">
                    <p className="text-xl font-bold uppercase tracking-tight sm:text-2xl">{stage.label}</p>
                    <p className="mt-1 text-sm text-ink/60">{stage.detail}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section>
            <SectionLabel index={2}>Built with</SectionLabel>
            <dl className="max-w-md">
              {STACK.map((item) => (
                <div key={item.label} className="border-b-2 border-ink/10 py-4">
                  <dt className="text-xs font-bold uppercase tracking-widest text-ink/50">{item.label}</dt>
                  <dd className="mt-1 text-sm">{item.detail}</dd>
                </div>
              ))}
            </dl>
          </section>
        </div>

        <div className="mt-20 flex flex-wrap items-center justify-between gap-6 border-t-4 border-ink pt-8">
          <p className="text-sm text-ink/60">
            No taxonomy is hardcoded — the platform reads its classes from whatever model you load.
          </p>
          <Link
            to="/projects"
            className="border-2 border-ink px-6 py-3 text-xs font-bold uppercase tracking-widest transition-colors duration-150 hover:border-orange hover:bg-orange hover:text-paper"
          >
            Enter workspace →
          </Link>
        </div>
      </main>
    </div>
  );
}
