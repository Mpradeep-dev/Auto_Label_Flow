import { Link } from "react-router-dom";
import { SectionLabel } from "@/components/layout/SectionLabel";

interface Stage {
  label: string;
  where: string;
  detail: string;
  needs?: string;
}

const STAGES: Stage[] = [
  {
    label: "Import",
    where: "Dataset · Images · Videos",
    detail:
      "Bring in raw material. Upload images or video directly, import a COCO or CVAT-XML .zip, or pull a project in from Roboflow. Video gets frame-extracted into images automatically.",
  },
  {
    label: "Auto-annotate",
    where: "Auto Annotation",
    detail:
      "Run a registered detector model over a dataset. It writes AUTO-source predictions and runs quality rules (e.g. the foot/cone confusion check) that flag suspicious ones for review.",
    needs: "A dataset with images, and a DETECTOR model registered on Models.",
  },
  {
    label: "Review",
    where: "Review Queue → opens each image in Annotate",
    detail:
      "Go through predictions one image at a time — approve, correct a box, or reject. The queue sorts flagged/suspicious images first so the worst predictions get seen before the easy ones.",
    needs: "At least one completed auto-annotation run (or hand-drawn boxes work too).",
  },
  {
    label: "Version",
    where: "Export → \"Create version\"",
    detail:
      "Freezes the currently-approved annotations into a train/val/test split — a snapshot you can export or train against. Nothing downstream can start until one of these exists.",
    needs: "At least one approved image in the dataset.",
  },
  {
    label: "Train",
    where: "Training Runs",
    detail:
      "Fine-tune a new detector on that version — locally on your own GPU, or on Kaggle. Progress (epoch, loss, mAP) streams live while it runs.",
    needs: "A dataset version, and a base DETECTOR model to fine-tune from.",
  },
  {
    label: "New model",
    where: "Models",
    detail:
      "When a training run finishes, the result registers itself here automatically and becomes selectable back in step 2 — that's the loop closing. Its class list is read from its own weights, never typed in.",
  },
];

function StageCard({ index, stage }: { index: number; stage: Stage }) {
  return (
    <div className="flex gap-4 border-b-2 border-ink/10 py-6">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center border-2 border-ink text-xs font-bold">
        {index}
      </span>
      <div>
        <p className="text-xl font-bold uppercase tracking-tight">{stage.label}</p>
        <p className="mt-0.5 text-[10px] font-bold uppercase tracking-widest text-ink/60">
          Sidebar → {stage.where}
        </p>
        <p className="mt-2 max-w-2xl text-sm text-ink/70">{stage.detail}</p>
        {stage.needs && (
          <p className="mt-2 text-xs text-ink/60">
            <span className="font-bold uppercase tracking-widest text-ink/70">Needs: </span>
            {stage.needs}
          </p>
        )}
      </div>
    </div>
  );
}

function BlockedRow({ symptom, fix }: { symptom: string; fix: React.ReactNode }) {
  return (
    <div className="border-b-2 border-ink/10 py-4">
      <p className="text-sm font-bold uppercase tracking-wide">{symptom}</p>
      <p className="mt-1 text-sm text-ink/60">{fix}</p>
    </div>
  );
}

export function HelpPage() {
  return (
    <div className="min-h-full px-8 py-12 sm:px-16 sm:py-20">
      <h1 className="mb-4 border-b-4 border-ink pb-8 text-5xl font-black uppercase tracking-tightest sm:text-7xl">
        Help
      </h1>
      <p className="mb-16 max-w-2xl text-sm text-ink/60">
        This app is one loop, six stages, walked in order. Everything else — Pipeline, the sidebar
        groups, breadcrumbs, search — exists to help you find your place in that loop. This page is
        the map.
      </p>

      <section className="mb-16">
        <SectionLabel index={1}>The loop</SectionLabel>
        <p className="mb-8 max-w-2xl text-sm text-ink/60">
          An existing model auto-annotates new footage → a human corrects it → the corrected data
          becomes a versioned dataset → that trains a new model → the new model gets registered and
          becomes the annotator for the next round. Each project's{" "}
          <span className="font-semibold">Pipeline</span> page (the first thing you see after
          opening a project) tracks exactly where you are in this loop and lights up "Continue"
          buttons for whatever's next — it's live, not just a diagram.
        </p>
        <div className="max-w-3xl">
          {STAGES.map((stage, i) => (
            <StageCard key={stage.label} index={i + 1} stage={stage} />
          ))}
        </div>
      </section>

      <section className="mb-16">
        <SectionLabel index={2}>Finding your way around the sidebar</SectionLabel>
        <div className="grid max-w-3xl gap-6 sm:grid-cols-2">
          <div className="border-2 border-ink p-5">
            <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-ink/60">Workflow</p>
            <p className="text-sm text-ink/70">
              Pipeline, Dataset, Images, Videos — the raw material. Where things go in.
            </p>
          </div>
          <div className="border-2 border-ink p-5">
            <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-ink/60">AI</p>
            <p className="text-sm text-ink/70">
              Auto Annotation, Review Queue, Models, Training Runs — the actual model work: predict,
              correct, retrain.
            </p>
          </div>
          <div className="border-2 border-ink p-5">
            <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-ink/60">Output</p>
            <p className="text-sm text-ink/70">
              Export (versioning) and Project Settings — what leaves the loop, and this project's own
              configuration.
            </p>
          </div>
          <div className="border-2 border-ink p-5">
            <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-ink/60">
              Always visible
            </p>
            <p className="text-sm text-ink/70">
              <span className="font-semibold">Projects</span> (switch projects) and{" "}
              <span className="font-semibold">Settings</span> (Kaggle/Roboflow account connections —
              see below) sit above the project-scoped groups and work with no project selected.
            </p>
          </div>
        </div>
        <p className="mt-6 max-w-2xl text-sm text-ink/60">
          Every project page also shows a <span className="font-semibold">breadcrumb trail</span>{" "}
          just below the top bar (e.g. <span className="tabular">Projects / My Project / Datasets</span>) —
          click any earlier crumb to jump back up. From anywhere, press{" "}
          <kbd className="border border-ink/30 px-1.5 py-0.5 text-xs">⌘K</kbd> /{" "}
          <kbd className="border border-ink/30 px-1.5 py-0.5 text-xs">Ctrl+K</kbd> to jump straight to
          a project, dataset, or model by name.
        </p>
      </section>

      <section className="mb-16">
        <SectionLabel index={3}>Two settings pages — not the same thing</SectionLabel>
        <div className="grid max-w-3xl gap-6 sm:grid-cols-2">
          <div className="border-2 border-ink p-5">
            <p className="mb-2 text-sm font-bold uppercase tracking-wide">Settings (top-level)</p>
            <p className="text-sm text-ink/70">
              Account-wide. Kaggle and Roboflow credentials — connect once, used by every project.
              Reachable with no project open.
            </p>
          </div>
          <div className="border-2 border-ink p-5">
            <p className="mb-2 text-sm font-bold uppercase tracking-wide">Project Settings</p>
            <p className="text-sm text-ink/70">
              Per-project. Name, description, the class taxonomy (read from the registered model's
              weights, not typed in), and which quality-rule packs are on.
            </p>
          </div>
        </div>
      </section>

      <section>
        <SectionLabel index={4}>Why is this stage blocked?</SectionLabel>
        <div className="max-w-2xl">
          <BlockedRow
            symptom="Auto-annotate is greyed out"
            fix="Import at least one image first — Dataset page → create or pick a dataset, then Images page to upload."
          />
          <BlockedRow
            symptom="Review has nothing in it"
            fix='Run auto-annotation on a dataset first, or click "Run quality analysis" on the Review Queue page to flag anything already there.'
          />
          <BlockedRow
            symptom="Version is blocked on Pipeline"
            fix="Approve at least one image in Review — versions are built from approved annotations, not raw predictions."
          />
          <BlockedRow
            symptom="Train is blocked"
            fix={
              <>
                Create a dataset version first — Export page → pick a dataset → "Create version".
              </>
            }
          />
          <BlockedRow
            symptom='Training Runs shows "No CUDA GPU detected" but I have a GPU'
            fix="Almost always a Docker Compose GPU-passthrough issue, not a missing driver — see the GPU training section in the project README."
          />
          <BlockedRow
            symptom="New model never appears on Models"
            fix="The training run needs to reach COMPLETED (check its status on Training Runs) — it registers itself automatically, nothing to click."
          />
        </div>
      </section>

      <p className="mt-16 max-w-2xl text-xs text-ink/60">
        Full technical reference (stack, local dev setup, API layout) lives in the repo's{" "}
        <code className="text-ink/60">README.md</code> and <code className="text-ink/60">docs/WORKFLOW.md</code>.
        Start a walkthrough any time from{" "}
        <Link to="/projects" className="underline hover:text-ink">
          Projects →
        </Link>
      </p>
    </div>
  );
}
