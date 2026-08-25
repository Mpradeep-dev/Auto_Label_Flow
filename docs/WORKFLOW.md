# Workflow guide

This is the "how do I actually use this thing" doc. For stack/setup, see the
main [`README.md`](../README.md). For the visual design language, see
[`frontend/DESIGN.md`](../frontend/DESIGN.md). The same content, written for
in-app reading, lives at `/help` once the app is running.

## The loop, in one sentence

> An existing model auto-annotates new footage → a human corrects it → the
> corrected data becomes a versioned dataset → that trains a new model → the
> new model gets registered and becomes the annotator for the next round.

Everything in the UI is scaffolding around that one loop. A project's
**Pipeline** page (the first thing you see after opening a project) tracks
where you are in it live — it's a dashboard driven by real data, not a static
diagram, so its six stages actually turn on/off based on what you've done.

## The six stages

| # | Stage | Sidebar location | What it does | Blocked until |
|---|---|---|---|---|
| 1 | **Import** | Dataset · Images · Videos | Upload images/video directly, import a COCO/CVAT-XML `.zip`, or pull in a Roboflow project. Video gets frame-extracted automatically. | — (always available) |
| 2 | **Auto-annotate** | Auto Annotation | Runs a registered detector over a dataset, writing `AUTO`-source predictions and flagging suspicious ones via the quality-rule layer. | A dataset has images, and a `DETECTOR` model is registered |
| 3 | **Review** | Review Queue → opens each image in Annotate | Approve, correct, or reject predictions. The queue sorts flagged/difficult images first. | At least one auto-annotation run (or manual boxes) exists |
| 4 | **Version** | Export → "Create version" | Freezes currently-approved annotations into a train/val/test split — the unit everything downstream (export, training) operates on. | At least one approved image |
| 5 | **Train** | Training Runs | Fine-tunes a new detector on a version, locally (your GPU) or on Kaggle. Progress streams live (epoch, loss, mAP). | A dataset version and a base detector model exist |
| 6 | **New model** | Models | Registers itself automatically when a training run reaches `COMPLETED`. Its class list is read from its own weights — never typed in. Becomes selectable back at stage 2, closing the loop. | A training run completes |

## Sidebar map

The sidebar is grouped by where each page sits in the loop above, not
alphabetically:

- **Workflow** — Pipeline, Dataset, Images, Videos. Raw material; where
  things go *in*.
- **AI** — Auto Annotation, Review Queue, Models, Training Runs. The actual
  model work: predict, correct, retrain.
- **Output** — Export (versioning), Project Settings. What leaves the loop,
  and this project's own configuration.
- **Always visible, no project needed** — *Projects* (switch projects),
  *Settings* (account-wide Kaggle/Roboflow connections — see below), *Help*
  (this guide, in-app).

Two more navigation aids on every project page:

- **Breadcrumbs**, just under the top bar (`Projects / My Project /
  Datasets / …`) — click any earlier crumb to jump back up a deeply nested
  route.
- **⌘K / Ctrl+K search** — jump straight to a project, dataset, or model by
  name from anywhere in the app.

## Two settings pages — not the same thing

| | Settings (top-level) | Project Settings |
|---|---|---|
| Scope | Account-wide | This project only |
| Reachable | With no project open | Inside a project, under Output |
| Holds | Kaggle / Roboflow credentials | Name, description, class taxonomy (read from the model, not typed), quality-rule packs |

## Why is this stage blocked?

| Symptom | Fix |
|---|---|
| Auto-annotate greyed out on Pipeline | Import at least one image — Dataset → create/pick a dataset → Images → upload |
| Review Queue is empty | Run auto-annotation on a dataset, or click "Run quality analysis" on the Review Queue page |
| Version blocked on Pipeline | Approve at least one image in Review — versions are built from approved annotations, not raw predictions |
| Train blocked | Create a dataset version first — Export → pick a dataset → "Create version" |
| Training Runs shows "No CUDA GPU detected" despite having a GPU | Almost always Docker Compose not passing the GPU through to the `backend`/`worker` containers — see [GPU training in Docker](../README.md#gpu-training-in-docker) in the README. `nvidia-smi` working on the host is not sufficient; the container needs its own `deploy.resources.reservations.devices` block. |
| New model never appears on Models | The training run needs to reach `COMPLETED` (check status on Training Runs) — registration happens automatically |

## Where each stage's data actually lives

For anyone reading code alongside this doc — the loop above maps to the
backend services roughly 1:1:

```
Import          → backend/app/services/dataset/, backend/app/services/storage/
Auto-annotate    → backend/app/services/inference/, workers (inference task)
Review           → backend/app/services/review/ (difficulty scoring),
                    backend/app/services/quality/ (flag packs)
Version          → backend/app/services/dataset/ (versioning, split, YOLO export)
Train            → backend/app/services/training/ (LOCAL, KAGGLE providers)
New model        → backend/app/api/v1/models.py (registration reads class_config from weights)
```
