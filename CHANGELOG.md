# Changelog

All notable changes to AutoLabelFlow are recorded here. This file is
maintained automatically by [release-please](https://github.com/googleapis/release-please)
from [Conventional Commits](https://www.conventionalcommits.org/) on `main`.

## [0.4.1](https://github.com/Mpradeep-dev/Auto_Label_Flow/compare/v0.4.0...v0.4.1) (2026-09-02)


### Bug Fixes

* stop auto-approving images on Roboflow versioned import ([#13](https://github.com/Mpradeep-dev/Auto_Label_Flow/issues/13)) ([eb39ec5](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/eb39ec56c9ca47a17dd9002a6ff924c63df20149))
* stop auto-approving images on Roboflow versioned import ([#13](https://github.com/Mpradeep-dev/Auto_Label_Flow/issues/13)) ([d1d86f4](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/d1d86f4220c6d74e8655fb1a439866f7c28a0754))

## [0.4.0](https://github.com/Mpradeep-dev/Auto_Label_Flow/compare/v0.3.0...v0.4.0) (2026-09-02)


### Features

* **datasets:** import labelled images from Azure Blob by reference ([6b93e6f](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/6b93e6fb2bb0d8139158cad15c796b4d09536928))
* included sam and annotation tools ([3e0cf76](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/3e0cf76cdf20cc283b27a802ab687a94b5331b08))


### Bug Fixes

* **desktop:** humanize update errors, handle all update/pack button failures ([8a675be](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/8a675be6d7e9d11f81773aaef439ba95e3b9eacc))
* **desktop:** require torch&gt;=2.7 for ultralytics safe-load, guard on mismatch ([486da81](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/486da81aecf7d18c4c667965e0882db5c9518d63))
* **frontend:** bind Vite dev server to 127.0.0.1 explicitly ([abed0d1](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/abed0d12ff483ea7a9578ac5942a705b37ca5ce8))
* **landing:** match Ballpit balls to the app's orange token ([60b6105](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/60b610590a308bd9ac3989d21cb7afc4a517153e))
* **roboflow:** retry transient 5xx/429 from the /search endpoint ([42f3a38](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/42f3a383208c681878609cebb91a843bec4cd747))
* **roboflow:** scope raw import to one upload batch, fix SDK crash and default export batch name ([c94416f](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/c94416f2e2974a80f467533e357f3422c95d6556))

## [0.3.0](https://github.com/Mpradeep-dev/Auto_Label_Flow/compare/v0.2.0...v0.3.0) (2026-09-01)


### Features

* add AutoLabelFlow landing page, branding, and collapsible sidebar nav ([ee85c36](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/ee85c36d09d397456539bf378564cbf8ccfe8075))
* added Roboflow and kaggle ([7cb824c](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/7cb824c340958ba84093daaa827afcf151e672bf))
* backend integrated ([e353d9c](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/e353d9c28c0265e1d169c8ba70d5bf5aa37489bb))
* landing page redesign, app-wide button color pass, live Kaggle training progress ([6096bc0](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/6096bc08a03d4d289fa7c8b721dcca9e8293f986))
* model registry, inference improvements, and UI navigation polish ([6e31ecf](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/6e31ecf24705c752caaca58bc34919a29b9682ee))
* standalone desktop app (Electron) with in-app updater ([4379743](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/437974316851a7327a440fbaa5ed8a32338505e8))
* **storage:** add Azure Blob Storage backend ([4f69f83](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/4f69f83191d6a2ae64f9c3aac724e23b09625c13))


### Bug Fixes

* bound image upload concurrency to avoid connection-pool exhaustion ([736f6d9](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/736f6d9cec353e6ae12620f1018cdfe172ebce50))
* **desktop:** httpx is a runtime dep; bsdtar for python extract; pack node_modules ([671c834](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/671c83401ef945e1673d3a7196ca2af0cf18d0f8))
* imporved UI and integrated MODAL ([28860b4](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/28860b48862565205d15dfe1a6838806e95f0fca))
* **roboflow:** surface real errors from the /search endpoint ([c75a401](https://github.com/Mpradeep-dev/Auto_Label_Flow/commit/c75a4018a9f4bd1f83d5394648721869ae1708a8))

## 0.2.0

Initial tracked release. Earlier history is in the git log.
