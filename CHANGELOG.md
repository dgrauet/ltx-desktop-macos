# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1](https://github.com/dgrauet/ltx-desktop-macos/compare/v0.2.0...v0.2.1) (2026-06-30)


### Bug Fixes

* **ci:** name the DMG from the release tag, drop silent 0.1.0 fallback ([69eabf1](https://github.com/dgrauet/ltx-desktop-macos/commit/69eabf1c54e9ddfd6c642eb5a5329fc4b212c52c))

## [0.2.0](https://github.com/dgrauet/ltx-desktop-macos/compare/v0.1.0...v0.2.0) (2026-06-30)


### Features

* **api:** training dataset endpoints (J4b P1) ([b1ff2fc](https://github.com/dgrauet/ltx-desktop-macos/commit/b1ff2fcb5a8676fc451acb05ed61419a41107984))
* **backend:** task-6 run-supervisor integration, preflight/run endpoints, generation guard ([fee079e](https://github.com/dgrauet/ltx-desktop-macos/commit/fee079e1fe771e6500fa9ff194a9122b17560fc0))
* **training:** 32GB-safe T2V config builder bound to real signature (J4b P0) ([5ed227e](https://github.com/dgrauet/ltx-desktop-macos/commit/5ed227e643abc4c68a6a05a81cd008626575506c))
* **training:** add ltx-trainer-mlx dependency (J4b P0) ([5dc5b91](https://github.com/dgrauet/ltx-desktop-macos/commit/5dc5b9197954e20bdbaf9234b00bb477f90c6f4a))
* **training:** config_builder low_ram flag (default normal training) (J4b P1) ([1f31193](https://github.com/dgrauet/ltx-desktop-macos/commit/1f3119398756619ab0566537df7d2eae6462d67f))
* **training:** dataset dirs + manifest→captions materialization (J4b P1) ([f82087f](https://github.com/dgrauet/ltx-desktop-macos/commit/f82087ff657b9281732d7911f72f6c1e348d240d))
* **training:** dataset_store with media + adequacy validation (J4b P0) ([b168244](https://github.com/dgrauet/ltx-desktop-macos/commit/b168244c109ce47b9851b74a1bbaac6d9ef09562))
* **training:** generation↔training exclusion lock (J4b P1) ([adb6bd9](https://github.com/dgrauet/ltx-desktop-macos/commit/adb6bd96225315c260995c882d1c31d273af3afb))
* **training:** preprocess + train subprocess runners with preflight (J4b P0) ([fc93b61](https://github.com/dgrauet/ltx-desktop-macos/commit/fc93b61eb119442cd4c6a84e66f45f611fef5fe2))
* **training:** stderr progress protocol + preflight verdict (J4b P0) ([f638cae](https://github.com/dgrauet/ltx-desktop-macos/commit/f638cae0570da46281314740d29514d1633163e7))
* **training:** training_store run persistence (J4b P1) ([018b139](https://github.com/dgrauet/ltx-desktop-macos/commit/018b139c888683391e7246df847019499ae46672))
* **ui:** auto-detect sidecar .txt captions when adding clips (J4b P1) ([9fca0a7](https://github.com/dgrauet/ltx-desktop-macos/commit/9fca0a7b2f9339c951842d949c550dc36cc26c55))
* **ui:** BackendService training API + WS decode (J4b P1) ([e6e5f0b](https://github.com/dgrauet/ltx-desktop-macos/commit/e6e5f0bc2749068765f491f177e178b279e97f4b))
* **ui:** dataset builder view (J4b P1) ([b590dfb](https://github.com/dgrauet/ltx-desktop-macos/commit/b590dfbe6f0956412dcc96025f28c597c0f44783))
* **ui:** float word-count + Enhance inside the prompt field ([ef346d4](https://github.com/dgrauet/ltx-desktop-macos/commit/ef346d4f216efd2c0f49a12af6e7376b56714272))
* **ui:** mark Training as beta (sidebar badge + banner) ([1b172fc](https://github.com/dgrauet/ltx-desktop-macos/commit/1b172fc2adf4c576c389937b489dac068d26ff31))
* **ui:** place image and audio drop zones side by side ([ea15de7](https://github.com/dgrauet/ltx-desktop-macos/commit/ea15de734ae034f1f3e445682ddc89c007544219))
* **ui:** training Codable models (J4b P1) ([677e0a1](https://github.com/dgrauet/ltx-desktop-macos/commit/677e0a114aa47b72cbd5e8773fd6f405eff502d4))
* **ui:** training config view + presets (J4b P1) ([168039d](https://github.com/dgrauet/ltx-desktop-macos/commit/168039dac70bcc2c4538b6c8790f1c12d627a09a))
* **ui:** training run view + Training tab (J4b P1) ([d15398a](https://github.com/dgrauet/ltx-desktop-macos/commit/d15398a4f9d2bc7be24e095f89f8c98bd78ebfec))
* **ui:** TrainingViewModel (J4b P1) ([bc1be6a](https://github.com/dgrauet/ltx-desktop-macos/commit/bc1be6a1db016cb3d41bb42a91d4380c207da8ec))


### Bug Fixes

* **api:** correct preflight/run data-root (.precomputed) and text-encoder (Gemma) (J4b P1) ([05c6476](https://github.com/dgrauet/ltx-desktop-macos/commit/05c647606d1a9038676f93f3af71f3b584b812ac))
* **api:** prevent path traversal in training dataset endpoints (J4b P1) ([8a86839](https://github.com/dgrauet/ltx-desktop-macos/commit/8a86839dfd1f85d0db06e589556e6b9eec2fbfb7))
* **api:** real training cancel + forward lr/seed + surface stderr (J4b P1) ([64ec086](https://github.com/dgrauet/ltx-desktop-macos/commit/64ec0864a1aa3dabf327b1902dd85b7f0517e14b))
* **api:** reject path traversal in training dataset endpoints (J4b P1) ([6e5e906](https://github.com/dgrauet/ltx-desktop-macos/commit/6e5e906babd570f6f04863444b875468b9ca51aa))
* **api:** training endpoint response shapes match Swift decoders (J4b P1) ([b3541cc](https://github.com/dgrauet/ltx-desktop-macos/commit/b3541cc039f28437689530389c113efdd2f2e763))
* **backend:** task-6 review — real cancel, lr/seed passthrough, preprocess stderr capture ([f335c93](https://github.com/dgrauet/ltx-desktop-macos/commit/f335c93e982c70a9b3331b21336e80e29b784f07))
* **dataset_store:** reject missing/corrupt clips explicitly in validate_clip ([a45f7b7](https://github.com/dgrauet/ltx-desktop-macos/commit/a45f7b764b7c5c1073aa86ee504e084987b15bbc))
* **lora:** per-generation selection is authoritative; UI stops overwriting it (J4b P1) ([bb17877](https://github.com/dgrauet/ltx-desktop-macos/commit/bb178776037f37a6b5296bd0c92a273d791d5995))
* **training:** bump ltx-2-mlx 0.14.11-&gt;0.14.13 (J4b P0) ([5255b23](https://github.com/dgrauet/ltx-desktop-macos/commit/5255b23404996e71bcbabdcec04cab3d828b473e))
* **training:** disable validation by default, cap MLX cache, add --validate ([7e5efbd](https://github.com/dgrauet/ltx-desktop-macos/commit/7e5efbdea02dfbd77ce7d70de5baba5e945e9940))
* **training:** error-wrap preprocess + clarify stderr protocol docstrings ([7d38b2a](https://github.com/dgrauet/ltx-desktop-macos/commit/7d38b2aaa8bede422d8dbaa6cf96c931514a4b86))
* **training:** optional done loraPath, surface STATUS lines, correct docs (J4b P1 final review) ([139c200](https://github.com/dgrauet/ltx-desktop-macos/commit/139c200e5f7502cf630e0bebd89bf18a18f00760))
* **ui:** long timeout for preflight (preprocess+probe takes minutes) (J4b P1) ([12308ff](https://github.com/dgrauet/ltx-desktop-macos/commit/12308ffe2142a0c8620a5f8b8a2d0d35536393ee))
* **ui:** serialize dataset drop URL accumulation + honest drop types (J4b P1) ([744a6d8](https://github.com/dgrauet/ltx-desktop-macos/commit/744a6d875afa2216c90814d8f3096340e4547960))
* **ui:** use macOS 14-compatible SF Symbol for running status (J4b P1) ([d38ded6](https://github.com/dgrauet/ltx-desktop-macos/commit/d38ded6365272c0c02d776334a59857bf114e14b))


### Miscellaneous Chores

* release 0.2.0 ([b7f5070](https://github.com/dgrauet/ltx-desktop-macos/commit/b7f5070ee1dae51f43451bc07541d5754ab56abf))

## [Unreleased]
