# aviorio-images

Free image pipeline for Aviorio product photos, used by the aviorio-image-optimizer skill.

- Commit a plan to `plans/<product-handle>.json` and the `images` workflow runs on GitHub's servers.
- `"mode": "audit"` writes a numbered contact sheet and `audit.json` to `out/<handle>/audit/`.
- `"mode": "process"` runs each image's action (keep, cut, upscale, detext, studio) and writes the
  finished images plus a before/after `preview.jpg` to `out/<handle>/process/`.
- Nothing here touches the Shopify store. Store changes happen only after Aahil approves a preview.

Models: BiRefNet via rembg (MIT), Real-ESRGAN (BSD-3), LaMa (Apache-2.0), RapidOCR (Apache-2.0).
