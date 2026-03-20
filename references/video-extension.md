# Video Extension (Future)

## Status: Out of scope for current version. Image-only.

## Planned additions when extending to video:
- Temporal consistency agent: detect flickering, unnatural motion
- rPPG agent: full temporal signal extraction across frames
- Lip sync agent: detect audio-visual misalignment
- Frame-level pipeline: run image pipeline on sampled frames, aggregate scores

## Video input schema (future):
```json
{
  "input_type": "video",
  "path": "string",
  "sample_rate": 1
}
```
