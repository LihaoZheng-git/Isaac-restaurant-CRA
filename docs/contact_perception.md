# Contact perception status

The implemented adapter consumes a fixed-label dangerous-part mask and reports
the fraction of object pixels assigned to the dangerous part. This is inferred
visual exposure. It is not sharpness, hardness, physical exposed area, contact
probability, or severity.

Sharpness, material hardness, and orientation toward the route require separate
specialists with their own confidence and provenance. Missing specialists stay
unavailable. No pretrained contact model or weights are integrated yet; model,
dataset, license, and restaurant-domain suitability still require verification.

## PACO specialist audit

PACO is the strongest identified fixed-label source for this step. Its official
release covers 75 objects and 456 object-part categories, including `knife` and
`knife:blade`, and publishes part masks across COCO/LVIS images. The official
PACO code is MIT-licensed and the PACO-LVIS annotation archive has publisher
SHA-256 `02ac4edb22c251e07853e6231d69aec3fad0a180f03de2f8c880650322debc80`.

The repository does not explicitly state a license for the added PACO
annotations. Underlying COCO images retain per-image licenses. This distinction
blocks a claim that PACO-trained weights are cleared for redistribution until
the annotation terms are resolved.

The official PACO model zoo reports part-mask AP, but its R50/R101/ViT FPN
models require an older Python 3.9, PyTorch 1.10, Torchvision 0.11, and
Detectron2 stack. They are candidate reference teachers/evaluation baselines,
not compatible lightweight runtime integrations for the current environment.

`scripts/data/inventory_paco_contact_labels.py` reads only the small category
arrays from the large JSON files. It records actual knife-blade image/mask
counts before any COCO image download or training. The verified PACO-LVIS v1
archive contains 1,653 train images/2,528 knife-blade masks, 82 validation
images/120 masks, and 332 test images/540 masks. Even a successful blade mask
would support dangerous-part exposure only. It would not measure edge radius,
sharpness, material hardness, or contact severity, so those fields remain
unavailable unless independently supported.
