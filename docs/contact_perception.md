# Contact perception status

The implemented adapter consumes a fixed-label dangerous-part mask and reports
the fraction of object pixels assigned to the dangerous part. This is inferred
visual exposure. It is not sharpness, hardness, physical exposed area, contact
probability, or severity.

Sharpness, material hardness, and orientation toward the route require separate
specialists with their own confidence and provenance. Missing specialists stay
unavailable. No pretrained contact model or weights are integrated yet; model,
dataset, license, and restaurant-domain suitability still require verification.
