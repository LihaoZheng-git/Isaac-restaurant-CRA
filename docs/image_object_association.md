# Stationary image-object association experiment

This development experiment addresses the real crate's disconnected LiDAR
slats and the recycling bin's unstable panel/body masks. It is default OFF and
the CLI permits it only with the stationary prop camera, all-object mapping,
and no humans or wheel motion. Accepted v2 and primary navigation are unchanged.

`--stationary-image-objects` uses the already installed, hash-audited SAM 2.1
tiny backend. Automatic image proposals (32 by 32 point grid) are supplemented
by at most 16 current observed-fragment extent prompts. No category, authored
target box, simulator instance mask, material or density enters inference.
At most 64 proposals survive the existing .85 quality, 200 pixel, image-boundary
and 40% image-area checks. These are image hypotheses, not accepted objects.

For each mask, the adapter selects exact current projected foreground returns.
Each included fragment needs at least four hits and 80% visible coverage; the
group needs at least eight points and at most 15% unresolved foreground hits.
The existing .5 m foreground-depth partition excludes farther background.
The group's maximum axis extent is bounded by the explicitly configured .9 m
separated-prop assumption (`--image-object-maximum-extent`). This does not prove
that nearby disconnected surfaces are one physical object. Merged masks with
two disjoint viable child hypotheses and competing nonnested masks reject.
Contained parts are suppressed in favor of a viable larger outline.

Three distinct sources spanning .2 s must support the original mask (IoU >=.75)
and original geometry (bidirectional nearest-return distance <=.06 m). Gaps
over .35 s reset the repetition proof. An original outline reference can reject
part-only updates for at most ten seconds from its original source; it supplies
no geometry, factor, mass or refreshed evidence. A missing/partial mask does not
publish a fresh group assessment. Raw IDs may change only while current geometry
independently satisfies this proof; raw identities are never merged/transferred.

Verified groups reuse the current mass, shape/contact, support and plume
estimators and the four-factor prior adapter. Per-return raw inferred velocities
are averaged for the group's motion input, not replaced with a measured-zero
claim. This remains vulnerable to centroid/visibility changes. All factors retain
their measured/inferred/prior/unavailable provenance. Existing geometry, unmatched
returns, fixed lambda and original field checks remain. Group state and plume
history commit only when the field is admitted. No cached group mask or factor
is represented as a new measurement.

## Broken plate sensing survey

The original 28 cm plate is only about 2.6 cm high. Original 1.85 m and new
1.1 m views have no above-floor raw returns in its independent bounding region.
Farther 2.5/2.8 m views along the original axis failed the unchanged setup
clearance guard before sensing. Clear south views were then selected by the
independent scene-layout check. At 2.5 m the plate remains unsampled; at 2.8 m
the last scan has three raw returns at about 2.53--2.58 cm, but zero after the
unchanged 5 cm obstacle filter. Authored bounds only score this diagnosis.

These results do not establish usable plate association or a contact comparison.
Lowering the global obstacle filter would admit floor/noise and does not solve
sparse support. Stereo/depth localization or another realistic supported
presentation requires separate sensor/method validation; it is not implemented.

## Validation status at implementation

Offline last-frame replay obtains a 2195-pixel bin outline, an intact-crate
1604-pixel mask with 117 returns across two raw fragments, and a broken-crate
1558-pixel mask with 66 returns across three fragments. These are hypotheses
on archived images, not temporal/native acceptance. Native stationary tests
and the complete regression run remain pending at this commit.

Recipes: `configs/restaurant_image_objects.json` and
`configs/restaurant_plate_survey.json`. Local evidence is retained under
`outputs/image-object-association/`. No navigation or final-demo acceptance.

The first fresh native series at `56c8db6` completed all three 18-source
zero-wheel runs with frozen source but FAILED the component gates. All three
targets had 16 consecutive verified broader masks after qualification. Two
integration defects remained: the strict mass adapter rejected the new source
name, and fixture reporting counted raw member fragments as competing objects.
The new source now needs a separate explicit mass opt-in (same geometry,
confidence, freshness and domain limits); it is never disguised as the old
point-prompted source. Object-level evaluation coalesces only the listed members
of a fresh verified group, retaining unrelated competitors and the original
.20 m unique fixture gate. Original records and occupancy are untouched.

Four regression failures came from old mock proposals lacking a source field;
the original source is now the backward-compatible default. A further adversarial
test prevents a duplicate mask from reviving a rejected nonnested conflict.
Fresh native repetition is required after these repairs. The initial stale
asset-README manifest rejection and every v1 failed native record are retained.
