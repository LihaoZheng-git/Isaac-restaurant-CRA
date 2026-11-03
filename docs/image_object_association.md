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
and 40% image-area checks. The existing silhouette criterion also requires at
least 98% of mask pixels in one connected component, rejecting larger masks
with detached background patches. Masks are never filled or cropped to pass.
These are image hypotheses, not accepted objects.

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

At `5934b86`, both crate stationary components PASSED, each with 16/18 current
group assessments (first two sources qualify support). Bin mask/mass/motion
passed but its contact gate FAILED: the selected larger mask had 2084/2171
pixels in the main component, with two detached patches. The original 98%
contact connectivity gate correctly rejected it. The image proposal stage now
reuses that existing criterion to reject the fragmented candidate; a connected
broader body proposal is already present. No contact formula or threshold is
relaxed. This correction requires another fresh stationary series. All 3692
unit tests passed before that final proposal tightening.

At `2fbf8ab`, both crate components again PASSED; the bin kept a consistent
1953--1968 pixel body mask and rough mass 3.47--4.37 kg, but FAILED the current
contact gate (4/16 fresh shape cues; other masks had unresolved interior gaps).
All 3693 unit tests passed. The image extent prompt was then corrected to use
the same half-open pixel bounds as the existing audited retry (maximum pixel
+1 before padding). Archived replay improves the cue availability but does not
make it universal. Preserve this bin failure; no hole filling or contact-gate
relaxation is permitted to manufacture acceptance.

## Final stationary results

Frozen runtime `1124364`, fresh `*-native-v4` captures:

| Object | Component gate | Fresh masks / sources | Raw fragments (last) | Rough mass range, kg | Supported contact updates | Median phi |
| --- | --- | --- | --- | --- | --- | --- |
| Recycling bin | PASS | 16 / 18 | 1 | 3.31--4.12 | 6 / 16 | .65724 |
| Intact crate | PASS | 16 / 18 | 2 | 2.12--2.49 | 16 / 16 | .59995 |
| Broken crate | PASS | 16 / 18 | 3 | 2.66--3.10 | 16 / 16 | .60505 |

The first two sources qualify repeated support. Final selected return counts are
824/118/66 respectively. Mask sizes remain 1941--1975 / 1601--1606 / 1554--1564
pixels; the bin no longer alternates between a decorative panel and its body.
All three runs used zero wheel commands and unchanged source fingerprints.
The original component gate requires a current supported contact result, not
support on every frame: the bin passes at the final frame but remains intermittent
(ten of its sixteen assessments use the labeled .25 contact prior). Preserve
the earlier failures; this is not consistent bin-contact acceptance.

Both crates have sixteen supported silhouette cues. Their median contact factor
scores are .12464 (intact) and .13580 (broken): a modest observed difference,
not a strong sharpness demonstration or calibrated physical edge-risk estimate.
Potential and thermal remain .25 configured priors. Kinetic uses inferred raw
track motion plus rough inferred mass. Mass confidence remains capped at .25;
visible LiDAR extents are not complete dimensions and simulator ego pose is
still used for frame placement. No complete fresh four-factor claim is made.

Full regression: **3693 unit tests passed in 149.15 s**, 53 focused tests passed,
Ruff and diff checks passed. Median complete field-build wall times are 1.87,
1.95 and 1.95 s: paced stationary inference, not real-time acceptance. The 106
files indexed by the accepted-v2 manifest and all original asset-file hashes
match; that archive and its manifest were not modified.

Final stationary-gate SHA256 values:

- Bin: `ef6eed25dba42ec3f8624a4dcfe9a399946f30fc1f3b9ac16668b3fe056af7ad`
- Intact: `99c9ce8ab8ed52a77764ca5f7cab01065fc3f96df78f2d54926656da4c4128f1`
- Broken: `eda0c8dae858511ce2c3aedd8db151b8d28a5113a66bde5ee3e7bc986e23d242`

Local figure/data: `outputs/image-object-association/native-comparison-v4.png`
and `.json`; raw captures, all failures, plate surveys and the final full test
log remain alongside them. Asset credits are in
`assets/restaurant/sketchfab_contact_props/README.md` and on the figure.

The plate is still not an accepted obstacle instance. Across all eighteen sources
at the best tested south 2.8 m pose it has only 3--4 raw above-floor region
returns and zero filtered obstacle returns. A dedicated near-floor sensing and
localization study is needed before selecting it for the final demo. Do not
lower the global floor filter or add fake thickness for a pass. Dense neighboring
objects, moving-camera grouping, sustained contact reliability and navigation
integration also remain unvalidated. Keep this branch default OFF and draft;
no main merge or accepted-v2 replacement.
