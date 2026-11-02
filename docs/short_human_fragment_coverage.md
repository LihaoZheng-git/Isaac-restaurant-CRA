# Bounded short human-fragment coverage experiment

`--human-short-fragment-coverage` is a separate default-OFF option. Navigation
also requires `--human-fragment-coverage-prior`; the zero-wheel component requires
its explicit validated body-envelope mode. The recipe
`configs/restaurant_service_demo_v2_short_fragment.json` changes only this option
on the component, startup/navigation and near-human commands.

The preserved furniture-fragment mission at6b12a65 stopped at source141.9s on
seven-point fragment3645, first observed at141.7s inside current human2367's body
envelope. It had one observation and no measured velocity. The original coverage
mode requires four observations and parent gaps<=.15s, so rejected it. Raw returns
did not support a.10m observed merge. Full mission acceptance failed:5/8 guides,
133.95s navigation. Report SHA256:
`b1d2f995c92012994c023113e2802acc2667982609b1b67ce876038db31ce01a`.

The user approved the following separate uncertainty policy:

- Store the complete positive fragment bounds, original risk, source and parent
  envelope provenance. A unique validated human envelope must contain the entire
  original rectangle. Even partial overlap by another human envelope rejects.
- A newly observed fragment may qualify without treating its initialized zero
  velocity as a static measurement. Its possible motion uses the existing
  all-direction1m/s point-speed PRIOR. An available excessive fragment velocity
  still rejects; parent speed and .3m/s inter-observation velocity-change bounds
  remain.
- At the missing observation, the same continuously validated parent must cover
  the original rectangle expanded by1m/s times actual original-source age.
  Parent source gaps may be at most the original.35s expiry in this mode. Missing,
  ambiguous or failed parent coverage permanently invalidates that old query;
  only new actual fragment returns can create a new source.
- Coverage never extends the original.35s expiry. The current built grid must
  cover every conservatively intersecting cell at at least the original risk.
  Domain clipping, holes and lower-risk cells reject. Applied envelope risk can
  rise to preserve that original floor; physical CRA factors are unchanged.
- No identity merge, mass/contact/thermal transfer, factor refresh, free-space
  claim or old-command continuation is introduced. Raw occupancy, body margins,
  fixed lambda1.25, nominal route and online command checks remain unchanged.

The module is staged with field admission and independently rerun by the nearby
continuity guard against the actual current occupancy. The component records
geometric proposals and actual grid checks separately. Its provenance audit
reports the number exercised; zero proposals would not establish an exercised
native exception. It is not an independent raw geometry or navigation audit.

Archived replay under `outputs/short-human-fragment/` preserves the strict-mode
rejection and finds the approved proof: original extent.281306m, expanded current
extent.487564m inside.65m. A deterministic reconstruction of the current body-only
raster subset covers87 cells at the original risk using the recorded rolling
bounds. This is an archived diagnostic, not the full saved field or a wheel run;
no actor truth supplies its input. Fresh actor/component/startup/near gates and
full actual arrival/stopping remain required. Accepted final demo/v1 and all
failed v2 records remain unchanged; main/primary/PR117 holds remain.

Before native execution:3450 unit tests passed in122.70s, including74 focused
coverage/real-CLI/recipe cases. Repository Ruff and diff checks passed. These
results include unchanged strict defaults, original expiry/no identity transfer,
both-epoch whole-region/ambiguity rejection, current field risk, no revival after
a missing parent, transactional field admission and source-provenance checks.
