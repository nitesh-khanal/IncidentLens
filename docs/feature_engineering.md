# Feature Engineering

## Target
`issue_type` (8 balanced classes, confirmed in Stage 5 EDA) — Stage 15
finalizes this as the classification target.

## Included features (known at ticket-creation time)
- **TF-IDF text vector** (99 dims, Stage 7) — the incident description itself.
- **Categorical (one-hot encoded)**: customer_segment, channel, product_area,
  priority, sla_plan, platform, region.
- **Temporal**: hour_of_day, day_of_week, month — extracted from created_at.
- **Numeric**: message_length (character count), has_attachment (0/1 flag,
  already present in the raw data).

## Excluded as leakage
resolution_summary, resolution_time_hours, has_resolution, status, reopened,
csat_score, customer_sentiment.

**Why:** all of these are determined during or after resolution — a real
deployment of this classifier would run the moment a ticket is created,
before any of these fields exist. Including them would let the model
"cheat" by learning patterns from post-resolution data unavailable at
prediction time, producing inflated accuracy that would not hold in
production. `customer_sentiment` is excluded defensively — its exact
capture timing isn't documented in the source dataset, so it isn't
assumed safe.

## Verification
`scripts/build_features.py` includes an explicit runtime assertion that
none of the LEAKAGE_FIELDS ever appear in the final feature name list —
not just a design intention, but a checked guarantee.
