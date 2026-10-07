SEP presence-policy candidate, not deployed
- Unknown template, failed embedding, ambiguous/mixed scores -> raw mix.
- Clear target match is provisional and expires after1sec if identity cadence slows.
- Muting needs2 fresh valid strong-negative refreshes and expires after250ms. Continuity alone never extends mute.
- Unit tests cover stale negative expiry, entry of clear target, uncertain reset, template replacement, invalid scores.
- Current thresholds remain provisional (.35/.10). No absence performance claim until synthetic target-only, other-only, target-entering and silence cases run on Pixel. Final integration pending benchmark winner.
- There is no proof that a1sec cached positive belongs to newly entering speech. UI must say provisional and never claim isolation guaranteed.
