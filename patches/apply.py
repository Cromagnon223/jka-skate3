"""Small fixes applied to chasm's copy of the Skate 3 engine before building.
Run from the repo root after fetching chasm into ./chasm."""
from pathlib import Path

def patch(path, old, new):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    if new in s:
        return
    if s.count(old) != 1:
        raise SystemExit(f"patch target not found exactly once in {path}: {old[:60]!r}")
    p.write_text(s.replace(old, new), encoding="utf-8")
    print("patched", path)

ANIM = "chasm/skate/crates/skate-host/src/skater_animation.rs"
# Engine features the SK8 engine hasn't implemented yet (AirDismounting, foot
# plants, skitching, ...) used to abort the whole simulation. Skip them instead
# and keep skating: the graph just doesn't play that one behaviour.
patch(ANIM,
"""        if !self.action.errors.is_empty() {
            return Err(self.action.errors.join("\\n"));
        }""",
"""        if !self.action.errors.is_empty() {
            // sk3jka: unfinished engine features are skipped, not fatal
            eprintln!("sk3jka skipped: {}", self.action.errors.join(" | "));
            self.action.errors.clear();
        }""")
patch(ANIM,
"""        if !self.motion.errors.is_empty() {
            return Err(self.motion.errors.join("\\n"));
        }""",
"""        if !self.motion.errors.is_empty() {
            // sk3jka: unfinished engine features are skipped, not fatal
            eprintln!("sk3jka skipped: {}", self.motion.errors.join(" | "));
            self.motion.errors.clear();
        }""")
