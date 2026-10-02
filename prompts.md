# Localization
## Anchor Prediction
```
Name the most likely functions where the bug lives.
Prefer the PRODUCER, not the symptom surface: if the issue describes a wrong/extra/missing value,
name the function that CONSTRUCTS/PARSES it, not the outer API that renders it.
Traps: the issue author's suggested fix site and the deepest traceback frame are hypotheses, not facts.
For a multi-fault issue, name at least one candidate per broken behaviour.
```
## Execution Path simulation
```
Do NOT write or run code; trace it as if you were the interpreter.
For each execution path, predict the following fields:
**CONCRETE INPUT**: one plausible input that shows the issue. You may invent one if the issue gives none.
**EXECUTION SIMULATION**: follow the caller path the input takes into the anchor and then down its callees.
For each hop, state what it receives and what it evaluates to.
Follow the wrong value back to its birthplace: <value> is FIRST CONSTRUCTED here, in <function> at <file:line>.

**FRAMES THAT MUST CHANGE**: every function the fix must edit for an end-to-end fix.
```

# Repair \& Refine
```
For EACH requirement and interface clause the units implement, pick one or two concrete inputs.
Simulate the post-patch path line by line and compare the result with the clause.
Each divergence is a BEHAVIOUR VIOLATION.
```

# Audit
## Requirement Extraction
```
Turn the issue and spec into TESTABLE points.
Each point is one observable behaviour, interface, signature, return shape, error message or edge rule, with the sentence it comes from quoted verbatim.
Do not invent requirements.
Prefer points a hidden unit test could check with a concrete input.
```

## Requirement Audit
```
You are an SPEC AUDITOR with NO tools; nothing you write will be executed. Assume each point is VIOLATED until the quoted code proves otherwise.
[id] VERDICT: SATISFIED | VIOLATED | UNVERIFIABLE | CONFLICT
EVIDENCE: quoted pack or diff lines
SIMULATION: execute the patched code mentally on ONE concrete input, using the spec's own example values

SATISFIED requires quoted evidence plus a simulation. 
Every point is behavioural (n/a is not allowed).
Follow the value through all call sites: a right value that a caller discards or overwrites is VIOLATED.
Contradictory clauses give CONFLICT.
End with AUDIT SUMMARY: VIOLATED: <ids>.
```
# Validate
```
**STEP A**: INPUT → EXPECTED table: the exact input, the exact expected literal (copied character for character if the spec shows it),
and the source sentence. Derive EXPECTED from the spec, never from what the patched code produces.
Use the minimal input that isolates the requirement, so a row cannot pass through a different path.
**STEP B:** mine corner cases (mandatory): empty, zero, None or missing; single vs multiple; duplicates; boundary and maximum values; type extremes; error paths; one input reaching each conditional branch the fix introduces. Add at least N corner rows.
```
