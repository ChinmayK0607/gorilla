BFCL Trajectory Verifier — Minimal Prompt

Inputs you see:
- Task text (user turns, in order).
- Tool calls made in the trajectory and their tool responses.
- List of tools available in the final environment (after the trajectory). No other tools are allowed.
- Use the task’s own tools (the same ones available to the model) to verify the trajectory: rerun key calls or equivalent checks to confirm outputs and detect hallucinated tool use. Prefer safe/read checks; avoid unnecessary writes.


How to judge
- Classify the task: Read (get info), Write (make changes), or Mixed.
- Verification: run your own minimal calls with the allowed task tools (re-run the same call or an equivalent check) to confirm the trajectory’s tool outputs and state. Flag hallucinated/incorrect tool calls or outputs.
- Tool validity: if any call uses a tool not in the available list (or a forbidden tool, if given) → score 0.
- Read tasks: did the assistant deliver the requested info, consistent with the shown tool outputs and your spot checks? Missing info, contradictions, or unsupported claims lower the score.
- Write tasks: did the shown calls (given their responses) achieve the requested changes? Use reads/inspects to confirm state. Unfixed errors, missing steps, or wrong arguments lower the score.
- Mixed tasks: score the read and write parts separately, then average.

Scoring (0–1, 2 decimals)
- Start at 1.0. Subtract reasonable penalties for each unmet requirement (missing info/change, wrong tool, hallucination). Floor at 0.
- Hard zero if using unavailable/forbidden tools or the main task is not addressed.

Output (JSON only)
```json
{"score": 0.00, "critic": "brief rationale citing task parts and key tool calls/responses"}
```
