# Rules for parallel agent sessions — read this first

Constantin **authorized this implementation at 13:16 CEST** (architecture v3.1 accepted). You are one of seven agents building in parallel. **Hard stop: 14:45 CEST.**

1. **Stay in your worktree.** Work only inside the worktree folder named in your prompt; it is already on your branch.
   - Do not switch branches, push, pull, merge, rebase, stash or reset.
   - Do not touch other worktrees or the main clone `metalearning-llmaaj`.
2. **Read the plan.** In `docs/IMPLEMENTATION_PLAN.md`, read §1 "Shared context" and your WP brief in §3. Where this file and the plan differ, **this file wins**: you commit locally and never push.
3. **Edit only the files your prompt lists.**
   - Never edit `rulejudge/contracts.py`, `rulejudge/render.py`, `rulejudge/jsonutil.py`, `rulejudge/fake_client.py`, `tests/fixtures.py`, or any `docs/*.md` (exception: WP7 owns `docs/README_SOLUTION_DRAFT.md`).
   - If a shared contract blocks you, stop and explain why in your final report.
4. **Read before you code:** your stub file(s), `rulejudge/contracts.py`, `render.py`, `jsonutil.py`, `fake_client.py` and `tests/fixtures.py`.
   - Keep the stub signatures. You may add optional keyword arguments with defaults and private `_helpers`.
5. **Test from your worktree root:** `python -m unittest discover -s tests -t .` — all tests must pass.
   - No network, API key, Docker or real Gemini calls.
   - `google-genai` may be missing locally: import it **lazily inside functions** (WP2 only). No other module imports it.
6. **Secrets:** never open, print, create or commit `.env` files, API keys or secrets.
7. **Commit early and often** once tests pass: `git add <your files>`, then `git commit -m "WP<N>: <what>"`. Your last commit before 14:45 must pass all tests. Small working increments beat one big unfinished change.
8. **Python version:** keep code compatible with 3.10+ (the local interpreter is 3.14; Docker uses 3.12). Standard library only — except `google-genai`, used only in `rulejudge/gemma_client.py`.
9. **Skip broad project research** (brief, decision log, literature). Everything you need is in the plan and the code.
10. **Final reply:**
    - (a) the functions you implemented;
    - (b) the test command and its result (number of tests, OK or failures);
    - (c) known gaps and assumptions;
    - (d) the output of `git rev-parse HEAD`.
