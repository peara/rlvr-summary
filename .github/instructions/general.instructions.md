---
applyTo: '**'
---
When there are uncertainty, unreasonable or not optimal points in an issue, should ask for clarification by commenting to the issue.
We are doing a research project, so:
- things must work, or just fail. We don't need things run but not correct.
- backward-compatible is not necessary.
- no need for demo, better update README and tests.
- we are working with cutting-edge tools and libs. if you don't know the docs, ask immediately. DO NOT GUESS.

What we are researching:
- PPO with Functional Reward for Text Summarization with LLM
- Using VERL as Reinforcement Learning framework
- Using Rule to calculate format, style, and other scores
- Using FENICE to calculate factuallity score

Project structure:
- main package is in `src/rlvr_summary`
- tests are in `tests`
- configs are in `configs`
- docs are in `docs`
- scripts are in `scripts` with each experiment is in 1 script