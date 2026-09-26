# RPS checkpoint evaluation

## Overview

Measured behavior of training job 858's final frozen policy. This is an
evaluation of uniformity and observation/history dependence, not additional
training or a claim about every possible recurrent state.

The checkpoint uses the standard PufferNet: six observation values, hidden
width 16, four minGRU layers, and three discrete action logits plus a value.
Training was current-policy self-play, with memory reset at 1,000-round match
boundaries. Its final checkpoint has 1,999,962,112 agent steps (the requested
2 billion was rounded down to complete batches).

## Measured results

| Situation | Rock | Paper | Scissors |
| --- | ---: | ---: | ---: |
| First round, zero observation and zero memory | 34.18677252% | 31.72173558% | 34.09149190% |
| Subsequent rounds on evaluated trajectories | 33.33285543% | 33.33435155% | 33.33279302% |
| Exact uniform reference | 33.33333333% | 33.33333333% | 33.33333333% |

All eight opponent scenarios produced the same probability sequence, at the
10-significant-digit precision of the saved CSVs. The largest post-opening
deviation from 1/3 was about 0.001018 percentage points. The opening is a real
exception to a claim of *always exactly uniform*.

Tests performed:

- 64 matches of 1,000 rounds against each of self-play, uniform random,
  always rock, always paper, always scissors, cycling, copying the learner's
  previous move, and win-stay/lose-shift (ties stay; losses advance one move).
  Total: 512,000 learner decisions. Moves were sampled from softmax, not argmax.
  Independent successive PRNG draws selected each side's move. Bots used no
  information about the learner's current sampled action.
- 2,048 stored recurrent histories (round indices 1, 10, 100, and 999 across
  the matches), each probed with all nine previous-move pairs while holding
  hidden state fixed. No probability difference was detected. Probe calls
  restored the original state before the actual trajectory continued.
- Holding each previous-move pair fixed across these recurrent histories
  also produced no detected probability difference.
- With memory artificially zeroed, all nine valid previous-move pairs gave
  uniform output. These are diagnostic interventions, not natural match starts.
- Native GPU `arch_forward` and the repository's CPU PufferNet agreed on
  10,000 probability vectors across ten controlled 1,000-step histories.
  Maximum difference at saved precision: 0.0 (acceptance threshold: 1e-5).
- Actual sampled self-play moves: rock 33.309375%, paper 33.3609375%, scissors
  33.3296875% over 64,000 rounds. Sampling fluctuations are distinct from the
  policy's directly measured probabilities.

The observed near-uniform policy does not adapt to exploit the tested
predictable bots: even an always-rock opponent leaves its output essentially
uniform. Balanced self-play scores alone would not establish this; the
probability and controlled-input measurements above are the evidence.

Limitations: finite histories and float32 inference, not a proof of
independence over all possible states or of PRNG independence. CPU/GPU parity
checks cover controlled inputs, not all 512,000 evaluated decisions. Confidence
intervals in summary.json are approximate, per-match reward intervals; isolated
sampled reward deviations are not evidence of learned exploitation. Variances
computed as E[p²]-E[p]² can show tiny cancellation artifacts; use saved min/max
and the controlled probability ranges for the independence test.

## Reproduction and provenance

- Training job: 858; snapshot `5a2e8ea3104d3e7dd706b373e06f73cb8d12ff26`.
- Checkpoint SHA256:
  `6ca507f3f12ac323e180df5498ea00dc68ef3c0a9765c6fc66adcf716c1c31ed`.
- Checkpoint:
  `/cluster/users/tosma/projects/pufferlib-local/jobs/a06fc0ee-da58-4b12-8de4-575f32d5dc10/artifacts/checkpoints/rock_paper_scissors/1790433943269/0000001999962112.bin`.
- CPU evaluation job: 859; snapshot
  `f87f0ac6a7681feeea614e2e6155fa23847ab184`.
- CPU results:
  `/cluster/users/tosma/projects/pufferlib-local/jobs/4299ea91-5cc4-41fc-ab34-3a68de66e5c9/artifacts/`.
  Contains summary.json, trajectories.csv, probes.csv, and provenance.txt.
- GPU parity job: 860; snapshot
  `a25ab58e333361ee3bff1b6e349ef3a41e1664e7`.
- GPU results:
  `/cluster/users/tosma/projects/pufferlib-local/jobs/9e3cebb7-fd56-49c2-8ebb-9839dd14e903/artifacts/`.
  Contains parity.json and CPU/GPU probability CSVs.
- Reproduction entry points: `scripts/rps_policy_eval.sh` (1 CPU, 1 GiB RAM,
  no GPU, 4 GiB scratch, 10 minutes) and `scripts/rps_gpu_check.sh` (2 CPUs,
  4 GiB RAM, one GPU, 4 GiB scratch, 10 minutes). Submit through `cluster run`.
  The CPU evaluator has only two small networks resident and streams CSVs;
  its allocation includes compiler/interpreter overhead. The GPU check uses
  the already validated build budget and ten inference slots.
- CPU inference, GPU kernels, and environment source were byte-compared
  against the training snapshot and matched. Each evaluation verifies the
  checkpoint hash before reading it. No training weights were modified.

## Worklog

- 2026-09-26: Evaluated job 858's final checkpoint; jobs 859 and 860 completed
  with exit code 0. Observed a biased opening followed by near-uniform outputs
  with no detected previous-move/history dependence in the tested cases.
  Outputs were published, artifact hashes verified, and scratch cleanup
  confirmed. Further training not run. Potential next steps: inspect earlier
  checkpoints to see when this behavior emerged; separately design a bot
  mixture if history-dependent exploitation is the desired training objective.
