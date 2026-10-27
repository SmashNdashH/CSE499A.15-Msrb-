# H2b decision (rule fixed 22 Oct, before any run)

Pass = answerable test-suite accuracy of at least 90% and RS(10) above 0.

- our-7B nf4: test-suite accuracy 69.0% [57.1, 80.5], RS(10) -89.6, unanswerable refused 94.4%: FAIL
- VL-3B nf4: test-suite accuracy 46.4% [32.2, 60.9], RS(10) -365.7, unanswerable refused 36.1%: FAIL
- Coder-1.5B nf4: test-suite accuracy 56.7% [44.4, 69.3], RS(10) -283.8, unanswerable refused 25.0%: FAIL
- Coder-3B nf4: test-suite accuracy 72.0% [58.2, 84.7], RS(10) -162.6, unanswerable refused 80.6%: FAIL

- Benchmark counting questions use the fixed query, not a model, whatever happens below.
- **Current system:** our 7B in 4-bit does not meet the bar.
- **Final low-resource system (provisional):** the base 3B in 4-bit does not meet the bar.
- **Fallback:** no separate model passes either (or none was run): ship fixed queries only, no free-form questions.
