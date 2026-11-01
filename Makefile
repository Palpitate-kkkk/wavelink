PYTHON ?= python3

.PHONY: check test audit smoke replay support

# Run from repository root with the environment activated.
check: test audit smoke replay

test:
	$(PYTHON) -m unittest discover -s tests -v

audit:
	$(PYTHON) -m wavelink.audit --report reports/reference
	$(PYTHON) -m wavelink.audit --report reports/support

smoke:
	$(PYTHON) examples/noiseless_loopback.py

replay:
	$(PYTHON) -m experiments.replay_reference

support:
	$(PYTHON) -m experiments.support_audit --output results/support
