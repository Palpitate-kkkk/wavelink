PYTHON ?= python3

.PHONY: check test audit smoke replay support crosscheck figures iq-demo sigmf-demo

# Run from repository root with the environment activated.
check: test audit smoke replay crosscheck iq-demo sigmf-demo

test:
	$(PYTHON) -m unittest discover -s tests -v

audit:
	$(PYTHON) -m wavelink.audit --report reports/reference
	$(PYTHON) -m wavelink.audit --report reports/support
	$(PYTHON) -m wavelink.audit --report reports/crosscheck
	$(PYTHON) -m wavelink.audit --report reports/overview
	$(PYTHON) -m wavelink.audit --report reports/iq_replay
	$(PYTHON) -m wavelink.audit --report reports/sigmf

smoke:
	$(PYTHON) examples/noiseless_loopback.py

replay:
	$(PYTHON) -m experiments.replay_reference

support:
	$(PYTHON) -m experiments.support_audit --output results/support

crosscheck:
	$(PYTHON) -m experiments.crosscheck --output results/crosscheck
	$(PYTHON) -m wavelink.audit --report results/crosscheck

figures:
	$(PYTHON) tools/build_overview_figures.py

iq-demo:
	$(PYTHON) -m experiments.iq_file_demo

sigmf-demo:
	$(PYTHON) -m experiments.sigmf_demo
