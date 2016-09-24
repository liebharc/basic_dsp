CARGO_CMD ?= cargo

packages = vector interop matrix

RUST_VERSION=$(shell rustc --version)
RUST_NIGHTLY = $(findstring nightly,$(RUST_VERSION))
ifeq ($(RUST_NIGHTLY), nightly)
CARGO_FLAGS ?= --features use_sse
endif

test:
	$(MAKE) run-all TASK="test"
bench:
ifeq ($(RUST_NIGHTLY), nightly)
			$(CARGO_CMD) bench --verbose $(CARGO_FLAGS)
else
			@echo "Bench requires Rust nigthly, skipping bench for $(RUST_VERSION)"
endif

run-all: $(packages)
	$(CARGO_CMD) $(TASK) --verbose $(CARGO_FLAGS)

$(packages):
	$(CARGO_CMD) $(TASK) --manifest-path $@/Cargo.toml $(CARGO_FLAGS)

.PHONY: $(packages) test
