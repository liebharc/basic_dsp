CARGO_CMD ?= cargo

packages = vector interop matrix

test:
	$(MAKE) run-all TASK="test"	

run-all: $(packages)
	$(CARGO_CMD) $(TASK) --verbose

$(packages):
	$(CARGO_CMD) $(TASK) --manifest-path $@/Cargo.toml

.PHONY: $(packages) test