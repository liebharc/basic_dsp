CARGO_CMD ?= cargo

packages = vector interop matrix

RUST_VERSION=$(shell rustc --version)
RUST_NIGHTLY = $(findstring nightly,$(RUST_VERSION))
ifeq ($(RUST_NIGHTLY), nightly)
CARGO_FLAGS ?= --features use_sse
endif

test:
	$(MAKE) run-all TASK="test" FLAGS="$(CARGO_FLAGS)"
bench:
ifeq ($(RUST_NIGHTLY), nightly)
			$(CARGO_CMD) bench --verbose $(CARGO_FLAGS) FLAGS="$(CARGO_FLAGS)"
else
			@echo "Bench requires Rust nigthly, skipping bench for $(RUST_VERSION)"
endif

update:
	$(MAKE) run-all TASK="update"
    
build:
	$(MAKE) run-all TASK="build" FLAGS="$(CARGO_FLAGS)"
      
build_all: build
	$(CARGO_CMD) clean --manifest-path vector/Cargo.toml
	$(CARGO_CMD) build --manifest-path vector/Cargo.toml
	$(CARGO_CMD) clean --manifest-path vector/Cargo.toml    
	$(CARGO_CMD) build --manifest-path vector/Cargo.toml --features use_avx
	$(CARGO_CMD) clean --manifest-path vector/Cargo.toml    
	$(CARGO_CMD) build --manifest-path vector/Cargo.toml --features use_sse
	$(CARGO_CMD) clean --manifest-path vector/Cargo.toml    
	$(CARGO_CMD) build --manifest-path vector/Cargo.toml --features use_gpu
	$(CARGO_CMD) clean --manifest-path vector/Cargo.toml    
	$(CARGO_CMD) build --manifest-path vector/Cargo.toml --no-default-features    
    
test_all: test
	$(CARGO_CMD) clean
	$(CARGO_CMD) test --features use_gpu

run-all: $(packages)
	$(CARGO_CMD) $(TASK) --verbose $(FLAGS)

$(packages):
	$(CARGO_CMD) $(TASK) --manifest-path $@/Cargo.toml $(FLAGS)

.PHONY: $(packages) test
