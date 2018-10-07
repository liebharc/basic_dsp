CARGO_CMD ?= cargo

packages = vector matrix interop

RUST_VERSION=$(shell rustc --version)
RUST_NIGHTLY = $(findstring nightly,$(RUST_VERSION))

test:
ifeq ($(RUST_NIGHTLY), nightly)
	$(MAKE) run-all TASK="test"
	$(CARGO_CMD) clean
	$(CARGO_CMD) test --manifest-path vector/Cargo.toml --no-default-features --lib
	$(CARGO_CMD) clean
	$(CARGO_CMD) test --features use_sse
	$(CARGO_CMD) clean
	$(CARGO_CMD) test --features use_avx
else
	$(MAKE) run-all TASK="test"
endif
	
bench:
ifeq ($(RUST_NIGHTLY), nightly)
	$(CARGO_CMD) bench --verbose --features use_avx
else
	@echo "Bench requires Rust nigthly, skipping bench for $(RUST_VERSION)"
endif

clean:
	$(MAKE) run-all TASK="clean"

update:
	$(MAKE) run-all TASK="update"
    
build:
	$(MAKE) run-all TASK="build"
      
build_all: build
	$(CARGO_CMD) clean --manifest-path vector/Cargo.toml
	$(CARGO_CMD) build --manifest-path vector/Cargo.toml
	$(CARGO_CMD) clean --manifest-path vector/Cargo.toml    
	$(CARGO_CMD) build --manifest-path vector/Cargo.toml --features use_avx
	$(CARGO_CMD) clean --manifest-path vector/Cargo.toml    
	$(CARGO_CMD) build --manifest-path vector/Cargo.toml --features use_sse
	$(CARGO_CMD) clean --manifest-path vector/Cargo.toml    
	$(CARGO_CMD) build --manifest-path vector/Cargo.toml --no-default-features
	
clippy:
ifeq ($(RUST_NIGHTLY), nightly)
	$(CARGO_CMD) clippy
else
	@echo "Skipping clippy for $(RUST_VERSION)"	
endif	
    
test_all: test
	$(CARGO_CMD) clean
	$(CARGO_CMD) test --manifest-path vector/Cargo.toml --no-default-features --lib
	$(CARGO_CMD) test --features use_sse
	$(CARGO_CMD) test --features use_avx

run-all: $(packages)
	$(CARGO_CMD) $(TASK) --verbose

$(packages):
	$(CARGO_CMD) $(TASK) --manifest-path $@/Cargo.toml

.PHONY: $(packages) test
