# Development

## Building against a local rmt-spca

scdim depends on the published `rmt-spca` from crates.io. To try changes in a
local checkout at `../rmt-spca` before publishing them, override the dependency
for a single command:

```sh
cargo build --config 'patch.crates-io.rmt-spca.path="../rmt-spca"'
```

The same flag works with `cargo test` and `cargo run`. To keep the override on,
put it in `.cargo/config.toml` (gitignored):

```toml
[patch.crates-io]
rmt-spca = { path = "../rmt-spca" }
```

Caveats:

- The local crate's version must satisfy the requirement in `Cargo.toml`.
  Otherwise Cargo ignores the patch and prints a warning.
- While the patch is active, Cargo rewrites `Cargo.lock` to point at the local
  path. Don't commit it in that state: run `git checkout Cargo.lock`, or
  `cargo update -p rmt-spca` once the patch is off.
- Don't put `[patch.crates-io]` in `Cargo.toml`. Anyone cloning scdim without
  `../rmt-spca` next to it would get a broken build.
