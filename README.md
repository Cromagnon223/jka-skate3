# JKA Skate 3

Skate 3 skating for Star Wars Jedi Academy (OpenJK), built on the
[Skate 3 Rust engine](https://github.com/SK8-ENGINE/skate-3-rust-engine) the way
[chasm's MW2 mashup](https://github.com/chasmlol/2010-rust-rewrite-mashup) uses it.

Private, personal project. No Skate 3 files are in this repo: the converter runs on
your own extracted copy.

- `sk3jka/` - C interface to the skate engine (`sk3jka.dll`)
- `.github/workflows/build.yml` - builds the DLL and the Skate 3 converter on Windows

Credits: SK8-ENGINE (skate engine, converter tools), chasmlol / IW4L (Apache-2.0;
skate bridge usage, rail finder, converter packaging), OpenJK (GPLv2).
