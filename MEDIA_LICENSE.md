# Media license and attribution

The MIT license in `LICENSE` applies only to code and original documentation.
It does not replace the licenses of source videos or derived demo videos.

## 360+x-derived demos

- Dataset: 360+x
- Official page: https://x360dataset.github.io/
- Dataset/media license recorded by the project: CC BY-NC-SA 4.0
- Attribution: 360+x dataset and the original video authors

Derived ref/target demos that identify `dataset: 360x` in their metadata must be
used under CC BY-NC-SA 4.0. Commercial reuse is not granted by this repository.
The dataset documentation describes automated Gaussian face blurring for privacy;
this project therefore assigns all 360+x clips to REF only.

## Wikimedia Commons-derived demos

The current TARGET pool contains the following stationary-camera sources. The
exact per-file license and source URL are also copied into `SOURCE_AUDIT.json`.

- `360 Video of LCS-15 Christening and Launch` — CC BY 3.0.
- `An Introduction to the Equatorial Coordinate System (360VR)` — CC BY 3.0.
- `Wind Tunnel Test of NASA's Most Powerful Rocket (360° Animation)` — NASA public domain.
- `World's First Live 360 Rocket Launch: Orbital ATK CRS-7` — NASA public domain.
- `GOES-S Atlas V Launch in 360` — NASA public domain.
- `360VR Lotte Tower Grand Opening Fireworks (South Korea)` — CC BY 3.0.

## Per-case record

Every generated case records source file, dataset, time window, source URL and
license in `site/media/cases/<case_id>/metadata.json`. The aggregate screening
ledger is `output/SOURCE_AUDIT.json`.
