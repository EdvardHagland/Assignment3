# Scraper

## Purpose

This folder contains the SEC ingestion and cleaning pipeline for the Assignment 3 corpus. The scraper is designed to build one coherent corpus of `10-K Item 1A. Risk Factors` from publicly listed U.S. defense-related firms.

The goal is not to scrape every company with any defense exposure. The goal is to assemble a universe that is analytically useful for comparing how risk is framed across different positions in the defense-industrial ecosystem.

## Scripts

- `sec_fetch_risk_factors.py`
  Downloads SEC filings, extracts `Item 1A`, and writes filing-level and paragraph-level outputs.
- `prepare_annotation_paragraphs.py`
  Cleans the extracted paragraphs into final annotation-ready units.

## Why These Companies

The company universe in `config/defense_companies.csv` is intentionally structured around two layers:

- `prime`
- `supplier`

This lets us compare large downstream contractors against upstream firms that face more direct exposure to materials, components, manufacturing bottlenecks, and sustainment constraints.

### Selection principles

The list is not meant to be exhaustive. It is meant to be defensible.

We prioritized firms that meet one or more of these criteria:

- significant and visible exposure to U.S. defense or national-security markets
- public SEC reporting that makes the corpus reproducible
- strategic variation in business model, such as primes, services, cyber, shipbuilding, satcom, autonomy, or components
- relevance to the project's label set, especially procurement, execution, industrial base, cyber, export controls, labor, and technology
- enough sectoral breadth to compare different kinds of defense risk language without making the corpus shapeless

### Prime layer rationale

These firms are included because they provide broad coverage of the major types of defense-facing actors whose published risk language is likely to differ in meaningful ways.

| Ticker | Company | Why include |
| --- | --- | --- |
| `AVAV` | AeroVironment | Unmanned systems and loitering munitions; useful for autonomy and drone-related risk language. |
| `BAH` | Booz Allen Hamilton | Core defense and intelligence consulting plus digital services comparator. |
| `BA` | Boeing | Major defense prime with large program, fixed-price, and supply-chain exposure. |
| `CACI` | CACI International | Defense and intelligence IT, cyber, ISR, and mission systems exposure. |
| `GD` | General Dynamics | Major prime across land, naval, IT, and aerospace. |
| `HII` | Huntington Ingalls Industries | Naval shipbuilding and industrial-base capacity exposure. |
| `KBR` | KBR | Mission engineering, logistics, government services, and support functions. |
| `KTOS` | Kratos Defense & Security Solutions | Defense tech, drones, tactical systems, and emerging capability areas. |
| `LDOS` | Leidos | Large mission-IT and federal services benchmark. |
| `LHX` | L3Harris Technologies | Major defense electronics, ISR, communications, and tactical systems actor. |
| `LMT` | Lockheed Martin | Major prime; baseline defense risk language for the sector. |
| `NOC` | Northrop Grumman | Major prime with strong space, aerospace, and strategic systems exposure. |
| `PSN` | Parsons | Defense and intelligence engineering, cyber, and geospatial exposure. |
| `RTX` | RTX | Major prime across missiles, sensors, aerospace, and sustainment. |
| `SAIC` | Science Applications International Corporation | Large mission-services and federal IT comparator. |
| `TXT` | Textron | Defense aviation, rotorcraft, and unmanned systems. |
| `VVX` | V2X | Mission support, logistics, base operations, and sustainment risk. |
| `BWXT` | BWX Technologies | Naval nuclear and strategic industrial-base relevance. |
| `CW` | Curtiss-Wright | Defense components, naval systems, and industrial supply-chain exposure. |
| `TDY` | Teledyne Technologies | Sensors, imaging, ISR, and defense electronics. |
| `MRCY` | Mercury Systems | Defense electronics and component or supply-chain stress case. |
| `IRDM` | Iridium Communications | Government satcom and resilience or space exposure. |
| `VSAT` | Viasat | Government communications, satcom, and cyber or international risk. |

### Supplier layer rationale

These firms are included because a prime-only corpus would miss important upstream risk language. Suppliers often frame risk in terms of capacity, specialty materials, bottlenecks, pricing, quality, and dependence on a smaller number of programs or customers.

| Ticker | Company | Why include |
| --- | --- | --- |
| `AIR` | AAR Corp. | Aviation services, sustainment, spares, and MRO exposure. |
| `ATI` | ATI Inc. | Specialty materials and alloys; industrial-base constraint signal. |
| `CRS` | Carpenter Technology | Specialty metals and materials; useful for capacity and raw-material risk. |
| `DCO` | Ducommun | Components and electronics supplier to aerospace and defense. |
| `HEI` | HEICO | Aerospace and defense parts and electronics supplier. |
| `HWM` | Howmet Aerospace | Aerospace components and forged or engine-linked supply-chain exposure. |
| `HXL` | Hexcel | Composites and materials; industrial-bottleneck indicator. |
| `MOG.A` | Moog | Flight controls, motion control, and defense systems. |
| `TDG` | TransDigm | Components and pricing or supply-chain exposure. |
| `VSEC` | VSE Corp. | Distribution, maintenance, and fleet support. |
| `WWD` | Woodward | Aerospace controls and subsystem supplier. |

## Why This Structure Matters

This universe is designed to support substantive comparisons such as:

- `prime` versus `supplier`
- services-heavy firms versus industrial manufacturers
- shipbuilding, satcom, and autonomy niches versus broad primes
- firms likely to emphasize budget and procurement risk versus firms likely to emphasize materials, execution, and capacity risk

That is why the corpus is broader than a simple “top defense primes” list, but still narrow enough to stay interpretable.

## Universe caveats

A few practical notes:

- The list is U.S.-listed because SEC filings provide a stable and reproducible reporting regime.
- The universe is defense-focused, not a perfect legal taxonomy of "defense companies."
- Some issuers have multiple share classes. The scraper normalizes ticker variants and skips duplicate issuers by `CIK`, so we keep one company rather than duplicating the same filing set.
- Inclusion in the universe does not mean every firm produces equally clean `Item 1A` extraction. Some firms will still need extractor improvements.

## Outputs

The scraper stage writes generated files into `data/intermediate/processed/`:

- `sec_10k_risk_sections.csv`
- `sec_10k_risk_paragraphs.csv`
- `sec_10k_risk_coverage_report.csv`

The coverage report is especially useful for checking which configured companies were resolved cleanly, how many filings matched the date window, how many sections were extracted, and which accessions failed extraction.

The cleaner then writes the tracked final dataset to `data/final/sec_defense_risk_dataset.csv`.
The intermediate files are ignored by Git because they are regenerated by the
scraper/cleaner path.
