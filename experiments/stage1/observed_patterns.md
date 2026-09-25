# Observed Data Patterns — Stage 1

Evidence: `matched_examples.txt` (30 random GT matches + 10 singletons), `sample_records.txt`,
`name_statistics.csv`, `address_statistics.csv`, `*_duplicate_*.csv`. Percentages are from the
full files, not the samples.

General rule seen everywhere: **S1 is clean and consistently formatted; S2 and S3 are
systematically corrupted copies of it**, each with its own formatting style.

## Business Names

- [x] Abbreviations — `Private Limited` → `Pvt Ltd` / `Private Ltd` / `प्रा. लि.`
- [x] Legal suffixes added/dropped/swapped — `Prairie Islands` → `Prairie Islands Inc.`; `LP` → `(LLC)`; `Direct Plum` → `Direct Plum Corp`
- [x] Punctuation differences — `Keys & Co` → `KEYS and CO`; `Riordan Ridge` → `Riordan-Ridge`; brackets `Cancer [Alliance]`, `(APPLIANCE)`; trailing commas (~1% of S2/S3 US names)
- [x] Typos — `Cotnsuhftanst`, `Scohlahohip`, `Allieamnec`, `Prdivaet`
- [x] Character substitution (leetspeak) — `Scho1arship`, `M0rgan`, `5unrise`
- [x] Word-order differences — `Coastal Co Avalanche`, `Limited Perfect ...`, `Anand Ventures Ltd Pvt`
- [x] Transliteration / native script — whole names in Devanagari, Malayalam, Bengali, or mixed (`आनंद Ventures प्रा. लि.`). 28% of S2-India and 18% of S3-India names contain non-ASCII; S1-India has 0%.
- [x] DBA/trade names — `Tavotavogild d/b/a Liz & Associates` (~1.4% of S3 names contain dba/aka)
- [x] Injected accents — `ANAND VÉNTURES`, `FAB FÁRMS`, `Prairie Íslands Íslands`
- [x] Case changes — 19% of S2 names are ALL CAPS, ~6% all lowercase; S1 has neither
- [x] Domain-style names — `fortunefinance.com`, `PRIVATEYADAVTRADINGCOM`, `sreetradecom` (~4% of S2/S3)
- [x] Honorific/filler tokens — `Smt`, `Mr`, `Dr`, `Sri` prefixes; `Center` suffix; junk prefixes `--`, `...`
- [x] Truncation — `Keys &`, `Lark G. Matthews,`, `AL MEDIA PRIVATE`
- [x] Acronyms — `DAC` for `Dorelle's Ace Cargo`
- [x] **Name replaced entirely** — `Zephkor`, `Pyravantagecira`, `Direct Partners` are true matches; only the address links them
- [x] Duplicated tokens — `Íslands Íslands`, `VIDYALAYA VIDYALAYA`
- [x] Generic/chain names — `Primary Care Group` ×253 in S1; 1.54M unique names across 2.21M S1 rows

## Addresses

- [x] Rd/Road, St/Street, Ave, Dr, Ct, Cir — S2/S3-US use abbreviations ~45%, S1-US only 3%
- [x] Wrong expansion — `Street` → `SAINT` (`47TH SAINT`, `WYETH SAINT`)
- [x] Missing PIN / ZIP — ZIP is in only ~11% of US addresses; PIN in ~1% of India addresses
- [x] Missing state
- [x] Missing city / street number — `SUTTON PARK RD, POUGHKEEPSIE`; `H.no.1/133, Keralam`
- [x] Landmark references — `Near Bhavani Mandap`, `Opp Hp Petrol Pump` (13% of S1-India)
- [x] Numbering variations — zero padding `00930`, `930.`, ranges `115-119`, suffixes `108-B`, `##`/`#867` prefixes, and changed digits `2022` → `4022` / `022`
- [x] Reordered components — `Illinois, # D, Sugar Grove, 109 Park Avenue`
- [x] State representation varies — S1-US uses codes (`IN`), S3-US spells them out (`Indiana`); S1-India spells out (`Maharashtra`), S3-India uses codes (`MH`) 59% of the time, S2/S3 sometimes native script (`महाराष्ट्र`, `ಕರ್ನಾಟಕ`)
- [x] Place-name variants — `Kerala`/`Keralam`, `Pune`/`Poona`, `Gurgaon`/`Gurugram`
- [x] Neighbouring/substituted locality — `Kettering` → `DAYTON`; `Poughkeepsie` → `Red Oaks Mill`; injected `CITY` suffix (`ROCKLAND CITY`)
- [x] Unit notation — `Unit C` → `# C`; `Unit STE 210` → `# STE 210`
- [x] Placeholder components — literal `NULL` / `N/A` inside ~2.7% of S2/S3 addresses (never as a whole value)
- [x] Fully missing address — 3.3% of train S2/S3, 2.7% of test S2/S3; S1 never
- [x] Upper case — 93% of S2-US addresses are ALL CAPS; S3 almost never
- [x] Repeated generic addresses — S3-India `Floor, Mumbai, MH` ×26 (truncated to unit + city)

## Country

- [ ] Country spelling variations — none: exactly one spelling per country (`US`, `India`, `France`)
- [ ] Missing country — none, in any file
- [x] Countries only appearing in test — `France` (~15% of every test file)
- [x] Test country mix differs from train — test India 47% / US 38% vs train India 40% / US 60%

## France (test only — no labels)

- Legal forms: `SARL`, `SAS`, `SASU`, `EURL`, `SCI`, `SA`, dotted `E.U.R.L.`; `(France)` injected into names
- Street abbreviations: `Rue` → `R.` / `R`, `Boulevard` → `BD.`, `Route` → `RTE`; `N°` / `NO` before numbers
- Region vs department: `Hauts-de-France` vs `Nord`; `Nouvelle-Aquitaine` vs `Gironde`
- Accent noise like the other countries (`Frèrês`, `Immôbilier`)
- Concentrated in few cities (Lille, Bordeaux, Nantes, Tourcoing, Roubaix, Dunkerque) → large city blocks
- Almost no postcodes (0.4% have 5 digits)

## Stage 1 decision table

| Finding (measured) | What it means for Stage 2+ |
| --- | --- |
| 5.58% singletons (US 5.58%, India 5.59%) | Singletons are rare but each is worth a full 1.0 → need a "no match" option, but the prior strongly favours matching |
| 94.4% of S1 have ≥1 match, mean 3.67, max 11 | Many-to-one: never force one-to-one; per-S1 output is a set |
| Each S2/S3 record links to **at most one** S1 | Matches partition S2/S3 → assign each S2/S3 to its single best S1 (exclusivity constraint helps precision) |
| ~26% of S2/S3 records are never matched | Distractors exist → high-recall blocking must be followed by a strict matcher |
| 100% of links are same-country | Country is a safe hard blocking key (open-set string, France included) |
| Up to 5 S2 and 6 S3 matches per S1 | S2/S3 contain internal duplicates → S2/S3 records can support each other (clustering evidence) |
| Generic names repeated hundreds of times | Name alone is unsafe; need address evidence |
| Some true matches have a totally different name | Name alone also isn't *required* → blocking needs an address-only path |
| ~3% missing addresses in S2/S3 | Blocking needs a name-only path too; missing-address flag as a feature |
| Typos, leetspeak, injected accents | Character n-gram / edit-distance similarity; accent stripping; digit→letter folding |
| Abbreviations, legal suffixes, honorifics | Normalization dictionaries (per country, incl. French forms) |
| Native-script names and state names | Transliteration or script-aware matching is needed for ~20-28% of India S2/S3 names; address evidence is critical for those |
| State as code / full / native script | Canonicalize state to one form per country |
| House numbers altered or padded | Number match is strong evidence but must tolerate padding/prefix noise |
| Word-order shuffles and component reordering | Token-set (order-invariant) similarities |
| France absent from train | Normalization and features must be country-agnostic or have explicit French rules; no country one-hot |
| Test country mix ≠ train | Validate per-country, not only overall |
