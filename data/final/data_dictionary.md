# Data Dictionary

Main dataset:

- `data/final/sec_defense_risk_dataset.csv`

Core columns:

- `annotation_id`: unique id for each final analysis unit
- `filing_id`: unique filing identifier built from ticker and accession number
- `ticker`: configured stock ticker used for the universe file
- `resolved_ticker`: SEC-resolved ticker symbol when share-class normalization is needed
- `company_name`: company name
- `company_layer`: `prime` or `supplier`
- `company_notes`: short rationale for why the company is in the universe
- `cik`: SEC Central Index Key
- `filing_date`: SEC filing date
- `filing_year`: filing year
- `period_bucket`: `pre_2022` or `post_2022`
- `comparison_window`: `pre_2018_2021` or `post_2022_2025`
- `form`: SEC form, currently `10-K`
- `accession_number`: SEC accession number
- `primary_document`: filing document name in EDGAR
- `source_url`: direct SEC filing URL
- `risk_section_char_count`: character length of the extracted Item 1A section
- `annotation_index`: within-filing order of the final analysis unit
- `merge_type`: `single`, `heading_pair`, `broken_pair`, or `bullet_split`
- `start_paragraph_index`: first source paragraph index used
- `end_paragraph_index`: last source paragraph index used
- `start_paragraph_id`: first source paragraph id used
- `end_paragraph_id`: last source paragraph id used
- `source_paragraph_ids`: source paragraph id or ids used to build the final row
- `source_paragraph_count`: number of source paragraphs used
- `list_context_text`: shared lead sentence carried into a split bullet item, if applicable
- `list_item_index`: position of the bullet item within its original list
- `list_item_count`: number of bullet items in the original list
- `text_char_count`: final text length in characters
- `text`: final cleaned analysis text
