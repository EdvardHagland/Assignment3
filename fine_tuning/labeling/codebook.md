# Risk Coding Codebook

## Purpose

This codebook is for labeling SEC `10-K Item 1A` risk-disclosure units from defense-sector firms.

Each row gets exactly one `primary_label`.

The current label set is intentionally more specific than the earlier pilot version. It is designed to separate major defense-sector risk domains without making the interface heavier for annotators.

## General coding rule

Choose the category that best captures the main risk being foregrounded by the company.

Do not code based on:

- whether the risk is true
- whether the company is exaggerating it
- whether you think another risk is more important

Code what the text is primarily about.

## Labels

### 1. `GOV_BUDGET_PROCUREMENT`

Government demand, budgets, procurement.

Use for:

- appropriations and defense budgets
- program funding
- contract awards and recompetes
- customer concentration
- DoD or federal spending priorities

### 2. `CONTRACT_EXECUTION_ECONOMICS`

Contract performance and fixed-price economics.

Use for:

- cost overruns
- fixed-price exposure
- schedule delays
- estimates at completion
- warranty exposure
- termination risk
- margin compression on contracts

### 3. `SUPPLY_CHAIN_INDUSTRIAL_BASE`

Supply chain, manufacturing, industrial base.

Use for:

- suppliers and subcontractors
- component shortages
- raw materials
- lead times
- sole-source suppliers
- manufacturing capacity
- quality failures
- logistics

### 4. `LABOR_CLEARANCES_HUMAN_CAPITAL`

Labor, talent, clearances.

Use for:

- hiring and retention
- skilled labor shortages
- engineers and scientists
- unions or strikes
- key personnel dependence
- security clearances

### 5. `CYBER_DATA_SYSTEMS`

Cybersecurity, data, IT systems.

Use for:

- cyberattacks
- ransomware
- data breaches
- IT outages
- CMMC and NIST-related exposure
- customer or supplier network security
- data privacy linked to systems risk

### 6. `LEGAL_REGULATORY_COMPLIANCE`

Legal, regulatory, compliance.

Use for:

- litigation
- investigations
- procurement-law compliance
- False Claims Act
- FCPA
- audits
- classified-information handling
- suspension or debarment
- general regulatory exposure

### 7. `GEOPOLITICAL_INTL_SANCTIONS_EXPORT`

Geopolitical, international, sanctions, export controls.

Use for:

- Russia or Ukraine
- China or Taiwan
- Middle East instability
- sanctions
- export controls
- tariffs
- foreign governments
- international sales
- trade restrictions
- political instability

### 8. `MACRO_FINANCIAL_CAPITAL`

Macro, inflation, rates, liquidity, capital markets.

Use for:

- inflation
- interest rates
- SOFR
- recession
- debt and credit facilities
- liquidity
- taxes
- pensions
- insurance
- impairment
- capital-market access

### 9. `TECH_PRODUCT_AI_AUTONOMY`

Technology, product innovation, AI, autonomy.

Use for:

- AI
- autonomy
- unmanned systems
- drones or UAS
- product obsolescence
- R&D risk
- advanced technologies
- technological competition

### 10. `IP_PROPRIETARY_RIGHTS`

Intellectual property and proprietary rights.

Use for:

- patents
- trade secrets
- infringement claims
- proprietary technology protection
- IP disputes

Keep this separate from technology innovation when the row is really about ownership, protection, or infringement rather than product development.

### 11. `M_AND_A_PORTFOLIO_STRATEGY`

M&A, integration, portfolio strategy.

Use for:

- acquisitions
- divestitures
- joint ventures
- integration risk
- restructuring
- synergies
- portfolio changes

### 12. `ENV_CLIMATE_ESG_SAFETY`

Environmental, climate, ESG, health and safety.

Use for:

- climate change
- emissions
- ESG or sustainability commitments
- environmental remediation
- hazardous materials
- natural disasters
- workplace health and safety

### 13. `GOVERNANCE_SECURITIES_STOCK`

Governance, securities, shareholder risk.

Use for:

- stock volatility
- securities litigation
- internal controls
- takeover defenses
- shareholder rights
- public-company governance

### 14. `PANDEMIC_PUBLIC_HEALTH`

Pandemic and public-health disruption.

Use for:

- COVID
- pandemic recurrence
- public-health restrictions
- vaccine mandates
- lockdown-type disruption

### 15. `OTHER_UNCLEAR`

Other or unclear.

Use this sparingly when:

- the row does not fit the codebook cleanly
- the text is too vague to place confidently
- the row mixes themes so evenly that no primary category dominates

In this project, `OTHER_UNCLEAR` is not a final resting place. It is a review signal: these rows should be examined in the recycle round and may be dropped from the final training set if they remain unstable.

## Boundary rules

### Budget versus execution

If the core issue is whether government demand, funding, or awards materialize, use `GOV_BUDGET_PROCUREMENT`.

If the award already exists and the risk is about delivering it profitably or on schedule, use `CONTRACT_EXECUTION_ECONOMICS`.

### Execution versus supply chain

If the row emphasizes internal performance, fixed-price exposure, estimates, warranty, or termination, use `CONTRACT_EXECUTION_ECONOMICS`.

If it emphasizes suppliers, shortages, lead times, manufacturing bottlenecks, or logistics, use `SUPPLY_CHAIN_INDUSTRIAL_BASE`.

### Cyber versus legal compliance

If the row is mainly about attacks, breaches, outages, or system compromise, use `CYBER_DATA_SYSTEMS`.

If it is mainly about regulatory obligations, audits, investigations, or legal exposure, use `LEGAL_REGULATORY_COMPLIANCE`.

### Technology versus IP

If the row is mainly about innovation, autonomy, obsolescence, or product competition, use `TECH_PRODUCT_AI_AUTONOMY`.

If it is mainly about patents, trade secrets, infringement, or protecting proprietary rights, use `IP_PROPRIETARY_RIGHTS`.

### Geopolitics versus export compliance

If the row centers on war, foreign instability, sanctions, international markets, or geopolitical exposure, use `GEOPOLITICAL_INTL_SANCTIONS_EXPORT`.

If it centers on compliance process, audits, or legal obligations in a broader sense, use `LEGAL_REGULATORY_COMPLIANCE`.

## When unsure

If two labels seem plausible, choose the one the row spends more space on.

If the row still feels genuinely unclear, choose `OTHER_UNCLEAR`. Disagreement handling and exclusion decisions happen later in the recycle and consensus process, not in the annotator interface.
