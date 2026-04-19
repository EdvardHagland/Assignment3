# Risk Coding Codebook

## Purpose

This codebook is for labeling risk-factor chunks from SEC `10-K Item 1A` disclosures for defense-sector firms.

Each chunk gets:

- one `primary_label`
- optionally one `secondary_label`
- optional notes when the chunk is ambiguous

The labels are intentionally broad. Broad labels are easier to code consistently and are better suited to a class project with multiple annotators.

## General coding rule

Choose the category that best captures the **main risk being foregrounded by the company**.

Do not code based on:

- whether the risk is true
- whether the company is exaggerating it
- whether you think another risk is more important

Code what the text is primarily **about**.

## Labels

### 1. GOVERNMENT_CONTRACTING

Use when the chunk is mainly about:

- dependence on government customers
- defense budgets and appropriations
- procurement cycles
- contract awards, renewals, cancellations, protests, or terminations
- program concentration
- fixed-price contract exposure

Examples:

- reliance on U.S. Department of Defense demand
- delays in appropriations
- termination for convenience
- concentration in a few major programs

### 2. GEOPOLITICS_INTERNATIONAL

Use when the chunk is mainly about:

- war
- conflict escalation
- foreign policy shifts
- international operations
- sanctions
- foreign military sales
- instability in overseas markets

Examples:

- effects of the war in Ukraine
- tensions in the Indo-Pacific
- foreign government demand shocks
- geopolitical disruptions to international business

### 3. SUPPLY_CHAIN_EXECUTION

Use when the chunk is mainly about:

- supplier disruption
- component shortages
- production delays
- quality failures
- manufacturing bottlenecks
- program execution risk
- dependence on sole-source suppliers

Examples:

- electronic component shortages
- schedule slippage
- cost overruns tied to execution
- inability to scale production

### 4. REGULATION_COMPLIANCE_EXPORT

Use when the chunk is mainly about:

- export controls
- ITAR / EAR-type restrictions
- compliance burdens
- investigations
- anti-corruption requirements
- environmental, labor, or disclosure regulation
- security clearance and classified-contract compliance

Examples:

- export-license restrictions
- government audits
- False Claims Act exposure
- failure to comply with procurement rules

### 5. CYBER_INFORMATION_SECURITY

Use when the chunk is mainly about:

- cyberattacks
- data breaches
- ransomware
- information security failures
- threats to digital infrastructure or networks
- compromise of sensitive government information

Examples:

- attacks on internal systems
- supplier cyber vulnerabilities
- classified or controlled-information exposure

### 6. BUSINESS_FINANCIAL_WORKFORCE

Use when the chunk is mainly about:

- inflation
- interest rates
- capital costs
- labor shortages
- retention
- pension exposure
- general macroeconomic pressure
- insurance or litigation costs when not clearly regulatory

Examples:

- inability to recruit engineers
- wage pressure
- margin pressure from inflation
- liquidity or financing constraints

## Secondary label rule

Add a `secondary_label` only when a second theme is clearly present and materially important.

Example:

- a chunk about export controls that is also strongly about geopolitics

If the second theme is weak, leave it blank.

## Hard cases

### Government customer + regulation

If the main issue is losing or changing government demand, use `GOVERNMENT_CONTRACTING`.

If the main issue is legal or compliance obligations, use `REGULATION_COMPLIANCE_EXPORT`.

### Geopolitics + supply chain

If the chunk says geopolitical conflict matters mainly because it disrupts inputs or production, choose `SUPPLY_CHAIN_EXECUTION` as primary and `GEOPOLITICS_INTERNATIONAL` as secondary when needed.

### Cyber + compliance

If the chunk is about attack exposure or technical compromise, choose `CYBER_INFORMATION_SECURITY`.

If it is mainly about failing to meet cyber-related regulatory requirements, consider `REGULATION_COMPLIANCE_EXPORT`.

## What to do when unsure

If two labels seem equally plausible:

1. choose the one the chunk spends more space on
2. use the other as `secondary_label`
3. leave a short note

If the chunk is too vague to code confidently, note the ambiguity for adjudication.
