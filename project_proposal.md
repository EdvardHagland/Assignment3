# Project Proposal

## Title

**Corporate Anxiety After 2022: Risk Narratives in Defense-Sector Filings**

## Research Question

How did U.S. defense-sector firms change the way they described risk after 2022?

## Corpus

The project analyzes **Item 1A. Risk Factors** from SEC **10-K** filings. This section is useful because firms are required to disclose their most significant risks there, which gives the corpus a clear internal logic. Instead of using whole annual reports, we focus only on this structured risk section.

The current corpus covers ten major defense or defense-adjacent firms:

- Lockheed Martin
- RTX
- Northrop Grumman
- General Dynamics
- L3Harris
- Huntington Ingalls
- Leidos
- Booz Allen Hamilton
- Kratos
- AeroVironment

Time coverage is **2018-2026**, allowing a comparison between pre-2022 and post-2022 disclosures. The extracted dataset currently includes **87 filings** and about **8,400 annotation-ready text units**.

## Aim

The goal is not to discover what firms privately believe, but to analyze how they publicly frame uncertainty. The defense sector is especially interesting because its disclosures sit at the intersection of war, geopolitics, procurement, regulation, supply chains, cybersecurity, and public spending. We want to see whether the language of risk changes after 2022 and which types of risk become more prominent.

## Method

The workflow has three steps.

First, we collect the filings from the SEC EDGAR system and extract only the risk-factor section. The text is cleaned so that duplicated summaries and generic boilerplate are removed.

Second, the group will manually annotate a subset of the corpus using broad categories such as:

- government contracting and public spending
- geopolitics and international conflict
- regulation and compliance
- supply chain and production
- cybersecurity and information security
- business, financial, and workforce pressure

Because the project is collaborative, part of the labeled sample will be coded by multiple group members in order to compare disagreement and refine the codebook.

Third, we will train a supervised classifier on the labeled sample and apply it to the full corpus. This will let us compare the distribution of risk categories across firms and across time, especially before and after 2022. Representative passages will then be used for interpretation.

## Expected Result

We expect the project to show that corporate risk disclosures are not only legal boilerplate, but also a useful corpus for studying how firms narrate strategic uncertainty. More specifically, the analysis should identify which risks dominate defense-sector disclosures, whether these emphases shift after 2022, and how firms differ in the way they frame geopolitical, regulatory, technological, and economic exposure.

## Relevance

This project fits the course because it combines corpus collection, text segmentation, collaborative annotation, and classification. It is also feasible within the assignment timeline because the corpus is already structured, the source is official, and the unit of analysis is clearly defined by the filing format.
