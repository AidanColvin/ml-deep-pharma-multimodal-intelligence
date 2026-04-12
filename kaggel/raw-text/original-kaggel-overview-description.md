# Deep-Pharma Multimodal Intelligence - DataHack 3.0

Predict drug interaction severity & side effects from molecular structures (SMILES).

---

# Deep-Pharma Multimodal Intelligence - DataHack 3.0

## Overview

💊 **Deep Pharma Challenge 2026: Drug-Drug Interaction Prediction**

Can your model save lives by predicting dangerous drug combinations before they are prescribed?

---

## Context

Modern medicine relies on **polypharmacy**—prescribing multiple drugs to treat complex conditions. However, this comes with a hidden cost: **Drug-Drug Interactions (DDIs)**.

- 30% of adverse drug events are caused by DDIs  
- $30 Billion+ is lost annually in healthcare costs due to preventable interactions  

### The Gap

Clinical trials cannot test every possible pair of drugs. We rely on **post-market surveillance (Pharmacovigilance)** to find these risks, often too late.

---

## The Mission

Your goal is to build a **Multi-Task Machine Learning Model** that predicts the safety profile of drug pairs.

You must predict:

- If an interaction occurs  
- How severe it is  
- What specific side effects will happen  
- How likely they are compared to the general population  

---

## The Tasks

This is a **triple-prediction challenge**. For every Drug Pair (A + B), you must output:

### 1. Severity Classification
- Is the interaction:
  - Minor  
  - Moderate  
  - Major  
  - Contraindication  

### 2. Side Effect Prediction (Binary)
- Which of the 50 specific adverse events (e.g., Nausea, Arrhythmia) will occur  

### 3. Risk Quantification (Regression)
- What is the **Proportional Reporting Ratio (PRR)** for each side effect  
- A statistical signal measure used by the FDA  

---

## The Data

You are provided with a **rich, multimodal dataset** derived from real FDA Adverse Event Reporting System (FAERS) data.

### Modalities

- **Chemistry**  
  - SMILES strings representing molecular structure  

- **Pharmacology**  
  - Mechanisms of Action (MoA)  
  - Metabolism  
  - Transporters  

### Scale

- Training Set: ~15,000 confirmed interacting pairs  
- Test Set: ~4,000 held-out pairs (Cold Start / New Scaffolds)  

---

## Why Join?

### Real-World Impact
Your code could be the foundation for next-gen **Clinical Decision Support Systems (CDSS)**  

### Scientific Discovery
Help uncover hidden patterns between molecular structure and biological toxicity  

---

## Timeline

- Start: 2 months ago  
- Close: 10 months to go  

---

# Evaluation

## 📊 Evaluation Metric

This competition uses a **Hardcore Clinical Score** designed to rigorously test your model’s ability to detect rare signals.

- Penalizes "lazy" predictions (predicting nothing)  
- Rewards precision in risk estimation  
- Final score range: **0.0 to 1.0 (higher is better)**  

---

## 🏆 Final Score Formula

*(Composite metric combining all three tasks)*

---

## Detailed Breakdown

### 1. Severity Classification (40%)

- Metric: **Macro F1-Score**  
- Description: Multi-class classification (Minor, Moderate, Major)  

**Why Macro F1?**
- Treats all classes equally  
- Ensures rare but critical cases (e.g., Contraindication) are not ignored  

---

### 2. Side Effect Prediction (30%)

- Metric: **Micro F1-Score (Global)**  
- Description: Binary multi-label classification across 50 side effects  

**Why Micro F1?**
- Does NOT reward predicting "nothing"  
- Rewards detecting actual side effects (true positives)  

---

### 3. PRR Regression (30%)

- Metric: **Inverse RMSE on Masked Data**

#### Description

- RMSE is calculated between predicted PRR and true PRR  
- Only evaluated where **True PRR > 0**  

#### Masked Rule

- No penalty for predicting PRR on non-existent side effects  
- No reward for correctly predicting zeros  

#### Why Inverse RMSE?

- Penalizes large errors heavily  
- Encourages precise risk estimation  
- Perfect model = 1.0  
- High error → score approaches 0  

---

# Submission Format

You must submit a CSV file with predictions for all test rows.

## Example
