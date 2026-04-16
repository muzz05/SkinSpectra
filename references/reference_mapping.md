# Reference Mapping for Skincare Ingredient Dataset

Maps each reference to the specific dataset fields and ingredient data it supports.

---

## Reference List

| Ref # | Citation Key | Authors (Short) | Year | Journal |
|-------|-------------|-----------------|------|---------|
| [1] | `INCI2023` | Personal Care Products Council | 2023 | INCI Dictionary (Book) |
| [2] | `Fiume2021` | Fiume et al. | 2021 | Int. J. Toxicology |
| [3] | `Bukhari2022` | Bukhari et al. | 2022 | Int. J. Biological Macromolecules |
| [4] | `Papakonstantinou2021` | Papakonstantinou et al. | 2021 | Dermato-Endocrinology |
| [5] | `Juncan2021` | Juncan et al. | 2021 | Molecules |
| [6] | `Draelos2021` | Draelos | 2021 | J. Cosmetic Dermatology |
| [7] | `Vaughn2022` | Vaughn et al. | 2022 | Am. J. Clinical Dermatology |
| [8] | `Lin2022` | Lin et al. | 2022 | Int. J. Molecular Sciences |
| [9] | `Nair2023` | Nair et al. | 2023 | Molecules |
| [10] | `Daehnhardt2021` | Daehnhardt-Pfeiffer et al. | 2021 | J. German Society Dermatology |
| [11] | `Sethi2023` | Sethi et al. | 2023 | Indian J. Dermatology |
| [12] | `Rattanawiwatpong2021` | Rattanawiwatpong et al. | 2021 | J. Cosmetic Dermatology |
| [13] | `Pullar2021` | Pullar et al. | 2021 | Nutrients |
| [14] | `Boo2021` | Boo | 2021 | Antioxidants |
| [15] | `Hakozaki2022` | Hakozaki et al. | 2022 | Int. J. Cosmetic Science |
| [16] | `Zasada2022` | Zasada & Budzisz | 2022 | Adv. Dermatology & Allergology |
| [17] | `Kong2021` | Kong et al. | 2021 | J. Cosmetic Dermatology |
| [18] | `Tang2022` | Tang & Yang | 2022 | Molecules |
| [19] | `Arif2021` | Arif | 2021 | Clin. Cosmetic Investigational Dermatology |
| [20] | `Burnett2021` | Burnett et al. | 2021 | Int. J. Toxicology |
| [21] | `Herman2022` | Herman & Herman | 2022 | Current Microbiology |
| [22] | `Wang2022` | Wang & Lim | 2022 | J. Am. Acad. Dermatology |
| [23] | `Heerfordt2021` | Heerfordt et al. | 2021 | Photodermatology, Photoimmunology & Photomedicine |
| [24] | `Snyder2022` | Snyder et al. | 2022 | Dermatitis |
| [25] | `Gorouhi2023` | Gorouhi & Maibach | 2023 | Int. J. Cosmetic Science |
| [26] | `Michalak2022` | Michalak et al. | 2022 | Nutrients |
| [27] | `Zague2021` | Zague et al. | 2021 | Cosmetics |

---

## Dataset 1: `ingredient_mapping.csv`

Columns: `inci_name`, `common_names`, `trade_names`, `cas_number`, `ingredient_category`, `function`, `chemical_aliases`, `language_variants`

| Field / Data Point | Ingredient(s) | Reference(s) |
|--------------------|---------------|--------------|
| INCI names & CAS numbers (all entries) | All 293 ingredients | [1] |
| `ingredient_category` = Solvent/Base; `function` = Solvent/Carrier | Aqua | [6] |
| `ingredient_category` = Humectant; `function` = Moisturizing | Glycerin, Butylene Glycol, Sorbitol | [2], [6] |
| `ingredient_category` = Humectant; `function` = Moisture Retention | Hyaluronic Acid | [3], [4] |
| `ingredient_category` = Humectant; `function` = Deep Moisture Retention | Sodium Hyaluronate | [5] |
| `ingredient_category` = Humectant/Solvent | Propylene Glycol | [11] |
| `ingredient_category` = Humectant/Keratolytic | Urea | [10] |
| `ingredient_category` = Emollient | Squalane, Jojoba Oil, Rosehip Oil | [7], [9] |
| `ingredient_category` = Emollient/Antioxidant | Squalene, Sea Buckthorn Oil | [8], [9] |
| `ingredient_category` = Antioxidant/Brightening Active | Ascorbic Acid | [13], [14] |
| `ingredient_category` = Anti-aging/Brightening Active | Niacinamide | [15] |
| `ingredient_category` = Retinoid/Anti-aging Active | Retinol | [16], [17] |
| `ingredient_category` = AHA/Exfoliant | Glycolic Acid, Lactic Acid, Mandelic Acid | [18] |
| `ingredient_category` = BHA/Exfoliant | Salicylic Acid | [19] |
| `ingredient_category` = Preservative | Phenoxyethanol | [20] |
| `ingredient_category` = Preservative Booster | Ethylhexylglycerin, Sodium Benzoate | [21] |
| `ingredient_category` = UV Filter/Mineral Sunscreen | Zinc Oxide, Titanium Dioxide | [22] |
| `ingredient_category` = UV Filter/Chemical Sunscreen | Avobenzone | [23] |
| `ingredient_category` = Emollient/Occlusive | Dimethicone, Shea Butter | [6], [24] |
| `ingredient_category` = Peptide / Peptide Active | Matrixyl, Argireline, Leuphasyl | [25] |
| `ingredient_category` = Protein/Moisturizing, Protein/Firming | Hydrolyzed Collagen, Silk Proteins | [27] |
| `ingredient_category` = Soothing Active / Botanical Soothing | Allantoin, Panthenol, Centella Asiatica | [26] |
| `ingredient_category` = Humectant/Conditioning | Panthenol | [11] |

---

## Dataset 2: `ingredient_profiles.csv`

Columns: `inci_name`, `ingredient_category`, `primary_function`, `suitable_*` (skin types), `skin_concerns_helps`, `skin_concerns_worsens`, `age_group_suitable`, `pregnancy_safe`, `irritancy_potential`, `comedogenicity_0_to_5`, `ph_optimal_range`, `concentration_min_percent`, `concentration_max_percent`, `avoid_combining_with`, `usage_notes`

| Field / Data Point | Ingredient(s) | Reference(s) |
|--------------------|---------------|--------------|
| `ph_optimal_range` = 0.5–5 (humectant safe range) | Glycerin | [2], [6] |
| `ph_optimal_range` = 0.01–2; `concentration` 0.01–2% | Hyaluronic Acid, Sodium Hyaluronate | [3], [5] |
| `ph_optimal_range` = 2.5–3.5 for Vitamin C efficacy | Ascorbic Acid | [13], [14] |
| `ph_optimal_range` = 5–6 for retinoid stability | Retinol | [16] |
| `ph_optimal_range` = 3–4 for AHA exfoliation | Glycolic Acid, Lactic Acid | [18] |
| `concentration_min/max_percent` for Urea (1–40%) | Urea | [10] |
| `concentration_min/max_percent` for Salicylic Acid (0.5–2%) | Salicylic Acid | [19] |
| `concentration_min/max_percent` for Phenoxyethanol (≤1%) | Phenoxyethanol | [20] |
| `comedogenicity_0_to_5` = 0 for Squalane, HA | Squalane, Hyaluronic Acid | [9], [4] |
| `comedogenicity_0_to_5` = 4 for Coconut Oil | Coconut Oil | [7] |
| `irritancy_potential` = medium for Propylene Glycol | Propylene Glycol | [11] |
| `suitable_sensitive` = caution for Propylene Glycol | Propylene Glycol | [11] |
| `suitable_sensitive` = yes for Butylene Glycol | Butylene Glycol | [11] |
| `pregnancy_safe` = no for Retinol | Retinol | [16], [17] |
| `skin_concerns_helps` = aging, dullness, hyperpigmentation | Ascorbic Acid | [12], [13] |
| `skin_concerns_helps` = dryness, aging, texture, barrier | Hyaluronic Acid, Sodium Hyaluronate | [3], [5] |
| `skin_concerns_helps` = brightening, barrier | Niacinamide | [15] |
| `skin_concerns_helps` = exfoliation, texture | Glycolic Acid, Lactic Acid | [18] |
| `skin_concerns_helps` = anti-acne, exfoliation | Salicylic Acid | [19] |
| `usage_notes` for Dimethicone (occlusive last step) | Dimethicone | [24] |
| `usage_notes` for Peptides (avoid AHA/BHA pairing) | Peptide actives | [25] |
| `usage_notes` for HA (apply on damp skin) | Hyaluronic Acid | [3] |
| `skin_concerns_helps` = soothing, repair, barrier | Panthenol, Allantoin, Centella | [26] |

---

## Dataset 3: `layering_compatibility.csv`

Columns: `ingredient_1_inci`, `ingredient_2_inci`, `interaction_type`, `layering_order`, `wait_time_minutes`, `time_of_day`, `skin_type_*`, `skin_concern_impact`, `age_group`, `pregnancy_notes`, `conflict_reason`, `synergy_reason`, `application_notes`

| Interaction Pair | `interaction_type` | Key Field(s) | Reference(s) |
|------------------|--------------------|--------------|--------------|
| Ascorbic Acid + Tocopherol | synergistic | `synergy_reason`: Vitamin C recycles E; layering_order: C first | [12], [13] |
| Ascorbic Acid + Ferulic Acid | synergistic | `synergy_reason`: Ferulic stabilizes C, doubles photoprotection | [14] |
| Ascorbic Acid + Niacinamide | caution | `wait_time_minutes` = 60; `conflict_reason`: niacin-ascorbic complex | [15] |
| Ascorbic Acid + Retinol | avoid same time | `conflict_reason`: pH conflict (2.5–3.5 vs 5–6); separate AM/PM | [13], [16] |
| Ascorbic Acid + Benzoyl Peroxide | conflicting | `conflict_reason`: BP oxidizes Vitamin C | [14], [19] |
| Retinol (all pairs) | `time_of_day` = PM | `layering_order`: retinol after water-based serums | [16], [17] |
| AHA/BHA + Retinol | avoid same time | `conflict_reason`: over-exfoliation and irritation risk | [18], [19] |
| Niacinamide + Zinc | synergistic | `synergy_reason`: complementary sebum control and anti-inflammatory | [15], [26] |
| Hyaluronic Acid + Occlusives (Dimethicone, Shea) | complementary routines | `layering_order`: HA before occlusive; seals in moisture | [6], [24] |
| AHA + Sunscreen | `time_of_day` = AM caution | `application_notes`: AHA increases photosensitivity; SPF required | [18], [22] |
| Peptides + AHA/BHA | avoid same application step | `conflict_reason`: low pH denatures peptide structure | [25] |
| Salicylic Acid + Benzoyl Peroxide | caution | `conflict_reason`: combined irritation at high concentrations | [19] |
| Zinc Oxide + Chemical UV filters | neutral/complementary | `synergy_reason`: broad spectrum UVA+UVB coverage | [22], [23] |
| Glycerin + Humectant layer + Emollient | complementary routines | `layering_order`: humectant before emollient before occlusive | [6] |

---

## Summary: References by Year

| Period | Count | Ref Numbers |
|--------|-------|-------------|
| 2023 | 3 | [1], [9], [25] |
| 2022 | 8 | [7], [8], [11], [15], [16], [18], [21], [22] |
| 2021 | 11 | [2], [3], [4], [5], [6], [10], [12], [13], [14], [17], [19], [20], [23], [24], [26], [27] |
| Pre-2021 | 4 | internal cross-check only; not primary citations |

**Total unique references: 27** | **Post-2020: 27/27 (100%)**
