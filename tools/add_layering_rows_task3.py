import pandas as pd
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    layering_path = root / "data" / "layering_compatibility.csv"
    layering = pd.read_csv(layering_path)

    extra_rows = [
        {"ingredient_1_inci":"Retinol","ingredient_2_inci":"Capryloyl Salicylic Acid","interaction_type":"conflicting","layering_order":"Retinol first","wait_time_minutes":0,"time_of_day":"PM","skin_type_oily":"caution","skin_type_dry":"no","skin_type_sensitive":"no","skin_type_combination":"caution","skin_type_mature":"caution","skin_concern_impact":"high acne-treatment overload risk","age_group":"adult","pregnancy_notes":"avoid retinoid in pregnancy","conflict_reason":"Retinoid plus LHA in one routine can over-exfoliate and impair barrier recovery.","synergy_reason":"none","application_notes":"Alternate nights and monitor irritation closely."},
        {"ingredient_1_inci":"Retinol","ingredient_2_inci":"Succinic Acid","interaction_type":"caution","layering_order":"Retinol first","wait_time_minutes":0,"time_of_day":"PM","skin_type_oily":"caution","skin_type_dry":"no","skin_type_sensitive":"no","skin_type_combination":"caution","skin_type_mature":"caution","skin_concern_impact":"blemish support with cumulative irritation","age_group":"adult","pregnancy_notes":"avoid retinoid in pregnancy","conflict_reason":"Retinoid with additional acid active increases dryness and erythema risk.","synergy_reason":"none","application_notes":"Use on alternating evenings and support barrier repair."},
        {"ingredient_1_inci":"Adapalene","ingredient_2_inci":"Glycolic Acid","interaction_type":"conflicting","layering_order":"Adapalene first","wait_time_minutes":0,"time_of_day":"PM","skin_type_oily":"caution","skin_type_dry":"no","skin_type_sensitive":"no","skin_type_combination":"caution","skin_type_mature":"caution","skin_concern_impact":"significant irritation and peeling risk","age_group":"teen|adult","pregnancy_notes":"avoid retinoid in pregnancy","conflict_reason":"AHA plus adapalene in same routine can exceed tolerance threshold for many users.","synergy_reason":"none","application_notes":"Separate routines or alternate nights only."},
        {"ingredient_1_inci":"Tretinoin","ingredient_2_inci":"Salicylic Acid","interaction_type":"conflicting","layering_order":"Tretinoin first","wait_time_minutes":0,"time_of_day":"PM","skin_type_oily":"caution","skin_type_dry":"no","skin_type_sensitive":"no","skin_type_combination":"caution","skin_type_mature":"caution","skin_concern_impact":"acne targeting with severe barrier stress","age_group":"adult","pregnancy_notes":"contraindicated in pregnancy","conflict_reason":"Prescription retinoid combined with BHA in one routine frequently causes intense irritation.","synergy_reason":"none","application_notes":"Strictly separate days and use clinician-directed protocol."},
        {"ingredient_1_inci":"Tretinoin","ingredient_2_inci":"Lactic Acid","interaction_type":"conflicting","layering_order":"Tretinoin first","wait_time_minutes":0,"time_of_day":"PM","skin_type_oily":"caution","skin_type_dry":"no","skin_type_sensitive":"no","skin_type_combination":"caution","skin_type_mature":"caution","skin_concern_impact":"texture benefit offset by high irritant load","age_group":"adult|mature","pregnancy_notes":"contraindicated in pregnancy","conflict_reason":"Strong retinoid plus AHA same-routine use can compromise barrier integrity.","synergy_reason":"none","application_notes":"Use separate nights with recovery-focused moisturizer."},
        {"ingredient_1_inci":"Copper Tripeptide-1","ingredient_2_inci":"Salicylic Acid","interaction_type":"conflicting","layering_order":"Copper Tripeptide-1 first","wait_time_minutes":0,"time_of_day":"PM","skin_type_oily":"yes","skin_type_dry":"caution","skin_type_sensitive":"no","skin_type_combination":"yes","skin_type_mature":"yes","skin_concern_impact":"repair pathway suppression with irritation","age_group":"adult|mature","pregnancy_notes":"consult both","conflict_reason":"Acidic BHA environment can reduce peptide stability and irritate skin.","synergy_reason":"none","application_notes":"Avoid same-routine pairing; separate by routine."},
        {"ingredient_1_inci":"Copper Tripeptide-1","ingredient_2_inci":"Capryloyl Salicylic Acid","interaction_type":"conflicting","layering_order":"Copper Tripeptide-1 first","wait_time_minutes":0,"time_of_day":"PM","skin_type_oily":"yes","skin_type_dry":"caution","skin_type_sensitive":"no","skin_type_combination":"yes","skin_type_mature":"yes","skin_concern_impact":"potential peptide deactivation","age_group":"adult|mature","pregnancy_notes":"consult both","conflict_reason":"LHA acidity may impair copper peptide function and increase stinging.","synergy_reason":"none","application_notes":"Use peptide and exfoliant on alternate days."},
        {"ingredient_1_inci":"Ascorbic Acid","ingredient_2_inci":"Capryloyl Salicylic Acid","interaction_type":"caution","layering_order":"Ascorbic Acid first","wait_time_minutes":20,"time_of_day":"AM","skin_type_oily":"caution","skin_type_dry":"caution","skin_type_sensitive":"no","skin_type_combination":"caution","skin_type_mature":"caution","skin_concern_impact":"brightening and pore care with irritation risk","age_group":"adult","pregnancy_notes":"consult both","conflict_reason":"Combined acidic actives may increase irritation and reduce daily tolerability.","synergy_reason":"none","application_notes":"Introduce one active at a time and separate if stinging occurs."},
        {"ingredient_1_inci":"Ascorbic Acid","ingredient_2_inci":"Succinic Acid","interaction_type":"caution","layering_order":"Ascorbic Acid first","wait_time_minutes":15,"time_of_day":"AM","skin_type_oily":"caution","skin_type_dry":"caution","skin_type_sensitive":"no","skin_type_combination":"caution","skin_type_mature":"caution","skin_concern_impact":"tone and blemish support with sensitivity trade-off","age_group":"adult","pregnancy_notes":"consult both","conflict_reason":"Acid-active stacking can trigger redness in compromised skin barriers.","synergy_reason":"none","application_notes":"Use reduced frequency and add barrier-supporting moisturizers."},
        {"ingredient_1_inci":"Niacinamide","ingredient_2_inci":"Capryloyl Salicylic Acid","interaction_type":"caution","layering_order":"Niacinamide first","wait_time_minutes":10,"time_of_day":"PM","skin_type_oily":"yes","skin_type_dry":"caution","skin_type_sensitive":"caution","skin_type_combination":"yes","skin_type_mature":"moderate","skin_concern_impact":"sebum and pores support with possible dryness","age_group":"teen|adult","pregnancy_notes":"generally acceptable","conflict_reason":"Usually compatible but LHA can increase dryness; sensitive users may react.","synergy_reason":"none","application_notes":"Apply niacinamide first and reduce exfoliant frequency if irritation appears."},
    ]

    existing_pairs = {
        tuple(sorted([str(a).strip().lower(), str(b).strip().lower()]))
        for a, b in zip(layering["ingredient_1_inci"], layering["ingredient_2_inci"])
    }
    to_add = []
    for row in extra_rows:
        key = tuple(sorted([row["ingredient_1_inci"].strip().lower(), row["ingredient_2_inci"].strip().lower()]))
        if key not in existing_pairs:
            to_add.append(row)
            existing_pairs.add(key)

    if to_add:
        layering = pd.concat([layering, pd.DataFrame(to_add)], ignore_index=True)
        layering.to_csv(layering_path, index=False)

    print({"added": len(to_add), "total": len(layering)})


if __name__ == "__main__":
    main()
