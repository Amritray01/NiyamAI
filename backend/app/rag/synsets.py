from __future__ import annotations

from typing import Dict, List, Set


class FSSAISynsetRegistry:
    """
    FSSAI Food Category Synset & Legal Regulatory Glossary.
    Maps colloquial culinary/trade names to statutory gazette categories,
    and maps chemical/contaminant terms to relevant FSSAI regulation chapters.
    """

    FOOD_CATEGORY_MAP: Dict[str, Dict[str, any]] = {
        "paneer": {
            "statutory_terms": [
                "chhana", "paneer", "heat-acid coagulated milk product",
                "dairy products", "milk fat", "moisture in paneer"
            ],
            "fssai_category": "Dairy Products",
            "regulation_ref": "Food Safety and Standards (Food Products Standards and Food Additives) Regulations"
        },
        "ghee": {
            "statutory_terms": [
                "ghee", "clarified butterfat", "milk fat", "vegetable fat in ghee",
                "prohibition of adulterated ghee", "Baudouin test", "Reichert Meissl value"
            ],
            "fssai_category": "Prohibitions",
            "regulation_ref": "Prohibition and Restrictions on Sales Regulations"
        },
        "milk": {
            "statutory_terms": [
                "milk", "unpasteurized milk", "standardized milk", "toned milk",
                "added water in milk", "adulteration of milk", "pasteurized milk"
            ],
            "fssai_category": "Prohibitions",
            "regulation_ref": "Prohibition and Restrictions on Sales Regulations"
        },
        "honey": {
            "statutory_terms": [
                "honey", "natural honey", "blossom honey", "caravia honey",
                "stingless bee honey", "trigona species", "moisture in honey",
                "sucrose content in honey", "SMR test", "pollen count"
            ],
            "fssai_category": "Sweetening Agents & Standards",
            "regulation_ref": "Chapter 2.8 Sweetening Agents including Honey"
        },
        "sugar": {
            "statutory_terms": [
                "refined sugar", "white sugar", "sucrose by mass", "sugar syrup",
                "invert sugar", "total soluble solids", "jaggery", "gur"
            ],
            "fssai_category": "Sweetening Agents & Standards",
            "regulation_ref": "Chapter 2.8 Sweetening Agents"
        },
        "juice": {
            "statutory_terms": [
                "fruit juice", "vegetable juice", "tomato juice", "nectar",
                "concentrated fruit juice", "thermally processed fruit beverages"
            ],
            "fssai_category": "Contaminants",
            "regulation_ref": "Contaminants, Toxins and Residues Regulations"
        },
        "beverage": {
            "statutory_terms": [
                "beverages", "soft drinks", "carbonated water", "concentrated soft drinks",
                "canned carbonated beverages", "soldered cans"
            ],
            "fssai_category": "Contaminants",
            "regulation_ref": "Contaminants, Toxins and Residues Regulations"
        },
    }

    CONTAMINANTS_MAP: Dict[str, Dict[str, any]] = {
        "lead": {
            "statutory_terms": [
                "lead", "metal contaminants", "Regulation 2.1", "Table 1",
                "ppm", "parts per million", "infant milk food", "canned food"
            ],
            "category": "Contaminants",
            "clause": "Regulation 2.1 Metal Contaminants"
        },
        "arsenic": {
            "statutory_terms": [
                "arsenic", "metal contaminants", "Regulation 2.1", "fruit juices", "0.2 mg/kg"
            ],
            "category": "Contaminants",
            "clause": "Regulation 2.1 Metal Contaminants"
        },
        "aflatoxin": {
            "statutory_terms": [
                "aflatoxin", "aflatoxin B1", "crop contaminants", "Regulation 2.2",
                "mycotoxins", "naturally occurring toxic substances", "ppb", "µg/kg"
            ],
            "category": "Contaminants",
            "clause": "Regulation 2.2 Crop Contaminants"
        },
        "pesticide": {
            "statutory_terms": [
                "pesticide residue", "MRL", "maximum residue limit",
                "Regulation 2.3", "agricultural food products"
            ],
            "category": "Contaminants",
            "clause": "Regulation 2.3 Pesticide Residues"
        },
        "advertis": {
            "statutory_terms": [
                "advertising and claims regulations", "Clause 4 nutritional claims", "no added sugar conditions",
                "sucrose fructose glucose dextrose maltose honey syrups", "sugar alcohols sorbitol maltitol xylitol erythritol",
                "sugar-containing concentrates", "contains naturally occurring sugars disclaimer", "Section 24 misleading advertisements"
            ],
            "category": "Advertising & Claims",
            "clause": "Advertising and Claims Regulations"
        },
        "claim": {
            "statutory_terms": [
                "nutritional claims", "no added sugar claim conditions", "100% natural",
                "sucrose fructose glucose dextrose maltose", "honey and fruit syrups concentrates",
                "sugar alcohols added as sweeteners", "2022 compendium amendment", "scientific validation data"
            ],
            "category": "Advertising & Claims",
            "clause": "Advertising and Claims Regulations"
        },
        "sugar": {
            "statutory_terms": [
                "no added sugar claims", "sucrose fructose glucose dextrose maltose", "honey syrups and fruit concentrates",
                "sugar alcohols polyols sorbitol maltitol erythritol xylitol", "invert sugar cane sugar solids",
                "contains naturally occurring sugars disclaimer", "FBO legal compliance Section 53 penalty"
            ],
            "category": "Advertising & Claims",
            "clause": "Advertising and Claims Regulations & Chapter 2.8"
        },
        "sweetener": {
            "statutory_terms": [
                "non-nutritive sweeteners", "sugar alcohols", "polyols", "sorbitol", "maltitol",
                "xylitol", "erythritol", "isomalt", "steviol glycosides", "mandatory sweetener declarations"
            ],
            "category": "Advertising & Claims",
            "clause": "Food Safety Standards (Food Products Standards and Food Additives)"
        }
    }

    @classmethod
    def expand_query(cls, query: str) -> List[str]:
        """
        Dynamically enriches the user query with statutory legal glossary terms,
        exhaustive ingredient definitions (sugars, polyols, syrups), FBO legal angles,
        and medical/nutritional contexts.
        """
        q_lower = query.lower()
        queries: List[str] = [query]
        expanded_keywords: Set[str] = set()

        # Specific No Added Sugar / Claim ingredient expansion
        if any(w in q_lower for w in ["sugar", "no added sugar", "claim", "label", "sweetener", "ingredient"]):
            sugar_ingredients = [
                "sucrose fructose glucose dextrose maltose",
                "honey syrups sugar-containing concentrates",
                "sugar alcohols sorbitol maltitol erythritol xylitol",
                "contains naturally occurring sugars disclaimer",
                "Clause 4 Schedule II Advertising Claims"
            ]
            for si in sugar_ingredients:
                expanded_keywords.add(si)

        for term, meta in cls.FOOD_CATEGORY_MAP.items():
            if term in q_lower:
                for stat_term in meta.get("statutory_terms", [])[:3]:
                    if stat_term not in q_lower:
                        expanded_keywords.add(stat_term)

        for term, meta in cls.CONTAMINANTS_MAP.items():
            if term in q_lower:
                for stat_term in meta.get("statutory_terms", [])[:3]:
                    if stat_term not in q_lower:
                        expanded_keywords.add(stat_term)

        if expanded_keywords:
            enriched = f"{query} {' '.join(list(expanded_keywords)[:5])}"
            queries.append(enriched)
            # Add an FBO compliance-specific expansion
            queries.append(f"{query} FBO statutory compliance mandatory labeling disclaimer legal risks")

        return queries

    @classmethod
    def infer_category_hint(cls, query: str) -> str | None:
        q_lower = query.lower()
        for term, meta in cls.CONTAMINANTS_MAP.items():
            if term in q_lower:
                return meta.get("category")
        for term, meta in cls.FOOD_CATEGORY_MAP.items():
            if term in q_lower:
                return meta.get("fssai_category")
        return None
