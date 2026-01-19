# XAI-NLG Framework
## Explicarea pe bază de prompt engineering a explicațiilor date de SHAP și LIME

### Raport Final de Implementare - Etapa 3

**Autori:** Toader Ana-Maria, Mereu Ioan-Flaviu, Arădoaie Ioana-Maria

**Data:** Ianuarie 2025

---

## 1. Introducere și Obiective

Proiectul XAI-NLG Framework transformă explicațiile tehnice generate de metodele SHAP (SHapley Additive exPlanations) și LIME (Local Interpretable Model-agnostic Explanations) în explicații în limbaj natural, accesibile utilizatorilor fără cunoștințe tehnice de machine learning.

**Obiective principale:**
- Integrarea metodelor XAI (SHAP și LIME) într-un pipeline unificat
- Generarea de explicații în limbaj natural folosind tehnici de prompt engineering
- Validarea automată a calității explicațiilor generate
- Suport pentru LLM-uri locale (Ollama) și remote (ReaderBench)

---

## 2. Arhitectura Sistemului

Framework-ul este organizat în **4 straturi** care procesează secvențial datele:

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Model ML + Instanță de explicat                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: Explainer (SHAP / LIME)                           │
│  - Generează contribuții numerice pentru fiecare feature    │
│  - SHAP: TreeExplainer pentru modele bazate pe arbori       │
│  - LIME: Aproximare locală cu model liniar interpretabil    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: Normalizer & Mapper                               │
│  - Normalizează contribuțiile în interval [0,1]             │
│  - Sortează features după importanță absolută               │
│  - Generează enunțuri descriptive pentru fiecare feature    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: NLG Generator                                     │
│  - Few-Shot: Exemple predefinite pentru ghidare             │
│  - Chain-of-Thought: Raționament pas cu pas                 │
│  - Self-Consistency: Agregare răspunsuri multiple           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4: Validator & Evidence Tracker                      │
│  - Verifică conservarea sumei SHAP                          │
│  - Calculează clarity score și coverage                     │
│  - Menține audit trail pentru trasabilitate                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: Explicație în limbaj natural + Metrici validare    │
└─────────────────────────────────────────────────────────────┘
```

**Structura codului:**
```
xai-nlg-framework/
├── config/
│   └── settings.py           # Configurări framework
├── src/
│   ├── explainer/            # SHAP și LIME explainers
│   ├── normalizer/           # Normalizare și mapping
│   ├── nlg/                  # Generatoare NLG + client Ollama
│   ├── validator/            # Validare și evidence tracking
│   └── pipeline.py           # Pipeline principal
├── examples/
│   └── breast_cancer_example.py
└── demos/                    # Jupyter notebooks demonstrative
```

---

## 3. Ce am adăugat față de versiunea anterioară

### 3.1 LIME Explainer - Extragere corectă a contribuțiilor

**Problema:** LIME returnează descrieri de tip `"516.45 < worst area <= 686.60"` în loc de nume simple de features, iar codul original folosea `as_map()` care returna indici ce nu se potriveau cu feature names.

**Soluția:** Am modificat metoda `explain()` să folosească `as_list()` și să parseze corect descrierile pentru a extrage numele features:

```python
exp_list = exp.as_list()
for description, weight in exp_list:
    for feature_name in self.feature_names:
        if feature_name in description:
            explanation[feature_name] = float(weight)
            break
```

### 3.2 Chain-of-Thought Generator - Prompt îmbunătățit

**Problema:** LLM-ul parafraza numele features (ex: "larger area" în loc de "worst area"), rezultând coverage 0%.

**Soluția:** Am adăugat instrucțiuni explicite în prompt:
```
CRITICAL RULES:
- You MUST use the EXACT feature names from the input
- Do NOT paraphrase or rename features
- The final explanation must mention AT LEAST the top 3 features by their exact names
```

### 3.3 Ollama Client - Suport ReaderBench

**Problema:** Codul original funcționa doar cu Ollama local.

**Soluția:** Am adăugat configurare pentru ReaderBench cu autentificare:
```python
DEFAULT_HOST_URL = "https://chat.readerbench.com/ollama"
DEFAULT_MODEL = "llama4:16x17b"
DEFAULT_API_KEY = "sk-56a239006a004929b080fd644a1f89ee"
```

### 3.4 Example actualizat

Am corectat `breast_cancer_example.py` pentru a folosi corect:
- `llm_call_fn=ollama_llm_call` în pipeline
- Structura corectă pentru validare: `result['validation']['clarity']['score']`
- Apelul corect pentru evidence tracker

---

## 4. Probleme întâmpinate și soluții

| Problemă | Cauză | Soluție |
|----------|-------|---------|
| LIME returnează contribuții 0.0 | `as_map()` returna indici greșiți | Folosire `as_list()` cu parsare descrieri |
| CoT coverage 0% | LLM parafraza feature names | Instrucțiuni explicite în prompt |
| ConnectionError Ollama | Server nu rula | Verificare `ollama serve` sau ReaderBench |
| KeyError 'clarity_score' | Structură validare schimbată | Acces `['clarity']['score']` |
| TypeError evidence tracker | Argumente greșite | Corectat semnătura `add_record()` |

---

## 5. Evaluarea rezultatelor

### 5.1 Metrici de evaluare

- **Clarity Score (0-100):** Bazat pe lungimea propozițiilor și complexitatea vocabularului
- **Coverage Score (0-1):** Procentul din top-5 features menționate în text
- **Sum Conservation:** Verificare proprietate SHAP: sum(contributions) + base_value ≈ prediction

### 5.2 Rezultate pe dataset Breast Cancer Wisconsin

**SHAP + Few-Shot:**
```
Clarity: 88.4 | Coverage: 100% | Valid: ✅
Explicație: "The model predicts a value of 1 based on several morphological 
features with positive SHAP contributions. The worst area, worst concave points, 
and mean concave points all show positive contributions..."
```

**LIME + Few-Shot:**
```
Clarity: 86.4 | Coverage: 40% | Valid: ✅
Explicație: "The model predicts a value of 1 primarily driven by positive 
contributions from size and texture features. The worst area, perimeter, 
and radius all show positive LIME contributions..."
```

**SHAP + Chain-of-Thought:**
```
Clarity: 92.5 | Coverage: 100% | Valid: ✅
Explicație: "The prediction of 1 is primarily driven by 'worst area', 
'worst concave points', and 'mean concave points', which all positively 
contribute to the outcome."
```

**SHAP + Self-Consistency:**
```
Clarity: 84.1 | Coverage: 100% | Valid: ✅
Explicație: "The prediction of 1 is primarily driven by tumor characteristics, 
with the worst area being a key factor... worst area, worst concave points, 
mean concave points, worst radius, and worst perimeter all support the prediction."
```

### 5.3 Tabel comparativ tehnici NLG

| Tehnică | Clarity | Coverage | Valid | Observații |
|---------|---------|----------|-------|------------|
| Few-Shot | 88.4 | 100% | ✅ | Cel mai consistent |
| Chain-of-Thought | 92.5 | 100% | ✅ | Clarity maxim |
| Self-Consistency | 84.1 | 100% | ✅ | Mai detaliat |

### 5.4 Comparație SHAP vs LIME

| Aspect | SHAP | LIME |
|--------|------|------|
| Top feature | worst area (0.0672) | worst area (0.0686) |
| Consistență | Foarte bună | Bună |
| Coverage explicații | 100% | 40-100% |
| Sum conservation | ✅ Verificabil | N/A |

---

## 6. Concluzii

Framework-ul XAI-NLG demonstrează cu succes transformarea explicațiilor tehnice SHAP și LIME în limbaj natural accesibil. 

**Puncte forte:**
- Arhitectură modulară pe 4 straturi
- Suport pentru 2 metode XAI (SHAP, LIME)
- 3 tehnici NLG cu rezultate consistente
- Validare automată cu metrici clare
- Flexibilitate LLM (local/remote)

**Limitări:**
- Testat doar pe date tabulare
- Coverage LIME mai scăzut uneori
- Dependent de calitatea LLM-ului

**Direcții viitoare:**
- Suport pentru date non-tabulare (imagini, text)
- Evaluare cu utilizatori reali
- Interfață web pentru demo interactiv
