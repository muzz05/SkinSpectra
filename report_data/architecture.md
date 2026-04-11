User Input (Ingredients / Image / Text)
        |
        +-- [Image Path] ---> OCR Handler (Tesseract) ---> Raw Ingredient Text
        |                                                    |
        +-- [Face Photo] ---> Facial Analysis (EfficientNetV2-B2) ---> Skin Type
        |                                                    |
        +-- [Text Input] ----------------------------------->|
                                                             v
                                             NLP Layer (sentence-transformers + FAISS)
                                             Normalizes -> INCI Standard Names
                                                             |
                                     +-----------------------+----------------------+
                                     v                                              v
                              Individual Scoring                            Layering Scoring
                              (XGBoost Regressor)                           (LightGBM Regressor)
                              Rule Engine + ML                              Rule Engine + ML
                              Score: 0-100                                  Score: 0-100
                                     |                                              |
                                     +-----------------------+----------------------+
                                                             v
                                              LLM Layer (Gemini 2.5 Flash)
                                              Personalized JSON Report
                                                             |
                                                             v
                                          FastAPI Response -> Frontend UI
