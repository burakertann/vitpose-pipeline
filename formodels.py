from huggingface_hub import list_models

# ViTPose++ modelleri genellikle 'vitpose-plus' adıyla etiketlenir.
# Bu anahtar kelimeyle arama yapalım ve en çok indirilenlere göre sıralayalım.
vitpose_plus_iterator = list_models(
    search="vitpose-plus",
    sort="downloads",
    direction=-1  # Büyükten küçüğe sırala (en popüler en üstte)
)

print("--- Sadece ViTPose++ Modelleri ---")
print("Not: '++' genellikle model adlarında 'plus' olarak yazılır.\n")

found_any = False
# Bulunan modelleri daha okunaklı bir şekilde yazdır
for model in vitpose_plus_iterator:
    # Arama bazen alakasız sonuçlar getirebilir.
    # Bu yüzden model ID'sinde 'vitpose-plus' geçtiğinden emin olmak daha kesin sonuç verir.
    if 'vitpose-plus' in model.modelId:
        print(f"Model ID: {model.modelId:<50} | İndirme: {model.downloads:,}")
        found_any = True

if not found_any:
    print("Bu arama kriteriyle ViTPose++ modeli bulunamadı.")