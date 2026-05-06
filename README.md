# Refactored DenseNet-121 Training Pipeline

Refaktoryzowany kod dla modelu DenseNet-121 do klasyfikacji zdjęć RTG klatki piersiowej.

## Struktura plików

```
refactored_src/
├── __init__.py              # Package initialization
├── config.py                # Paths, constants, hyperparameters
├── data_loader.py           # Data loading and Dataset class
├── transforms.py            # Image transformations
├── checkpoint.py            # Checkpoint management
├── metrics.py               # Validation and metrics
├── training.py              # Training loop
├── model.py                 # Model initialization and creation
├── utils.py                 # Utility functions
├── main.py                  # Main training pipeline
└── README.md                # This file
```

## Opis modułów

### `config.py`
Zawiera wszystkie ścieżki, stałe i hiperparametry:
- Ścieżki do danych, checkpointów i wyników
- Lista chorób (labels)
- Parametry treningu (batch size, learning rates, liczba epok)
- Konfiguracja urządzenia (GPU/CPU)

### `data_loader.py`
Wczytywanie danych i klasa Dataset:
- Klasa `ChestXrayDataset` - PyTorch Dataset dla zdjęć RTG
- Funkcje do wczytywania list obrazów i danych z CSV
- Filtrowanie DataFrame dla train/val setów

### `transforms.py`
Transformacje obrazów:
- `train_transforms` - augmentacja danych dla treningu
- `val_transforms` - czyszczenie danych dla walidacji

### `checkpoint.py`
Zarządzanie checkpointami:
- `save_checkpoint()` - zapisanie stanu modelu
- `load_checkpoint()` - wczytanie stanu modelu

### `metrics.py`
Walidacja i metryki:
- `validate_model()` - walidacja z danym threshold
- `validate_epoch_threshold_range()` - walidacja dla zakresu thresholdów

### `training.py`
Pętla treningowa:
- `train_model()` - główna funkcja treningowa z zapisem wyników

### `model.py`
Inicjalizacja modelu:
- `create_densenet121_model()` - tworzenie modelu z pretrenowanymi wagami
- Funkcje do zamrażania/odmrażania wag
- `get_class_weights()` - obliczanie wag dla niezbalansowanego datasetu

### `utils.py`
Funkcje pomocnicze:
- Tworzenie data loaderów
- Tworzenie loss function, optimizera, schedulera
- Tworzenie wymaganych katalogów

### `main.py`
Główny pipeline:
- Wczytywanie danych
- Inicjalizacja modelu
- Konfiguracja optimizacji
- Trenowanie (faza 1 i 2 - opcjonalnie)
- Walidacja

## Jak uruchomić

```bash
# Z folderu refactored_src
python main.py

# Lub z głównego folderu projektu
python refactored_src/main.py
```

## Strukturalne różnice od oryginału

| Aspekt | Oryginalny | Refaktoryzowany |
|--------|-----------|-----------------|
| **Organizacja** | Jeden duży plik | Wiele modułów |
| **Reusability** | Trudne | Łatwe - każdy moduł niezależny |
| **Testowanie** | Wyzwaniem | Proste - można testować moduły |
| **Czytelność** | Niska | Wysoka |
| **Maintenance** | Trudne | Łatwe |

## Główne funkcjonalności

✓ Wczytywanie danych z CSV i zdjęć  
✓ Data augmentation dla treningu  
✓ Multi-label classification  
✓ Transfer learning (DenseNet-121)  
✓ Klasy wagi dla niezbalansowanego datasetu  
✓ Checkpoint management  
✓ Walidacja z różnymi thresholdami  
✓ Learning rate scheduling  
✓ Zapisywanie metryk do JSON  

## Uwagi

- Kod zachowuje wszystkie funkcjonalności oryginału
- Konfiguracja odbywa się przez `config.py`
- Fazy treningu (1 i 2) są domyślnie wyłączone - odkomentuj w `main.py`
- Kompatybilny z GPU i CPU
