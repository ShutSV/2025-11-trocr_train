from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_NAME = "microsoft/trocr-small-handwritten"
# MODEL_NAME = "raxtemur/trocr-base-ru"
processor = TrOCRProcessor.from_pretrained(MODEL_NAME, use_fast=False)

#=====================================================================
test_text = "Привет, мир!"
tokens = processor.tokenizer.encode(test_text)
print(processor.tokenizer.convert_ids_to_tokens(tokens))
# Вы должны увидеть буквы, а не [UNK]

test_text = "Hello, World!"
tokens = processor.tokenizer.encode(test_text)
print(processor.tokenizer.convert_ids_to_tokens(tokens))
# Вы должны увидеть буквы, а не [UNK]
#=====================================================================




# 2. Определяем алфавит (добавьте нужные вам символы: ё, спецзнаки и т.д.)
cyrillic_alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
special_chars = ".,!?- "
all_new_tokens = list(cyrillic_alphabet + special_chars)

# 3. Добавляем токены в словарь
num_added_toks = processor.tokenizer.add_tokens(all_new_tokens)
print(f"Добавлено токенов: {num_added_toks}")

# # 4. ВАЖНО: Модель тоже должна узнать о новых токенах
# model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
# model.decoder.resize_token_embeddings(len(processor.tokenizer))




#=====================================================================
test_text = "Привет, мир!"
tokens = processor.tokenizer.encode(test_text)
print(processor.tokenizer.convert_ids_to_tokens(tokens))
# Вы должны увидеть буквы, а не [UNK]

test_text = "Hello, World!"
tokens = processor.tokenizer.encode(test_text)
print(processor.tokenizer.convert_ids_to_tokens(tokens))
# Вы должны увидеть буквы, а не [UNK]
#=====================================================================

"""
# 1. Основная кириллица (регистр важен!)
cyrillic_ru = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

# 2. Дореволюционные символы (если архив старый, до 1918 г.)
# ять, фита, ижица, десятеричное И
cyrillic_old = "ѣѢѳѲѵѴіІ" 

# 3. Расширенная пунктуация для рукописей
# Включает разные виды тире, кавычек и спецзнаков
punctuation = (
    ".,!?;:()[]{}<> "  # Базовые
    "/\\|@#$%^&*+_="   # Технические
    "\"'«»„“"           # Кавычки (в рукописях часто встречаются „нижние“)
    "—–-"               # Длинное тире, среднее и дефис
    "№§¶°"              # Специфические архивные знаки
    "…"                 # Многоточие (единым символом)
)

# 4. Цифры
digits = "0123456789"

# Сборка финального списка
all_new_tokens = list(set(cyrillic_ru + cyrillic_old + punctuation + digits))
"""